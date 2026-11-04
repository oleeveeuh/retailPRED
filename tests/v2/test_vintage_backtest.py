from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from forecasting.evaluation.vintage import (
    availability_as_of,
    vintage_panel_backtest,
)
from forecasting.models.base import Forecaster
from forecasting.models.panel_adapter import PerSeriesPanelAdapter
from forecasting.models.seasonal_naive import SeasonalNaive

SERIES = ("alpha", "beta", "gamma")
START = "2015-01-01"
MONTHS = 84


def _true_value(series: str, month_index: int) -> float:
    seasonal = 1.0 + 0.05 * np.sin(2 * np.pi * month_index / 12)
    scale = {"alpha": 100.0, "beta": 60.0, "gamma": 20.0}[series]
    drift = 0.15 * month_index
    return float(scale * seasonal + drift)


def _versioned_panel() -> pd.DataFrame:
    """Preliminary values land 15 days after month end; revisions after 45 days."""
    months = pd.date_range(START, periods=MONTHS, freq="MS")
    records = []
    for month_index, month in enumerate(months):
        month_end = month + pd.offsets.MonthEnd(0)
        for series in SERIES:
            truth = _true_value(series, month_index)
            records.append(
                {
                    "series_id": series,
                    "observation_time": month,
                    "value": round(truth + 1.0, 6),
                    "release_time": pd.NaT,
                    "vintage_time": (month_end + pd.Timedelta(days=15)).tz_localize("UTC"),
                    "ingested_at": (month_end + pd.Timedelta(days=16)).tz_localize("UTC"),
                    "source": "fred_alfred",
                }
            )
            records.append(
                {
                    "series_id": series,
                    "observation_time": month,
                    "value": truth,
                    "release_time": pd.NaT,
                    "vintage_time": (month_end + pd.Timedelta(days=45)).tz_localize("UTC"),
                    "ingested_at": (month_end + pd.Timedelta(days=46)).tz_localize("UTC"),
                    "source": "fred_alfred",
                }
            )
    return pd.DataFrame.from_records(records)


class RecordingModel:
    """Delegating forecaster that records every training panel it receives."""

    name = "recording"
    version = "1.0.0"

    def __init__(self, inner_factory, sink: list[pd.DataFrame]) -> None:
        self._inner_factory = inner_factory
        self._sink = sink
        self._inner = inner_factory()

    def fit(self, history: pd.DataFrame) -> "RecordingModel":
        self._sink.append(history.copy())
        self._inner = self._inner_factory().fit(history)
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        return self._inner.predict(horizon)

    def predict_quantiles(self, horizon: int, quantiles: tuple[float, ...]) -> pd.DataFrame:
        return self._inner.predict_quantiles(horizon, quantiles)

    def metadata(self) -> dict:
        return self._inner.metadata()

    def save(self, path) -> None:
        self._inner.save(path)


class RecordingFactory:
    def __init__(self, inner_factory) -> None:
        self.inner_factory = inner_factory
        self.training_panels: list[pd.DataFrame] = []

    def __call__(self) -> Forecaster:
        return RecordingModel(self.inner_factory, self.training_panels)


def test_availability_as_of_is_month_end_plus_lag_utc() -> None:
    as_of = availability_as_of(pd.Timestamp("2020-02-01"), 45)
    assert as_of == pd.Timestamp("2020-04-14").tz_localize("UTC")
    assert as_of.tzinfo is not None
    with pytest.raises(ValueError):
        availability_as_of(pd.Timestamp("2020-02-01"), 0)


def test_training_uses_preliminary_origin_month_and_revised_history() -> None:
    panel = _versioned_panel()
    factory = RecordingFactory(lambda: PerSeriesPanelAdapter(SeasonalNaive))
    result = vintage_panel_backtest(
        panel,
        factory,
        horizons=(1,),
        min_train_size=60,
        availability_lag_days=30,
        step=48,
    )
    assert not result.excluded_origins
    training = factory.training_panels[0]
    origin_month = pd.Timestamp(training["observation_time"].max())

    prelim_vintage = (origin_month + pd.offsets.MonthEnd(0)).tz_localize("UTC") + pd.Timedelta(days=15)
    month_index = list(pd.date_range(START, periods=MONTHS, freq="MS")).index(origin_month)
    expected_prelim = round(_true_value("alpha", month_index) + 1.0, 6)
    origin_rows = training.loc[
        (training["series_id"] == "alpha") & (training["observation_time"] == origin_month)
    ]
    assert len(origin_rows) == 1
    assert float(origin_rows.iloc[0]["value"]) == pytest.approx(expected_prelim)
    assert pd.Timestamp(origin_rows.iloc[0]["vintage_time"]) == prelim_vintage

    alpha_rows = training.loc[training["series_id"] == "alpha"].sort_values("observation_time")
    previous_month = pd.Timestamp(alpha_rows["observation_time"].iloc[-2])
    previous_rows = training.loc[
        (training["series_id"] == "alpha") & (training["observation_time"] == previous_month)
    ]
    revised_index = list(pd.date_range(START, periods=MONTHS, freq="MS")).index(previous_month)
    assert float(previous_rows.iloc[0]["value"]) == pytest.approx(_true_value("alpha", revised_index))


def test_actuals_use_latest_revision() -> None:
    panel = _versioned_panel()
    result = vintage_panel_backtest(
        panel,
        lambda: PerSeriesPanelAdapter(SeasonalNaive),
        horizons=(1,),
        min_train_size=60,
        availability_lag_days=30,
        step=48,
    )
    months = pd.date_range(START, periods=MONTHS, freq="MS")
    folds = result.folds
    for row in folds.itertuples():
        month_index = list(months).index(pd.Timestamp(row.target_time))
        assert row.actual == pytest.approx(_true_value(row.series_id, month_index))
        assert row.actual_source == "latest_revision_vintage"


def test_origins_missing_a_series_vintage_are_excluded_with_reasons() -> None:
    panel = _versioned_panel()
    months = pd.date_range(START, periods=MONTHS, freq="MS")
    # Remove gamma's preliminary vintage for one late month so that origin
    # cannot assemble a complete three-series real-time panel.
    target_month = months[MONTHS - 20]
    prelim_vintage = (target_month + pd.offsets.MonthEnd(0)).tz_localize("UTC") + pd.Timedelta(days=15)
    dropped = (panel["series_id"] == "gamma") & (panel["vintage_time"] == prelim_vintage)
    panel = panel.loc[~dropped].reset_index(drop=True)

    result = vintage_panel_backtest(
        panel,
        lambda: PerSeriesPanelAdapter(SeasonalNaive),
        horizons=(1,),
        min_train_size=60,
        availability_lag_days=30,
    )
    excluded_origins = {item["forecast_origin"] for item in result.excluded_origins}
    assert any(
        "training panel is ragged across series" in item["reason"]
        for item in result.excluded_origins
    )
    fold_origins = {pd.Timestamp(origin).date().isoformat() for origin in result.folds["forecast_origin"]}
    assert fold_origins.isdisjoint(excluded_origins)


def test_future_months_visible_at_origin_are_excluded() -> None:
    panel = _versioned_panel()
    months = pd.date_range(START, periods=MONTHS, freq="MS")
    target_month = months[MONTHS - 20]
    future_month = target_month + pd.offsets.MonthBegin(1)
    # A prematurely visible future month makes the as-of panel leak; the
    # harness must refuse to evaluate that origin.
    leak_vintage = (target_month + pd.offsets.MonthEnd(0)).tz_localize("UTC") + pd.Timedelta(days=30)
    panel = pd.concat(
        [
            panel,
            pd.DataFrame(
                {
                    "series_id": ["alpha"],
                    "observation_time": [future_month],
                    "value": [123.0],
                    "release_time": [pd.NaT],
                    "vintage_time": [leak_vintage],
                    "ingested_at": [leak_vintage],
                    "source": ["fred_alfred"],
                }
            ),
        ],
        ignore_index=True,
    )

    result = vintage_panel_backtest(
        panel,
        lambda: PerSeriesPanelAdapter(SeasonalNaive),
        horizons=(1,),
        min_train_size=60,
        availability_lag_days=30,
    )
    assert any(
        "most recent training month" in item["reason"] or "ragged" in item["reason"]
        for item in result.excluded_origins
    )


def test_models_share_identical_origins() -> None:
    panel = _versioned_panel()
    naive = vintage_panel_backtest(
        panel,
        lambda: PerSeriesPanelAdapter(SeasonalNaive),
        horizons=(1, 3),
        min_train_size=60,
        availability_lag_days=30,
        step=12,
    )
    recording = RecordingFactory(lambda: PerSeriesPanelAdapter(SeasonalNaive))
    adapted = vintage_panel_backtest(
        panel,
        recording,
        horizons=(1, 3),
        min_train_size=60,
        availability_lag_days=30,
        step=12,
    )


def test_insufficient_panel_raises() -> None:
    panel = _versioned_panel().loc[
        _versioned_panel()["observation_time"] < "2016-01-01"
    ]
    with pytest.raises(ValueError):
        vintage_panel_backtest(
            panel,
            PerSeriesPanelAdapter(SeasonalNaive),
            min_train_size=60,
        )


def test_naive_arguments_are_validated() -> None:
    panel = _versioned_panel()
    with pytest.raises(ValueError):
        vintage_panel_backtest(panel, PerSeriesPanelAdapter(SeasonalNaive), min_train_size=0)
    with pytest.raises(ValueError):
        vintage_panel_backtest(panel, PerSeriesPanelAdapter(SeasonalNaive), horizons=(0,))
