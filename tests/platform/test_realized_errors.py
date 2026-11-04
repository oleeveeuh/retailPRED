from datetime import datetime, timezone

import pandas as pd
import pytest

from retailpred.live_loop.realized import (
    compute_realized_errors,
    run_realized_monitor,
)
from retailpred.platform.local_store import LocalTableStore
from retailpred.platform.specs import GOLD_FORECASTS, GOLD_REALIZED_ERRORS, SILVER_MONTHLY

NOW = datetime(2026, 4, 1, 15, 0, tzinfo=timezone.utc)
ORIGIN = pd.Timestamp("2025-12-01")
VINTAGE = pd.Timestamp("2026-01-14T15:30:00Z")


def _forecasts(**overrides) -> pd.DataFrame:
    records = []
    for series, prediction, naive in (
        ("alpha", 110.0, 108.0),
        ("beta", 55.0, 52.0),
        ("gamma", 18.0, None),
    ):
        for horizon in (1, 2):
            records.append(
                {
                    "dataset": "mrts",
                    "series_id": series,
                    "forecast_origin": ORIGIN,
                    "target_time": ORIGIN + pd.offsets.MonthBegin(horizon),
                    "horizon": horizon,
                    "model_name": "global_lightgbm",
                    "model_version": "2.0.0",
                    "forecast_vintage_time": VINTAGE,
                    "prediction": prediction + horizon,
                    "lower": None,
                    "upper": None,
                    "data_cutoff": ORIGIN,
                    "origin_as_of": VINTAGE,
                    "uncertainty_status": "research_only_not_calibrated",
                    "seasonal_naive_prediction": None if naive is None else naive + horizon,
                    "seasonal_naive_source_time": ORIGIN
                    + pd.offsets.MonthBegin(horizon)
                    - pd.DateOffset(months=12),
                    "seasonal_naive_source_vintage_time": VINTAGE,
                    "seasonal_naive_status": "available" if naive is not None else "unavailable",
                    "seasonal_naive_unavailable_reason": None
                    if naive is not None
                    else "prior_year_observation_missing",
                    "generated_at": VINTAGE,
                    "code_revision": "a" * 40,
                }
            )
    frame = pd.DataFrame.from_records(records, columns=GOLD_FORECASTS.columns)
    for key, value in overrides.items():
        frame[key] = value
    return frame


def _silver() -> pd.DataFrame:
    months = [ORIGIN + pd.offsets.MonthBegin(1), ORIGIN + pd.offsets.MonthBegin(2)]
    records = []
    for series, base in (("alpha", 105.0), ("beta", 50.0), ("gamma", 17.0)):
        for month_index, month in enumerate(months):
            records.append(
                {
                    "series_id": series,
                    "observation_time": month,
                    "value": base + month_index,
                    "release_time": pd.NaT,
                    "vintage_time": month + pd.offsets.MonthEnd(0) + pd.Timedelta(days=15),
                    "ingested_at": month + pd.offsets.MonthEnd(0) + pd.Timedelta(days=16),
                    "source": "census_mrts",
                }
            )
    frame = pd.DataFrame.from_records(records)
    revised = frame.assign(
        vintage_time=frame["vintage_time"] + pd.Timedelta(days=30),
        ingested_at=frame["ingested_at"] + pd.Timedelta(days=30),
        value=frame["value"] - 1.0,
    )
    return pd.concat([frame, revised], ignore_index=True)


def test_realized_errors_cover_model_naive_and_beat_flag() -> None:
    errors = compute_realized_errors(_forecasts(), _silver(), now=NOW)
    # 6 forecast rows x 2 actual vintages: first-release and revised coexist.
    assert len(errors) == 12
    alpha_january_all = errors.loc[
        (errors["series_id"] == "alpha") & (errors["horizon"] == 1)
    ]
    revised = alpha_january_all.loc[alpha_january_all["evaluation_status"] == "revised"].iloc[0]
    revised_actual = 105.0 - 1.0
    assert revised["actual"] == pytest.approx(revised_actual)
    assert revised["model_absolute_error"] == pytest.approx(abs(111.0 - revised_actual))
    assert revised["model_abs_pct_error"] == pytest.approx(
        abs(111.0 - revised_actual) / revised_actual
    )
    naive = 108.0 + 1
    assert revised["seasonal_naive_prediction"] == pytest.approx(naive)
    assert revised["seasonal_naive_absolute_error"] == pytest.approx(
        abs(naive - revised_actual)
    )
    assert bool(revised["model_beat_seasonal_naive"]) == (
        abs(111.0 - revised_actual) < abs(naive - revised_actual)
    )


def test_gamma_without_benchmark_keeps_naive_fields_null() -> None:
    errors = compute_realized_errors(_forecasts(), _silver(), now=NOW)
    gamma = errors.loc[errors["series_id"] == "gamma"].iloc[0]
    assert pd.isna(gamma["seasonal_naive_prediction"])
    assert pd.isna(gamma["seasonal_naive_absolute_error"])
    assert pd.isna(gamma["model_beat_seasonal_naive"])
    assert not pd.isna(gamma["model_absolute_error"])


def test_zero_actual_nulls_every_percentage_error() -> None:
    silver = _silver()
    gamma_january = (
        (silver["series_id"] == "gamma")
        & (silver["observation_time"] == ORIGIN + pd.offsets.MonthBegin(1))
    )
    silver.loc[gamma_january, "value"] = 0.0
    errors = compute_realized_errors(_forecasts(), silver, now=NOW)
    row = errors.loc[
        (errors["series_id"] == "gamma")
        & (errors["horizon"] == 1)
        & (errors["evaluation_status"] == "first_release")
    ].iloc[0]
    assert row["actual"] == 0.0
    assert pd.isna(row["model_abs_pct_error"])
    assert pd.isna(row["signed_pct_error"])
    assert not pd.isna(row["model_absolute_error"])


def test_first_release_evaluation_is_distinguished_from_revised() -> None:
    errors = compute_realized_errors(_forecasts(), _silver(), now=NOW)
    statuses = errors.groupby(["series_id", "horizon"])["evaluation_status"].apply(list)
    for status_list in statuses:
        assert sorted(status_list) == ["first_release", "revised"]


def test_uncovered_targets_produce_no_rows() -> None:
    future_only = _forecasts()
    future_only["target_time"] = pd.Timestamp("2030-06-01")
    errors = compute_realized_errors(future_only, _silver(), now=NOW)
    assert errors.empty


def test_monitor_merges_idempotently(tmp_path) -> None:
    store = LocalTableStore(tmp_path)
    store.merge(SILVER_MONTHLY, _silver())
    store.replace_snapshot(GOLD_FORECASTS, _forecasts())

    first = run_realized_monitor(store, now=NOW)
    second = run_realized_monitor(store, now=datetime(2026, 4, 2, tzinfo=timezone.utc))
    assert first.rows_computed == 12
    assert first.rows_appended == 12
    assert first.rows_in_table == 12
    assert second.rows_computed == 12  # recomputed candidates
    assert second.rows_appended == 0  # exact replay is a no-op
    assert second.rows_in_table == 12

    table = store.read(GOLD_REALIZED_ERRORS)
    assert not table.loc[:, list(GOLD_REALIZED_ERRORS.key)].duplicated().any()


def test_monitor_requires_published_forecasts(tmp_path) -> None:
    store = LocalTableStore(tmp_path)
    store.merge(SILVER_MONTHLY, _silver())
    with pytest.raises(ValueError, match="realized monitoring is unavailable"):
        run_realized_monitor(store, now=NOW)
