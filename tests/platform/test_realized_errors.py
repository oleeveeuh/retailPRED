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


def _forecasts() -> pd.DataFrame:
    records = []
    for series, prediction in (("alpha", 110.0), ("beta", 55.0), ("gamma", 18.0)):
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
                    "uncertainty_status": "research_only_not_calibrated",
                    "generated_at": VINTAGE,
                    "code_revision": "a" * 40,
                }
            )
    return pd.DataFrame.from_records(records, columns=GOLD_FORECASTS.columns)


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


def test_realized_errors_use_latest_revision_and_allow_zero_actual() -> None:
    silver = _silver()
    forecasts = _forecasts()
    # gamma's January target has a zero revised actual: percentage error is null.
    gamma_january = (
        (silver["series_id"] == "gamma")
        & (silver["observation_time"] == ORIGIN + pd.offsets.MonthBegin(1))
    )
    silver.loc[gamma_january, "value"] = 0.0

    errors = compute_realized_errors(forecasts, silver, now=NOW)
    assert len(errors) == 6
    alpha_january = errors.loc[
        (errors["series_id"] == "alpha") & (errors["horizon"] == 1)
    ].iloc[0]
    revised_actual = 105.0 - 1.0  # second (latest) vintage
    assert alpha_january["actual"] == pytest.approx(revised_actual)
    assert alpha_january["abs_pct_error"] == pytest.approx(abs(111.0 - revised_actual) / revised_actual)
    gamma_row = errors.loc[
        (errors["series_id"] == "gamma") & (errors["horizon"] == 1)
    ].iloc[0]
    assert pd.isna(gamma_row["abs_pct_error"]) and pd.isna(gamma_row["signed_pct_error"])
    assert not errors[list(GOLD_REALIZED_ERRORS.key)].duplicated().any()


def test_uncovered_targets_produce_no_rows() -> None:
    silver = _silver()
    future_only = _forecasts()
    future_only["target_time"] = pd.Timestamp("2030-06-01")
    errors = compute_realized_errors(future_only, silver, now=NOW)
    assert errors.empty


def test_monitor_merges_idempotently(tmp_path) -> None:
    store = LocalTableStore(tmp_path)
    store.merge(SILVER_MONTHLY, _silver())
    store.replace_snapshot(GOLD_FORECASTS, _forecasts())

    first = run_realized_monitor(store, now=NOW)
    second = run_realized_monitor(store, now=datetime(2026, 4, 2, tzinfo=timezone.utc))
    assert first.rows_computed == 6
    assert first.rows_appended == 6
    assert first.rows_in_table == 6
    assert second.rows_computed == 6  # recomputed candidates
    assert second.rows_appended == 0  # exact replay is a no-op
    assert second.rows_in_table == 6

    table = store.read(GOLD_REALIZED_ERRORS)
    assert not table.loc[:, list(GOLD_REALIZED_ERRORS.key)].duplicated().any()


def test_monitor_requires_published_forecasts(tmp_path) -> None:
    store = LocalTableStore(tmp_path)
    store.merge(SILVER_MONTHLY, _silver())
    with pytest.raises(ValueError, match="realized monitoring is unavailable"):
        run_realized_monitor(store, now=NOW)
