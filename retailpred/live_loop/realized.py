"""Realized-error monitoring: score published forecasts against actuals.

When an actual month becomes available, every applicable stored forecast
vintage gains an immutable realized-error row keyed by the actual vintage,
so first-release evaluation and later revised evaluations coexist without
overwriting each other. Each row records how the model did AND how the
frozen seasonal-naive benchmark did on the same target, including whether
the model beat the benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from forecasting.data.monthly_dataset import validate_monthly_observations
from retailpred.platform.local_store import LocalTableStore
from retailpred.platform.specs import GOLD_FORECASTS, GOLD_REALIZED_ERRORS, SILVER_MONTHLY


@dataclass(frozen=True)
class RealizedOutcome:
    rows_computed: int
    rows_appended: int
    rows_in_table: int
    evaluated_at: datetime


def versioned_actuals(silver: pd.DataFrame) -> pd.DataFrame:
    """Every published actual vintage, so first-release evaluation survives."""
    clean = validate_monthly_observations(silver)
    return clean.loc[:, ["series_id", "observation_time", "value", "vintage_time"]]


def _pct_error(prediction: float | None, actual: float) -> float | None:
    if prediction is None or actual == 0:
        return None
    return abs(prediction - actual) / actual


def compute_realized_errors(
    forecasts: pd.DataFrame, silver: pd.DataFrame, *, now: datetime
) -> pd.DataFrame:
    """Join published forecasts to every actual vintage and quantify performance.

    One row per (forecast vintage, actual vintage): a first-release actual and
    any later revision each get their own immutable evaluation row.
    """
    if forecasts.empty:
        raise ValueError("no published forecasts exist; realized monitoring is unavailable")
    actuals = versioned_actuals(silver)
    joined = forecasts.merge(
        actuals.rename(
            columns={
                "observation_time": "target_time",
                "value": "actual",
                "vintage_time": "actual_vintage_time",
            }
        ),
        on=["series_id", "target_time"],
        how="inner",
        validate="many_to_many",
    )
    first_release_times = actuals.groupby(["series_id", "observation_time"], sort=False)[
        "vintage_time"
    ].min()

    records: list[dict[str, object]] = []
    for row in joined.itertuples():
        actual = float(row.actual)
        model_absolute = abs(float(row.prediction) - actual)
        naive_prediction = None if pd.isna(row.seasonal_naive_prediction) else float(row.seasonal_naive_prediction)
        naive_absolute = None if naive_prediction is None else abs(naive_prediction - actual)
        beat = None
        if naive_absolute is not None:
            beat = bool(model_absolute < naive_absolute)
        evaluation_status = (
            "first_release"
            if pd.Timestamp(row.actual_vintage_time)
            == first_release_times.loc[(str(row.series_id), pd.Timestamp(row.target_time))]
            else "revised"
        )
        records.append(
            {
                "dataset": str(row.dataset),
                "series_id": str(row.series_id),
                "forecast_origin": pd.Timestamp(row.forecast_origin),
                "target_time": pd.Timestamp(row.target_time),
                "horizon": int(row.horizon),
                "model_name": str(row.model_name),
                "model_version": str(row.model_version),
                "forecast_vintage_time": pd.Timestamp(row.forecast_vintage_time),
                "prediction": float(row.prediction),
                "seasonal_naive_prediction": naive_prediction,
                "actual": actual,
                "actual_vintage_time": pd.Timestamp(row.actual_vintage_time),
                "model_absolute_error": model_absolute,
                "model_abs_pct_error": _pct_error(float(row.prediction), actual),
                "signed_pct_error": None if actual == 0 else (float(row.prediction) - actual) / actual,
                "seasonal_naive_absolute_error": naive_absolute,
                "seasonal_naive_abs_pct_error": _pct_error(naive_prediction, actual),
                "model_beat_seasonal_naive": beat,
                "evaluation_status": evaluation_status,
            }
        )
    return pd.DataFrame.from_records(records, columns=GOLD_REALIZED_ERRORS.columns)


def run_realized_monitor(
    store: LocalTableStore, *, now: datetime
) -> RealizedOutcome:
    """Append realized-error rows for every newly covered forecast target."""
    errors = compute_realized_errors(store.read(GOLD_FORECASTS), store.read(SILVER_MONTHLY), now=now)
    before = store.read(GOLD_REALIZED_ERRORS)
    merged = store.merge(GOLD_REALIZED_ERRORS, errors) if not errors.empty else before
    return RealizedOutcome(
        rows_computed=len(errors),
        rows_appended=len(merged) - len(before),
        rows_in_table=len(merged),
        evaluated_at=now,
    )
