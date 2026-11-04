"""Realized-error monitoring: score published forecasts against revised actuals.

Once actual months are published, every covering forecast gains an immutable
realized-error row. Revisions append new rows keyed by the actual vintage
rather than overwriting history, so the monitoring table itself is
revision-aware.
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


def latest_revisions(silver: pd.DataFrame) -> pd.DataFrame:
    """Latest known value per series/month from the versioned silver table."""
    clean = validate_monthly_observations(silver)
    latest = clean.groupby(["series_id", "observation_time"], sort=False)[
        "vintage_time"
    ].idxmax()
    return clean.loc[latest, ["series_id", "observation_time", "value", "vintage_time"]]


def compute_realized_errors(
    forecasts: pd.DataFrame, silver: pd.DataFrame, *, now: datetime
) -> pd.DataFrame:
    """Join published forecasts to revised actuals and quantify the realized error."""
    if forecasts.empty:
        raise ValueError("no published forecasts exist; realized monitoring is unavailable")
    actuals = latest_revisions(silver)
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
        validate="many_to_one",
    )
    records: list[dict[str, object]] = []
    for row in joined.itertuples():
        actual = float(row.actual)
        if actual == 0:
            abs_pct: float | None = None
            signed_pct: float | None = None
        else:
            abs_pct = abs(float(row.prediction) - actual) / actual
            signed_pct = (float(row.prediction) - actual) / actual
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
                "actual": actual,
                "actual_vintage_time": pd.Timestamp(row.actual_vintage_time),
                "abs_pct_error": abs_pct,
                "signed_pct_error": signed_pct,
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
