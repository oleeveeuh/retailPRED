"""Canonical storage boundary for versioned native-monthly observations."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


MONTHLY_COLUMNS = (
    "series_id",
    "observation_time",
    "value",
    "vintage_time",
    "ingested_at",
    "source",
)
MONTHLY_KEY = ("series_id", "observation_time", "vintage_time")


def validate_monthly_observations(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the canonical observation table."""
    missing = set(MONTHLY_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"monthly observations are missing columns: {sorted(missing)}")
    result = frame.loc[:, MONTHLY_COLUMNS].copy()
    result["series_id"] = result["series_id"].astype("string")
    result["source"] = result["source"].astype("string")
    result["observation_time"] = pd.to_datetime(
        result["observation_time"], format="mixed", errors="raise"
    )
    result["vintage_time"] = pd.to_datetime(
        result["vintage_time"], format="mixed", utc=True, errors="raise"
    )
    result["ingested_at"] = pd.to_datetime(
        result["ingested_at"], format="mixed", utc=True, errors="raise"
    )
    result["value"] = pd.to_numeric(result["value"], errors="raise").astype(float)

    if result[list(MONTHLY_COLUMNS)].isna().any().any():
        raise ValueError("canonical monthly observations cannot contain nulls")
    if not np.isfinite(result["value"]).all():
        raise ValueError("monthly observation values must be finite")
    if result["observation_time"].dt.tz is not None:
        raise ValueError("observation_time must be a timezone-naive calendar month")
    normalized_month = result["observation_time"].dt.to_period("M").dt.to_timestamp()
    if not result["observation_time"].equals(normalized_month):
        raise ValueError("observation_time must be normalized to midnight on the first day of its month")
    if result[list(MONTHLY_KEY)].duplicated().any():
        raise ValueError("duplicate series/month/vintage keys are not allowed")
    if (result["ingested_at"] < result["vintage_time"]).any():
        raise ValueError("ingested_at cannot precede vintage_time")
    return result.sort_values(list(MONTHLY_KEY), ignore_index=True)


def observations_as_of(frame: pd.DataFrame, as_of: datetime) -> pd.DataFrame:
    """Select the latest known vintage for every series/month at an as-of time."""
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")
    clean = validate_monthly_observations(frame)
    eligible = clean.loc[clean["vintage_time"] <= pd.Timestamp(as_of)]
    if eligible.empty:
        raise ValueError("no monthly observations were available at the requested as_of time")
    latest = eligible.groupby(["series_id", "observation_time"], sort=False)["vintage_time"].idxmax()
    return eligible.loc[latest].sort_values(["series_id", "observation_time"], ignore_index=True)


def write_monthly_csv(frame: pd.DataFrame, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    validate_monthly_observations(frame).to_csv(destination, index=False)


def read_monthly_csv(path: str | Path) -> pd.DataFrame:
    return validate_monthly_observations(pd.read_csv(path))
