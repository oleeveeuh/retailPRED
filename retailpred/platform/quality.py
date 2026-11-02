"""Fail-closed data quality gates for monthly retail facts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from forecasting.data.monthly_dataset import validate_monthly_observations
from forecasting.registry import MRTS_BY_ID


class DataQualityError(ValueError):
    pass


@dataclass(frozen=True)
class QualityReport:
    rows: int
    series: int
    first_observation: pd.Timestamp
    last_observation: pd.Timestamp
    checks: tuple[str, ...]


def validate_silver_monthly(
    frame: pd.DataFrame,
    *,
    as_of: datetime | None = None,
    max_age_months: int | None = None,
) -> tuple[pd.DataFrame, QualityReport]:
    try:
        clean = validate_monthly_observations(frame)
    except ValueError as exc:
        raise DataQualityError(str(exc)) from exc
    unknown = sorted(set(clean["series_id"]) - set(MRTS_BY_ID))
    if unknown:
        raise DataQualityError(f"unknown series_id values: {unknown}")
    if (clean["value"] < 0).any() or not np.isfinite(clean["value"]).all():
        raise DataQualityError("monthly retail sales must be finite and non-negative")

    checks = ["schema", "required", "unique_key", "registered_series", "non_negative"]
    if as_of is not None or max_age_months is not None:
        if as_of is None or max_age_months is None:
            raise ValueError("as_of and max_age_months must be provided together")
        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise ValueError("as_of must be timezone-aware")
        latest = clean["observation_time"].max().to_period("M")
        cutoff = pd.Timestamp(as_of).tz_convert("UTC").tz_localize(None).to_period("M") - max_age_months
        if latest < cutoff:
            raise DataQualityError(
                f"monthly observations are stale: latest={latest}, required_at_least={cutoff}"
            )
        checks.append("freshness")

    report = QualityReport(
        rows=len(clean),
        series=clean["series_id"].nunique(),
        first_observation=clean["observation_time"].min(),
        last_observation=clean["observation_time"].max(),
        checks=tuple(checks),
    )
    return clean, report


def assert_reconciled(
    parent: pd.Series,
    children: pd.DataFrame,
    *,
    absolute_tolerance: float,
) -> None:
    if absolute_tolerance < 0:
        raise ValueError("absolute_tolerance cannot be negative")
    child_sum = children.sum(axis=1)
    difference = (parent - child_sum).abs()
    if (difference > absolute_tolerance).any():
        worst = float(difference.max())
        raise DataQualityError(
            f"hierarchy reconciliation failed: max difference {worst} exceeds {absolute_tolerance}"
        )


def assert_monthly_continuity(frame: pd.DataFrame) -> None:
    for series_id, group in frame.groupby("series_id", sort=False):
        periods = pd.to_datetime(group["observation_time"]).dt.to_period("M").sort_values()
        expected = pd.period_range(periods.iloc[0], periods.iloc[-1], freq="M")
        if len(periods) != len(expected) or periods.duplicated().any():
            raise DataQualityError(f"monthly continuity failed for {series_id}")
