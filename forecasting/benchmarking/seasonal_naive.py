"""Vintage-safe seasonal-naive benchmark.

The benchmark for target month ``t`` is the actual observation for
``t - 12 months`` as it was published at or before the forecast origin.
A revision published after the origin must never retroactively change a
stored benchmark, so the source value and its vintage are frozen beside
the forecast at generation time. A missing or not-yet-published prior-year
observation stays explicitly missing: no zero, no substitute.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from forecasting.data.monthly_dataset import validate_monthly_observations

SEASONAL_NAIVE_AVAILABLE = "available"
SEASONAL_NAIVE_UNAVAILABLE = "unavailable"

REASON_PRIOR_YEAR_MISSING = "prior_year_observation_missing"
REASON_PRIOR_YEAR_NOT_PUBLISHED = "prior_year_observation_not_published_at_origin"

BENCHMARK_COLUMNS = (
    "seasonal_naive_prediction",
    "seasonal_naive_source_time",
    "seasonal_naive_source_vintage_time",
    "seasonal_naive_status",
    "seasonal_naive_unavailable_reason",
)


def seasonal_naive_source_time(target_time: Any) -> pd.Timestamp:
    """Same calendar month one year earlier; month-start arithmetic is exact."""
    return pd.Timestamp(target_time) - pd.DateOffset(months=12)


def attach_seasonal_naive_benchmark(
    forecasts: pd.DataFrame,
    versioned_panel: pd.DataFrame,
    *,
    origin_as_of: Any,
) -> pd.DataFrame:
    """Freeze the vintage-safe seasonal-naive benchmark beside every forecast.

    ``origin_as_of`` is the moment the forecast was produced. For each
    forecast row the prior-year observation published at or before that
    moment is selected; anything published later is invisible.
    """
    as_of = pd.Timestamp(origin_as_of)
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("origin_as_of must be timezone-aware")
    clean = validate_monthly_observations(versioned_panel)

    result = forecasts.copy().reset_index(drop=True)
    records: list[dict[str, Any]] = []
    for row in result.itertuples():
        source_time = seasonal_naive_source_time(row.target_time)
        series_id = str(row.series_id)
        candidates = clean.loc[
            (clean["series_id"] == series_id)
            & (clean["observation_time"] == source_time)
            & (clean["vintage_time"] <= as_of)
        ]
        if not candidates.empty:
            latest = candidates.loc[candidates["vintage_time"].idxmax()]
            records.append(
                {
                    "seasonal_naive_prediction": float(latest["value"]),
                    "seasonal_naive_source_time": source_time,
                    "seasonal_naive_source_vintage_time": pd.Timestamp(
                        latest["vintage_time"]
                    ),
                    "seasonal_naive_status": SEASONAL_NAIVE_AVAILABLE,
                    "seasonal_naive_unavailable_reason": None,
                }
            )
            continue
        ever_published = clean.loc[
            (clean["series_id"] == series_id)
            & (clean["observation_time"] == source_time)
        ]
        reason = (
            REASON_PRIOR_YEAR_NOT_PUBLISHED
            if not ever_published.empty
            else REASON_PRIOR_YEAR_MISSING
        )
        records.append(
            {
                "seasonal_naive_prediction": None,
                "seasonal_naive_source_time": source_time,
                "seasonal_naive_source_vintage_time": None,
                "seasonal_naive_status": SEASONAL_NAIVE_UNAVAILABLE,
                "seasonal_naive_unavailable_reason": reason,
            }
        )

    benchmark = pd.DataFrame.from_records(records, columns=BENCHMARK_COLUMNS)
    for column in BENCHMARK_COLUMNS:
        if column in result.columns:
            result = result.drop(columns=[column])
    return pd.concat([result, benchmark], axis=1)
