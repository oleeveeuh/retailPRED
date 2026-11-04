from datetime import datetime, timezone

import pandas as pd
import pytest

from forecasting.benchmarking.seasonal_naive import (
    REASON_PRIOR_YEAR_MISSING,
    REASON_PRIOR_YEAR_NOT_PUBLISHED,
    SEASONAL_NAIVE_AVAILABLE,
    SEASONAL_NAIVE_UNAVAILABLE,
    attach_seasonal_naive_benchmark,
    seasonal_naive_source_time,
)

ORIGIN = datetime(2026, 1, 14, 15, 0, tzinfo=timezone.utc)


def _panel(series: str, months_and_vintages: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "series_id": series,
                "observation_time": pd.Timestamp(month),
                "value": value,
                "release_time": pd.NaT,
                "vintage_time": pd.Timestamp(vintage),
                "ingested_at": pd.Timestamp(vintage),
                "source": "census_mrts",
            }
            for month, vintage, value in months_and_vintages
        ]
    )


def _forecast_rows(series: str, targets: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": "mrts",
                "series_id": series,
                "forecast_origin": pd.Timestamp("2025-12-01"),
                "target_time": pd.Timestamp(target),
                "horizon": horizon,
                "prediction": 1000.0 + horizon,
            }
            for horizon, target in enumerate(targets, start=1)
        ]
    )


def test_source_time_is_exact_twelve_month_offset() -> None:
    assert seasonal_naive_source_time(pd.Timestamp("2026-01-01")) == pd.Timestamp("2025-01-01")
    # December target resolves to the prior January, not a calendar-year roll.
    assert seasonal_naive_source_time(pd.Timestamp("2025-12-01")) == pd.Timestamp("2024-12-01")
    # Leap-year boundary: February 2024 resolves to February 2023.
    assert seasonal_naive_source_time(pd.Timestamp("2024-02-01")) == pd.Timestamp("2023-02-01")


def test_benchmark_uses_latest_revision_published_before_origin() -> None:
    panel = _panel(
        "retail_total",
        [
            ("2024-12-01", "2025-01-15T14:00:00Z", 7000.0),
            ("2024-12-01", "2025-02-14T14:00:00Z", 7010.0),
        ],
    )
    forecasts = _forecast_rows("retail_total", ["2025-12-01"])
    result = attach_seasonal_naive_benchmark(forecasts, panel, origin_as_of=ORIGIN)
    row = result.iloc[0]
    assert row["seasonal_naive_status"] == SEASONAL_NAIVE_AVAILABLE
    assert row["seasonal_naive_prediction"] == pytest.approx(7010.0)
    assert pd.Timestamp(row["seasonal_naive_source_vintage_time"]) == pd.Timestamp(
        "2025-02-14T14:00:00Z"
    )


def test_revision_published_after_origin_is_never_selected() -> None:
    panel = _panel(
        "retail_total",
        [
            ("2024-12-01", "2025-01-15T14:00:00Z", 7000.0),
            # Revision published AFTER the forecast origin must be invisible.
            ("2024-12-01", "2026-03-01T00:00:00Z", 7050.0),
        ],
    )
    forecasts = _forecast_rows("retail_total", ["2025-12-01"])
    result = attach_seasonal_naive_benchmark(forecasts, panel, origin_as_of=ORIGIN)
    row = result.iloc[0]
    assert row["seasonal_naive_status"] == SEASONAL_NAIVE_AVAILABLE
    assert row["seasonal_naive_prediction"] == pytest.approx(7000.0)
    assert pd.Timestamp(row["seasonal_naive_source_vintage_time"]) == pd.Timestamp(
        "2025-01-15T14:00:00Z"
    )


def test_missing_prior_year_stays_explicitly_missing() -> None:
    panel = _panel("retail_total", [("2025-01-01", "2025-02-14T14:00:00Z", 7000.0)])
    forecasts = _forecast_rows("retail_total", ["2025-12-01"])
    result = attach_seasonal_naive_benchmark(forecasts, panel, origin_as_of=ORIGIN)
    row = result.iloc[0]
    assert row["seasonal_naive_status"] == SEASONAL_NAIVE_UNAVAILABLE
    assert row["seasonal_naive_unavailable_reason"] == REASON_PRIOR_YEAR_MISSING
    assert pd.isna(row["seasonal_naive_prediction"])
    assert row["seasonal_naive_prediction"] is not 0  # type: ignore[comparison-overlap]
    assert pd.isna(row["seasonal_naive_source_vintage_time"])


def test_prior_year_published_only_after_origin_is_reported_separately() -> None:
    panel = _panel(
        "retail_total",
        [("2024-12-01", "2026-06-01T00:00:00Z", 7000.0)],
    )
    forecasts = _forecast_rows("retail_total", ["2025-12-01"])
    result = attach_seasonal_naive_benchmark(forecasts, panel, origin_as_of=ORIGIN)
    row = result.iloc[0]
    assert row["seasonal_naive_status"] == SEASONAL_NAIVE_UNAVAILABLE
    assert row["seasonal_naive_unavailable_reason"] == REASON_PRIOR_YEAR_NOT_PUBLISHED


def test_benchmark_is_frozen_against_a_later_panel_state() -> None:
    panel_before = _panel(
        "retail_total",
        [("2024-12-01", "2025-01-15T14:00:00Z", 7000.0)],
    )
    forecasts = _forecast_rows("retail_total", ["2025-12-01"])
    frozen = attach_seasonal_naive_benchmark(forecasts, panel_before, origin_as_of=ORIGIN)

    panel_after = _panel(
        "retail_total",
        [
            ("2024-12-01", "2025-01-15T14:00:00Z", 7000.0),
            ("2024-12-01", "2026-06-01T00:00:00Z", 7050.0),
        ],
    )
    reattached = attach_seasonal_naive_benchmark(forecasts, panel_after, origin_as_of=ORIGIN)
    # The origin timestamp, not the current panel, governs the selection.
    assert frozen["seasonal_naive_prediction"].equals(reattached["seasonal_naive_prediction"])


def test_timezone_naive_origin_is_rejected() -> None:
    panel = _panel("retail_total", [("2024-12-01", "2025-01-15T14:00:00Z", 7000.0)])
    with pytest.raises(ValueError, match="timezone-aware"):
        attach_seasonal_naive_benchmark(
            _forecast_rows("retail_total", ["2025-12-01"]),
            panel,
            origin_as_of=datetime(2026, 1, 14, 15, 0),
        )
