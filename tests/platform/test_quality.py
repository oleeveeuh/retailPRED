from datetime import datetime, timezone

import pandas as pd
import pytest

from retailpred.platform.quality import (
    DataQualityError,
    assert_monthly_continuity,
    assert_reconciled,
    validate_silver_monthly,
)


def _observations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_id": ["retail_total", "retail_total"],
            "observation_time": ["2024-12-01", "2025-01-01"],
            "value": [100.0, 101.0],
            "release_time": [pd.NaT, pd.NaT],
            "vintage_time": ["2025-02-01T00:00:00Z"] * 2,
            "ingested_at": ["2025-02-01T00:00:00Z"] * 2,
            "source": ["census_mrts"] * 2,
        }
    )


def test_quality_report_includes_freshness_gate() -> None:
    clean, report = validate_silver_monthly(
        _observations(),
        as_of=datetime(2025, 2, 15, tzinfo=timezone.utc),
        max_age_months=2,
    )
    assert len(clean) == 2
    assert "freshness" in report.checks


def test_unknown_series_negative_values_and_stale_data_fail() -> None:
    unknown = _observations().assign(series_id="unknown")
    with pytest.raises(DataQualityError, match="unknown series"):
        validate_silver_monthly(unknown)

    negative = _observations()
    negative.loc[0, "value"] = -1
    with pytest.raises(DataQualityError, match="non-negative"):
        validate_silver_monthly(negative)

    with pytest.raises(DataQualityError, match="stale"):
        validate_silver_monthly(
            _observations(),
            as_of=datetime(2026, 1, 1, tzinfo=timezone.utc),
            max_age_months=3,
        )


def test_reconciliation_gate() -> None:
    parent = pd.Series([10.0, 20.0])
    children = pd.DataFrame({"a": [4.0, 8.0], "b": [6.0, 12.0]})
    assert_reconciled(parent, children, absolute_tolerance=0)
    with pytest.raises(DataQualityError, match="reconciliation failed"):
        assert_reconciled(parent + 1, children, absolute_tolerance=0.5)


def test_gold_continuity_gate_rejects_missing_month() -> None:
    gap = _observations().copy()
    gap.loc[1, "observation_time"] = "2025-02-01"
    with pytest.raises(DataQualityError, match="continuity failed"):
        assert_monthly_continuity(gap)
