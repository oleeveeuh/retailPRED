from datetime import datetime, timezone

import pandas as pd
import pytest

from retailpred.platform.releases import attach_release_time, normalize_retail_releases


INGESTED = datetime(2025, 3, 20, tzinfo=timezone.utc)


def _schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "product": ["marts", "mrts"],
            "observation_month": ["2025-01-01", "2025-01-01"],
            "release_time": ["2025-02-14T08:30:00-05:00", "2025-03-14T08:30:00-04:00"],
            "release_kind": ["advance", "revised"],
        }
    )


def test_release_catalog_distinguishes_advance_and_revised_values() -> None:
    releases = normalize_retail_releases(
        _schedule(),
        source_url="https://www.census.gov/retail/release_schedule.html",
        ingested_at=INGESTED,
    )
    assert releases["release_id"].nunique() == 2
    assert set(releases["release_kind"]) == {"advance", "revised"}

    observations = pd.DataFrame(
        {"series_id": ["retail_total"], "observation_time": [pd.Timestamp("2025-01-01")]}
    )
    attached = attach_release_time(
        observations, releases, product="mrts", release_kind="revised"
    )
    assert attached.iloc[0]["release_time"] == pd.Timestamp("2025-03-14T12:30:00Z")


def test_release_catalog_rejects_future_dates_and_missing_coverage() -> None:
    future = _schedule().iloc[[0]].copy()
    future["release_time"] = "2026-01-01T08:30:00-05:00"
    with pytest.raises(ValueError, match="future scheduled"):
        normalize_retail_releases(
            future,
            source_url="https://www.census.gov/retail/release_schedule.html",
            ingested_at=INGESTED,
        )

    releases = normalize_retail_releases(
        _schedule(),
        source_url="https://www.census.gov/retail/release_schedule.html",
        ingested_at=INGESTED,
    )
    missing = pd.DataFrame(
        {"series_id": ["retail_total"], "observation_time": [pd.Timestamp("2024-12-01")]}
    )
    with pytest.raises(ValueError, match="does not cover"):
        attach_release_time(missing, releases, product="mrts", release_kind="revised")
