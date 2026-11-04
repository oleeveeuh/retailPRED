from datetime import datetime, timezone

import pandas as pd
import pytest

from retailpred.platform.gold import build_gold_monthly
from retailpred.platform.local_store import LocalTableStore
from retailpred.platform.mrts_pipeline import ExtractContext, ingest_extract
from retailpred.platform.specs import BRONZE_MRTS, GOLD_MONTHLY, SILVER_MONTHLY


def _payload(value: str = "7004") -> list[list[str]]:
    return [
        [
            "data_type_code",
            "time_slot_id",
            "seasonally_adj",
            "category_code",
            "cell_value",
            "error_data",
            "time",
        ],
        ["SM", "1", "yes", "443", value, "", "2025"],
        ["SM", "1", "no", "443", "6500", "", "2025"],
        ["IM", "1", "yes", "443", "100", "", "2025"],
    ]


def _context(extract_id: str, vintage: datetime) -> ExtractContext:
    return ExtractContext(
        extract_id=extract_id,
        source_url="https://api.census.gov/data/timeseries/eits/mrts?time=2025",
        release_time=vintage,
        vintage_time=vintage,
        ingested_at=vintage,
    )


def test_extract_is_idempotent_across_bronze_and_silver(tmp_path) -> None:
    store = LocalTableStore(tmp_path)
    vintage = datetime(2025, 2, 14, tzinfo=timezone.utc)
    first_bronze, first_silver = ingest_extract(store, _payload(), _context("release-1", vintage))
    second_bronze, second_silver = ingest_extract(store, _payload(), _context("release-1", vintage))

    assert len(first_bronze) == len(second_bronze) == 3
    assert len(first_silver) == len(second_silver) == 1
    assert len(store.read(BRONZE_MRTS)) == 3
    assert len(store.read(SILVER_MONTHLY)) == 1


def test_point_in_time_gold_reconstructs_the_latest_known_revision(tmp_path) -> None:
    store = LocalTableStore(tmp_path)
    first_vintage = datetime(2025, 2, 14, tzinfo=timezone.utc)
    revised_vintage = datetime(2025, 3, 14, tzinfo=timezone.utc)
    ingest_extract(store, _payload("7004"), _context("release-1", first_vintage))
    ingest_extract(store, _payload("7010"), _context("release-2", revised_vintage))
    silver = store.read(SILVER_MONTHLY)

    first_gold = build_gold_monthly(silver, as_of=first_vintage)
    revised_gold = build_gold_monthly(silver, as_of=revised_vintage)
    assert first_gold.iloc[0]["value"] == 7004
    assert revised_gold.iloc[0]["value"] == 7010
    assert list(revised_gold.columns) == list(GOLD_MONTHLY.columns)

    store.replace_snapshot(GOLD_MONTHLY, revised_gold)
    persisted = store.read(GOLD_MONTHLY)
    pd.testing.assert_frame_equal(persisted, revised_gold)


def test_extract_without_eligible_monthly_rows_stops_before_silver(tmp_path) -> None:
    store = LocalTableStore(tmp_path)
    vintage = datetime(2025, 2, 14, tzinfo=timezone.utc)
    payload = [_payload()[0], ["IM", "1", "yes", "443", "100", "", "2025"]]
    with pytest.raises(ValueError, match="no registered seasonally adjusted"):
        ingest_extract(store, payload, _context("invalid-release", vintage))
    assert store.read(SILVER_MONTHLY).empty


def test_silver_derives_months_from_period_column_with_ordinal_slots(tmp_path) -> None:
    """Mirror the observed current API schema end to end through bronze."""
    store = LocalTableStore(tmp_path)
    vintage = datetime(2026, 11, 15, 15, 0, tzinfo=timezone.utc)
    header = [
        "data_type_code",
        "time_slot_id",
        "seasonally_adj",
        "category_code",
        "cell_value",
        "time",
    ]
    payload = [
        header,
        ["SM", "0", "yes", "443", "7004", "2025-01"],
        ["SM", "0", "yes", "443", "7010", "2025-02"],
    ]
    bronze, silver = ingest_extract(store, payload, _context("current-schema", vintage))
    assert silver["observation_time"].dt.strftime("%Y-%m").tolist() == ["2025-01", "2025-02"]
    assert bronze["year"].tolist() == ["2025-01", "2025-02"]
