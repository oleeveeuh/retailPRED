from datetime import datetime, timezone

import pytest

from retailpred.platform.delta_sync import plan_sync, write_mode
from retailpred.platform.gold import build_gold_monthly
from retailpred.platform.local_store import LocalTableStore
from retailpred.platform.mrts_pipeline import ExtractContext, ingest_extract
from retailpred.platform.specs import (
    BRONZE_MRTS,
    GOLD_MONTHLY,
    SILVER_MONTHLY,
)

PAYLOAD = [
    [
        "data_type_code",
        "time_slot_id",
        "seasonally_adj",
        "category_code",
        "cell_value",
        "error_data",
        "time",
    ],
    ["SM", "1", "yes", "443", "7004", "", "2025"],
    ["SM", "2", "yes", "443", "7010", "", "2025"],
]


def _context(extract_id: str, vintage: datetime) -> ExtractContext:
    return ExtractContext(
        extract_id=extract_id,
        source_url="https://api.census.gov/data/timeseries/eits/mrts",
        release_time=vintage,
        vintage_time=vintage,
        ingested_at=vintage,
    )


def _populate(tmp_path) -> LocalTableStore:
    store = LocalTableStore(tmp_path)
    vintage = datetime(2025, 2, 14, tzinfo=timezone.utc)
    ingest_extract(store, PAYLOAD, _context("sync-1", vintage))
    gold = build_gold_monthly(store.read(SILVER_MONTHLY), as_of=vintage)
    store.replace_snapshot(GOLD_MONTHLY, gold)
    return store


def test_bronze_and_silver_merge_while_gold_replaces() -> None:
    assert write_mode(BRONZE_MRTS) == "merge_immutable"
    assert write_mode(SILVER_MONTHLY) == "merge_immutable"
    assert write_mode(GOLD_MONTHLY) == "replace_snapshot"


def test_plan_lists_populated_tables_with_manifest_hashes(tmp_path) -> None:
    store = _populate(tmp_path)
    actions = plan_sync(store)
    by_name = {action.spec.name: action for action in actions}
    assert set(by_name) == {"mrts_extract_rows", "monthly_observations", "monthly_series"}
    assert by_name["mrts_extract_rows"].mode == "merge_immutable"
    assert by_name["monthly_observations"].mode == "merge_immutable"
    assert by_name["monthly_series"].mode == "replace_snapshot"
    assert by_name["monthly_observations"].row_count == 2
    assert all(len(action.manifest_sha256) == 64 for action in actions)


def test_plan_is_stable_across_replays(tmp_path) -> None:
    store = _populate(tmp_path)
    first = plan_sync(store)
    second = plan_sync(store)
    assert first == second


def test_empty_store_yields_no_actions(tmp_path) -> None:
    assert plan_sync(LocalTableStore(tmp_path)) == []


def test_plan_respects_manifest_integrity(tmp_path) -> None:
    store = _populate(tmp_path)
    manifest = store.root / "silver" / "monthly_observations.manifest.json"
    manifest.write_text('{"table": "monthly_observations", "schema_version": 1}')
    with pytest.raises(Exception):
        plan_sync(store)
