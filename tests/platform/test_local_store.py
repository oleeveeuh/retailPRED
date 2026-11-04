from datetime import datetime, timezone
import json

import pandas as pd
import pytest

from retailpred.platform.local_store import (
    LocalTableStore,
    MergeConflictError,
    SchemaContractError,
    StorageIntegrityError,
    merge_immutable,
)
from retailpred.platform.specs import SILVER_MONTHLY


VINTAGE = datetime(2025, 2, 1, tzinfo=timezone.utc)


def _batch(value: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_id": ["retail_total"],
            "observation_time": [pd.Timestamp("2025-01-01")],
            "value": [value],
            "release_time": [pd.NaT],
            "vintage_time": [VINTAGE],
            "ingested_at": [VINTAGE],
            "source": ["census_mrts"],
        }
    )


def test_exact_replay_is_idempotent_and_manifest_is_auditable(tmp_path) -> None:
    store = LocalTableStore(tmp_path)
    first = store.merge(SILVER_MONTHLY, _batch())
    manifest_path = tmp_path / "silver" / "monthly_observations.manifest.json"
    first_manifest = json.loads(manifest_path.read_text())
    second = store.merge(SILVER_MONTHLY, _batch())
    second_manifest = json.loads(manifest_path.read_text())

    pd.testing.assert_frame_equal(first, second)
    assert len(second) == 1
    assert first_manifest == second_manifest
    assert first_manifest["row_count"] == 1
    assert len(first_manifest["sha256"]) == 64


def test_conflicting_immutable_fact_fails_closed() -> None:
    with pytest.raises(MergeConflictError, match="conflicting immutable fact"):
        merge_immutable(_batch(100), _batch(101), SILVER_MONTHLY)


def test_schema_evolution_is_rejected_until_contract_version_changes() -> None:
    unexpected = _batch().assign(unreviewed_column="surprise")
    with pytest.raises(SchemaContractError, match="extra=.*unreviewed_column"):
        merge_immutable(pd.DataFrame(columns=SILVER_MONTHLY.columns), unexpected, SILVER_MONTHLY)


def test_read_detects_manifest_or_table_tampering(tmp_path) -> None:
    store = LocalTableStore(tmp_path)
    store.merge(SILVER_MONTHLY, _batch())
    table_path = tmp_path / "silver" / "monthly_observations.parquet"
    table_path.write_bytes(table_path.read_bytes() + b"tampered")
    with pytest.raises(StorageIntegrityError, match="content hash mismatch"):
        store.read(SILVER_MONTHLY)
