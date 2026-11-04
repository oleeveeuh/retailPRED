"""Local-store to Delta synchronization driven by the shared table contracts.

Bronze and silver tables hold immutable facts, so synchronization uses the
same immutable-merge semantics as ``LocalTableStore.merge``. Gold tables are
derived point-in-time snapshots, so synchronization replaces them, mirroring
``LocalTableStore.replace_snapshot``. Every action is planned before any
Spark call so a dry run is possible without a Spark session.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from retailpred.platform.local_store import LocalTableStore, validate_table_schema
from retailpred.platform.specs import ALL_TABLE_SPECS, TableSpec
from retailpred.platform.spark_delta import (
    ensure_delta_table,
    merge_delta_immutable,
    replace_delta_snapshot,
)


def to_spark_ready(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize timezone-aware columns to naive UTC for spark.createDataFrame."""
    result = frame.copy()
    for column in result.columns:
        if pd.api.types.is_datetime64tz_dtype(result[column]):
            result[column] = result[column].dt.tz_convert("UTC").dt.tz_localize(None)
    return result


def _to_spark_frame(spark: Any, frame: pd.DataFrame, spec: TableSpec) -> Any:
    from retailpred.platform.spark_delta import contract_schema

    return spark.createDataFrame(to_spark_ready(frame), schema=contract_schema(spec))


@dataclass(frozen=True)
class SyncAction:
    spec: TableSpec
    mode: str
    row_count: int
    manifest_sha256: str

    @property
    def description(self) -> str:
        return f"{self.spec.layer}.{self.spec.name}: {self.mode} {self.row_count:,} rows"


def write_mode(spec: TableSpec) -> str:
    """Gold snapshots replace; bronze/silver immutable facts merge."""
    if spec.layer == "gold":
        return "replace_snapshot"
    return "merge_immutable"


def plan_sync(
    store: LocalTableStore,
    *,
    specs: tuple[TableSpec, ...] = ALL_TABLE_SPECS,
) -> list[SyncAction]:
    """Plan synchronization from local manifests without touching Spark."""
    actions: list[SyncAction] = []
    for spec in specs:
        path = store.root / spec.layer / f"{spec.name}.parquet"
        if not path.exists():
            continue
        frame = validate_table_schema(store.read(spec), spec)
        actions.append(
            SyncAction(
                spec=spec,
                mode=write_mode(spec),
                row_count=len(frame),
                manifest_sha256=_manifest_sha256(store, spec),
            )
        )
    return actions


def _manifest_sha256(store: LocalTableStore, spec: TableSpec) -> str:
    import hashlib
    import json

    manifest_path = store.root / spec.layer / f"{spec.name}.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return hashlib.sha256(
        f"{manifest['table']}|{manifest['schema_version']}|{manifest['sha256']}".encode(
            "utf-8"
        )
    ).hexdigest()


def sync_store(spark: Any, store: LocalTableStore, *, catalog: str | None) -> list[SyncAction]:
    """Execute the planned synchronization against Delta tables."""
    actions = plan_sync(store)
    for action in actions:
        frame = _to_spark_frame(spark, store.read(action.spec), action.spec)
        if action.mode == "replace_snapshot":
            replace_delta_snapshot(spark, frame, action.spec, catalog=catalog)
        else:
            merge_delta_immutable(spark, frame, action.spec, catalog=catalog)
    return actions
