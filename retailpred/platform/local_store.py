"""Small local Parquet implementation of the shared lakehouse table contracts."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd

from retailpred.platform.specs import TableSpec


class SchemaContractError(ValueError):
    pass


class MergeConflictError(ValueError):
    pass


class StorageIntegrityError(RuntimeError):
    pass


def validate_table_schema(frame: pd.DataFrame, spec: TableSpec) -> pd.DataFrame:
    missing = set(spec.columns).difference(frame.columns)
    extra = set(frame.columns).difference(spec.columns)
    if missing or extra:
        raise SchemaContractError(
            f"{spec.name} schema mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    result = frame.loc[:, spec.columns].copy()
    required = [column for column in spec.columns if column not in spec.nullable]
    if result[required].isna().any().any():
        raise SchemaContractError(f"{spec.name} contains nulls in required columns")
    if result[list(spec.key)].duplicated().any():
        raise SchemaContractError(f"{spec.name} contains duplicate keys")
    return result


def merge_immutable(existing: pd.DataFrame, incoming: pd.DataFrame, spec: TableSpec) -> pd.DataFrame:
    """Insert new immutable facts; exact replays are no-ops and conflicts fail."""
    current = validate_table_schema(existing, spec) if not existing.empty else existing
    new = validate_table_schema(incoming, spec)
    if current.empty:
        return new.sort_values(list(spec.key), ignore_index=True)

    combined = pd.concat(
        [current.assign(_batch="existing"), new.assign(_batch="incoming")],
        ignore_index=True,
    )
    compare_columns = list(spec.columns)
    for _, group in combined.groupby(list(spec.key), dropna=False, sort=False):
        if len(group[compare_columns].drop_duplicates()) > 1:
            key_values = tuple(group.iloc[0][column] for column in spec.key)
            raise MergeConflictError(f"conflicting immutable fact for {spec.name} key={key_values}")
    result = combined.drop_duplicates(subset=list(spec.key), keep="first")
    return result.loc[:, spec.columns].sort_values(list(spec.key), ignore_index=True)


class LocalTableStore:
    """Persist contract-validated tables with an auditable manifest."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _table_path(self, spec: TableSpec) -> Path:
        return self.root / spec.layer / f"{spec.name}.parquet"

    def _manifest_path(self, spec: TableSpec) -> Path:
        return self.root / spec.layer / f"{spec.name}.manifest.json"

    def read(self, spec: TableSpec) -> pd.DataFrame:
        path = self._table_path(spec)
        if not path.exists():
            return pd.DataFrame(columns=spec.columns)
        manifest_path = self._manifest_path(spec)
        if not manifest_path.exists():
            raise StorageIntegrityError(f"missing manifest for {spec.layer}.{spec.name}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema_version") != spec.schema_version:
            raise StorageIntegrityError(
                f"schema version mismatch for {spec.name}: "
                f"stored={manifest.get('schema_version')}, expected={spec.schema_version}"
            )
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if manifest.get("sha256") != actual_hash:
            raise StorageIntegrityError(f"content hash mismatch for {spec.layer}.{spec.name}")
        result = validate_table_schema(pd.read_parquet(path), spec)
        if manifest.get("row_count") != len(result):
            raise StorageIntegrityError(f"row count mismatch for {spec.layer}.{spec.name}")
        return result

    def merge(self, spec: TableSpec, incoming: pd.DataFrame) -> pd.DataFrame:
        existing = self.read(spec)
        merged = merge_immutable(existing, incoming, spec)
        if not existing.empty and merged.equals(existing):
            return existing
        self._write(spec, merged)
        return merged

    def replace_snapshot(self, spec: TableSpec, frame: pd.DataFrame) -> pd.DataFrame:
        validated = validate_table_schema(frame, spec).sort_values(list(spec.key), ignore_index=True)
        self._write(spec, validated)
        return validated

    def _write(self, spec: TableSpec, frame: pd.DataFrame) -> None:
        table_path = self._table_path(spec)
        manifest_path = self._manifest_path(spec)
        table_path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=table_path.parent, suffix=".parquet", delete=False) as handle:
            temporary_table = Path(handle.name)
        try:
            frame.to_parquet(temporary_table, index=False)
            content_hash = hashlib.sha256(temporary_table.read_bytes()).hexdigest()
            os.replace(temporary_table, table_path)
        finally:
            if temporary_table.exists():
                temporary_table.unlink()

        manifest = {
            "table": spec.name,
            "layer": spec.layer,
            "schema_version": spec.schema_version,
            "row_count": len(frame),
            "key": list(spec.key),
            "sha256": content_hash,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        with NamedTemporaryFile(
            mode="w", dir=manifest_path.parent, suffix=".json", delete=False
        ) as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
            temporary_manifest = Path(handle.name)
        os.replace(temporary_manifest, manifest_path)
