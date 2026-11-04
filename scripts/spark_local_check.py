#!/usr/bin/env python3
"""Prove the Delta synchronization path against a local delta-spark session.

Writes small contract tables with ``LocalTableStore``, synchronizes them into
local Delta tables using the exact code path the Databricks job uses (immutable
merge for bronze/silver, snapshot replace for gold), reads them back, and
verifies row counts. Requires the Spark environment (``make setup-spark``) and
a Java 17 runtime.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from retailpred.platform.context import ExtractContext
from retailpred.platform.delta_sync import plan_sync, sync_store
from retailpred.platform.local_store import LocalTableStore
from retailpred.platform.mrts_pipeline import ingest_extract
from retailpred.platform.specs import GOLD_MONTHLY, SILVER_MONTHLY

TINY_PAYLOAD = [
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


def _code_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep", action="store_true", help="Keep the temporary Delta directory"
    )
    args = parser.parse_args()

    workspace = Path(tempfile.mkdtemp(prefix="retailpred-spark-check-"))
    try:
        store = LocalTableStore(workspace / "store")
        vintage = datetime(2025, 2, 14, tzinfo=timezone.utc)
        context = ExtractContext(
            extract_id="spark-local-check",
            source_url="https://example.invalid/mrts",
            vintage_time=vintage,
            ingested_at=vintage,
            release_time=vintage,
        )
        ingest_extract(store, TINY_PAYLOAD, context)

        from retailpred.platform.gold import build_gold_monthly

        gold = build_gold_monthly(store.read(SILVER_MONTHLY), as_of=vintage)
        store.replace_snapshot(GOLD_MONTHLY, gold)

        actions = plan_sync(store)
        print("Planned synchronization:")
        for action in actions:
            print(f"  - {action.description}")
        if {action.mode for action in actions} != {"merge_immutable", "replace_snapshot"}:
            raise RuntimeError("proof store does not cover both write modes")

        from retailpred.platform.spark_delta import configure_local_session

        spark = configure_local_session(str(workspace / "delta"))
        catalog = None  # local OSS session: default catalog, two-part table names
        executed = sync_store(spark, store, catalog=catalog)
        if [action.description for action in executed] != [
            action.description for action in actions
        ]:
            raise RuntimeError("executed sync diverged from the plan")

        for action in executed:
            table = f"{action.spec.layer}.{action.spec.name}"
            rows = spark.table(table).count()
            if rows != action.row_count:
                raise RuntimeError(f"{table} has {rows} rows; expected {action.row_count}")
            print(f"Verified {table}: {rows} rows")

        print(f"Local Delta proof passed (revision {_code_revision()[:12]}).")
    finally:
        if args.keep:
            print(f"Delta workspace retained at {workspace}")
        else:
            shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    main()
