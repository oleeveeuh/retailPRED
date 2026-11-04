"""Synchronize contract Parquet tables staged in a workspace volume to Delta.

The local pipeline writes ``<layer>/<table>.parquet`` plus manifests under
``data/v2/store``. Upload that directory into a Unity Catalog volume, then
run this job to merge immutable bronze/silver facts and replace gold
snapshots inside the declared Delta tables.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from retailpred.platform.delta_sync import sync_store  # noqa: E402
from retailpred.platform.local_store import LocalTableStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument(
        "--staged-root",
        required=True,
        help="Volume path containing <layer>/<table>.parquet from the local store",
    )
    args = parser.parse_args()

    store = LocalTableStore(args.staged_root)
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    actions = sync_store(spark, store, catalog=args.catalog)
    for action in actions:
        print(f"Synced {action.description}")
    if not actions:
        print("No local tables were staged; nothing was synchronized.")


if __name__ == "__main__":
    main()
