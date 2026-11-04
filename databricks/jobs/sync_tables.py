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


# Serverless python tasks exec this file without defining __file__, so the
# bundle passes the synced repository root as an explicit parameter
# (${workspace.file_path}, which already ends in /files), validated here,
# with a workspace-layout fallback and diagnostics because the import
# depends on it.
def _repository_root() -> str:
    argv = sys.argv
    if "--repo-root" in argv:
        root = Path(argv[argv.index("--repo-root") + 1])
        if (root / "retailpred").is_dir():
            return str(root)
        print(f"[{TASK_NAME}] --repo-root given but no retailpred package inside: {root}")
    try:
        for candidate in sorted(Path("/Workspace/Users").glob("*/.bundle/*/dev/files")):
            if (candidate / "retailpred").is_dir():
                print(f"[{TASK_NAME}] fallback located repository root: {candidate}")
                return str(candidate)
    except OSError as error:
        print(f"[{TASK_NAME}] /Workspace/Users not listable: {error}")
    return "."


TASK_NAME = "sync"
REPOSITORY_ROOT = Path(_repository_root())
print(
    f"[{TASK_NAME}] repository root: {REPOSITORY_ROOT} "
    f"exists={REPOSITORY_ROOT.exists()} "
    f"package={(REPOSITORY_ROOT / 'retailpred').is_dir()}"
)
if not (REPOSITORY_ROOT / "retailpred").is_dir():
    try:
        print(
            f"[{TASK_NAME}] /Workspace/Users entries:",
            sorted(path.name for path in Path("/Workspace/Users").iterdir())[:10],
        )
    except OSError as error:
        print(f"[{TASK_NAME}] /Workspace/Users not listable: {error}")
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
    parser.add_argument("--repo-root", default=".", help="Synced bundle root; consumed before imports.")
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
