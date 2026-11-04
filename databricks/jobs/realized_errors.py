"""Append realized-error rows to the declared Delta gold table.

Reads the published gold forecasts and versioned silver observations from
Unity Catalog, scores every forecast target that now has a revised actual,
and merges the immutable realized-error rows. Safe to run on any schedule:
exact replays are no-ops and revised actuals append new rows.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
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


TASK_NAME = "realized"
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

import pandas as pd  # noqa: E402

from retailpred.live_loop.realized import compute_realized_errors  # noqa: E402
from retailpred.platform.delta_sync import to_spark_ready  # noqa: E402
from retailpred.platform.spark_delta import merge_delta_immutable  # noqa: E402
from retailpred.platform.specs import (  # noqa: E402
    GOLD_FORECASTS,
    GOLD_REALIZED_ERRORS,
    SILVER_MONTHLY,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--repo-root", default=".", help="Synced bundle root; consumed before imports.")
    args = parser.parse_args()

    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    forecasts = spark.table(
        f"`{args.catalog}`.`gold`.`{GOLD_FORECASTS.name}`"
    ).toPandas()
    if forecasts.empty:
        print("No published forecasts exist yet; realized monitoring is skipped.")
        return
    silver = spark.table(
        f"`{args.catalog}`.`silver`.`{SILVER_MONTHLY.name}`"
    ).toPandas()

    errors = compute_realized_errors(
        forecasts, silver, now=datetime.now(timezone.utc)
    )
    if errors.empty:
        print("No forecast targets have revised actuals yet; nothing to append.")
        return

    merge_delta_immutable(
        spark, to_spark_ready(errors), GOLD_REALIZED_ERRORS, catalog=args.catalog
    )
    total = spark.table(
        f"`{args.catalog}`.`gold`.`{GOLD_REALIZED_ERRORS.name}`"
    ).count()
    print(f"Appended realized-error rows; table now holds {total:,} rows.")


if __name__ == "__main__":
    main()
