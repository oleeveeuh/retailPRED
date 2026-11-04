"""Governed scoring task: promotion gate, point-in-time gold, benchmark.

Reads the versioned silver table from Delta, refuses when the frozen
portfolio has not authorized the champion (the expected state before the
vintage gate passes), and otherwise refits the promoted model at the scoring
moment, freezes the seasonal-naive benchmark beside every forecast, and
replaces the gold forecast snapshot. The model is never persisted; every run
refits from governed data.
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


TASK_NAME = "score"
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

from retailpred.live_loop.scoring import score_from_panel, scoring_decision  # noqa: E402
from retailpred.platform.delta_sync import to_spark_ready  # noqa: E402
from retailpred.platform.spark_delta import replace_delta_snapshot  # noqa: E402
from retailpred.platform.specs import GOLD_FORECASTS, SILVER_MONTHLY  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--job-run-id", default=None)
    parser.add_argument("--repo-root", default=".", help="Synced bundle root; consumed before imports.")
    parser.add_argument(
        "--end-year",
        type=int,
        default=datetime.now(timezone.utc).year,
        help="Only used to bound nothing; kept for interface symmetry.",
    )
    args = parser.parse_args()

    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    decision = scoring_decision("mrts")
    if decision["status"] != "authorized":
        print(
            f"Scoring refused (model={decision['model']}, "
            f"model_status={decision['model_status']}, "
            f"required_action={decision['required_action']})."
        )
        print("No forecasts were generated; gold forecast tables remain untouched.")
        return

    silver = spark.table(
        f"`{args.catalog}`.`silver`.`{SILVER_MONTHLY.name}`"
    ).toPandas()
    revision = (
        f"databricks-job:{args.job_run_id}" if args.job_run_id else "databricks-job:manual"
    )
    forecasts = score_from_panel(silver, now=datetime.now(timezone.utc), revision=revision)
    replace_delta_snapshot(
        spark, to_spark_ready(forecasts), GOLD_FORECASTS, catalog=args.catalog
    )
    statuses = forecasts["seasonal_naive_status"].value_counts().to_dict()
    print(f"Forecast rows written: {len(forecasts):,}")
    print(f"Seasonal-naive benchmark statuses: {statuses}")


if __name__ == "__main__":
    main()
