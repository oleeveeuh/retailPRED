"""Publish the public snapshot to the machine-written ``live-data`` branch.

Assembles the contract-validated snapshot from the Delta gold tables and
commits exactly one JSON file to the dedicated data branch through the
GitHub Contents API. The fine-grained credential lives in the secret scope
and never appears in output. Identical snapshots are skipped.
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


TASK_NAME = "publish"
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

from retailpred.live_loop.publish_github import commit_snapshot  # noqa: E402
from retailpred.platform.snapshot import build_snapshot, write_snapshot  # noqa: E402
from retailpred.platform.specs import GOLD_FORECASTS, GOLD_REALIZED_ERRORS  # noqa: E402


def resolve_publisher_token(secret_scope: str) -> str:
    from databricks.sdk.runtime import dbutils

    return dbutils.secrets.get(scope=secret_scope, key="github-publisher-token")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--repo", default="oleeveeuh/retailPRED")
    parser.add_argument("--branch", default="live-data")
    parser.add_argument("--path", default="live/v2-latest.json")
    parser.add_argument("--secret-scope", default="retailpred")
    parser.add_argument("--job-run-id", default=None)
    parser.add_argument("--repo-root", default=".", help="Synced bundle root; consumed before imports.")
    parser.add_argument(
        "--is-live",
        action="store_true",
        help="Set only after the recurring schedule has executed end to end.",
    )
    args = parser.parse_args()

    import requests
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    forecasts = spark.table(
        f"`{args.catalog}`.`gold`.`{GOLD_FORECASTS.name}`"
    ).toPandas()
    realized = spark.table(
        f"`{args.catalog}`.`gold`.`{GOLD_REALIZED_ERRORS.name}`"
    ).toPandas()
    if forecasts.empty:
        print(
            "No gold forecasts exist; the frozen portfolio has not authorized "
            "scoring yet. Nothing was published."
        )
        return
    now = datetime.now(timezone.utc)
    snapshot = build_snapshot(
        forecasts,
        generated_at=now,
        is_live=args.is_live,
        pipeline_run_id=args.job_run_id,
        provenance={
            "pipeline": "databricks-live-loop",
            "published_at": now.isoformat(),
        },
        realized=realized,
    )
    staging = Path("/tmp") / "retailpred-snapshot-staging" / args.path.replace("/", "_")
    write_snapshot(snapshot, staging)
    content = staging.read_bytes()

    session = requests.Session()
    result = commit_snapshot(
        session,
        repo=args.repo,
        branch=args.branch,
        path=args.path,
        token=resolve_publisher_token(args.secret_scope),
        content=content,
        message=f"live snapshot from pipeline run {args.job_run_id or 'manual'}",
    )
    print(f"Snapshot publish action: {result.action}")
    if result.commit_sha:
        print(f"Data-branch commit: {result.commit_sha[:12]}")


if __name__ == "__main__":
    main()
