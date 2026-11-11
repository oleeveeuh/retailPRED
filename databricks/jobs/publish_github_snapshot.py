"""Publish the snapshot and the per-run status record to ``live-data``.

Assembles the contract-validated snapshot from the Delta gold tables and the
contract ``2.2-status`` ingest-status record from the run's own reads plus
the frozen portfolio decision, and commits them to the dedicated data branch
through the GitHub Contents API. The status record is written on EVERY
completed run — including when the snapshot declines — and carries run
mechanics and gate verdicts only: no forecasts, no benchmark values, no
model metrics. The fine-grained credential lives in the secret scope and
never appears in output. Identical content is skipped.

``--dry-run`` assembles and prints the status record without writing to
GitHub; pair it with ``--dry-run-input`` to exercise the assembly from a
JSON fixture instead of live Spark tables (no hosted run required). Fixture
mode is a pure assembly projection: it never reads or commits snapshots.
"""

from __future__ import annotations

import argparse
import json
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

from retailpred.live_loop.publish_github import append_run_line, commit_snapshot  # noqa: E402
from retailpred.live_loop.scoring import scoring_decision  # noqa: E402
from retailpred.live_loop.status import (  # noqa: E402
    INGEST_STATUS_SCHEMA_VERSION,
    IngestObservation,
    RealizedGate,
    ScoreGate,
    assemble_status,
    run_log_line,
    started_at_from_epoch_ms,
    write_status,
)
from retailpred.platform.snapshot import build_snapshot, write_snapshot  # noqa: E402
from retailpred.platform.specs import (  # noqa: E402
    GOLD_FORECASTS,
    GOLD_REALIZED_ERRORS,
    SILVER_MONTHLY,
)

STAGING_ROOT = Path("/tmp") / "retailpred-snapshot-staging"

FIXTURE_KEYS = frozenset(
    {"rows_in_delta", "data_cutoff", "series_count", "forecasts_empty", "realized_rows", "publish_verdict"}
)


def resolve_publisher_token(secret_scope: str) -> str:
    from databricks.sdk.runtime import dbutils

    return dbutils.secrets.get(scope=secret_scope, key="github-publisher-token")


def _fixture_observations(path: str) -> tuple[IngestObservation, bool, int, str, str]:
    """Load table observations from a JSON fixture for --dry-run (no Spark).

    Unknown fixture keys fail closed — a typo must never silently become a
    null observation. Returns the observation, whether gold forecasts are
    empty, the realized-row count, the ingest gate, and the projected
    publish verdict. Fixture mode assembles the status record only — it
    never builds or commits a snapshot.
    """
    fixture = json.loads(Path(path).read_text(encoding="utf-8"))
    unknown = set(fixture) - FIXTURE_KEYS
    if unknown:
        raise ValueError(f"dry-run fixture carries unknown keys: {sorted(unknown)}")
    observation = IngestObservation(
        rows_in_delta=fixture.get("rows_in_delta"),
        data_cutoff=fixture.get("data_cutoff"),
        series_count=fixture.get("series_count"),
    )
    complete = all(
        (
            observation.rows_in_delta is not None,
            observation.data_cutoff is not None,
            observation.series_count is not None,
        )
    )
    verdict = fixture.get("publish_verdict", "declined")
    return (
        observation,
        bool(fixture["forecasts_empty"]),
        int(fixture["realized_rows"]),
        "success" if complete else "failed",
        verdict,
    )


def _delta_observations(catalog: str) -> tuple[IngestObservation, object, int]:
    """Read this run's post-merge table state from Delta."""
    import pyspark.sql.functions as F
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    silver_ref = f"`{catalog}`.`silver`.`{SILVER_MONTHLY.name}`"
    stats = (
        spark.table(silver_ref)
        .agg(
            F.count("*").alias("rows"),
            F.max("observation_time").alias("cutoff"),
            F.countDistinct("series_id").alias("series"),
        )
        .collect()[0]
    )
    observation = IngestObservation(
        rows_in_delta=int(stats["rows"]),
        data_cutoff=stats["cutoff"].date().isoformat(),
        series_count=int(stats["series"]),
    )
    forecasts = spark.table(f"`{catalog}`.`gold`.`{GOLD_FORECASTS.name}`").toPandas()
    realized_rows = int(
        spark.table(f"`{catalog}`.`gold`.`{GOLD_REALIZED_ERRORS.name}`").count()
    )
    return observation, forecasts, realized_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--repo", default="oleeveeuh/retailPRED")
    parser.add_argument("--branch", default="live-data")
    parser.add_argument("--path", default="live/v2-latest.json")
    parser.add_argument("--status-path", default="live/v2-ingest-status.json")
    parser.add_argument("--runs-path", default="live/runs.jsonl")
    parser.add_argument("--secret-scope", default="retailpred")
    parser.add_argument("--job-run-id", default=None)
    parser.add_argument("--job-id", default=None)
    parser.add_argument(
        "--pipeline-started-at-ms",
        default=None,
        help="Job run start ({{start_time}} macro); unresolved macros record as null.",
    )
    parser.add_argument("--repo-root", default=".", help="Synced bundle root; consumed before imports.")
    parser.add_argument(
        "--is-live",
        action="store_true",
        help="Set only after the recurring schedule has executed end to end.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Assemble and print the status record; write nothing to GitHub.",
    )
    parser.add_argument(
        "--dry-run-input",
        default=None,
        help="JSON fixture with table observations; skips Spark entirely.",
    )
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    started_at = started_at_from_epoch_ms(args.pipeline_started_at_ms)
    fixture_mode = args.dry_run and args.dry_run_input is not None

    if fixture_mode:
        (
            ingest_obs,
            forecasts_empty,
            realized_rows,
            ingest_gate,
            publish_verdict,
        ) = _fixture_observations(args.dry_run_input)
    else:
        ingest_obs, forecasts, realized_rows = _delta_observations(args.catalog)
        ingest_gate = "success"
        forecasts_empty = forecasts.empty

    decision = scoring_decision("mrts")
    score_gate = ScoreGate(
        status=decision["status"],
        model=decision["model"],
        model_status=decision["model_status"],
        required_action=decision["required_action"],
    )
    realized_gate = RealizedGate(
        status="evaluated" if (not forecasts_empty and realized_rows > 0) else "skipped",
        realized_rows=realized_rows,
    )

    if not fixture_mode:
        if forecasts.empty:
            publish_verdict = "declined"
            print(
                "No gold forecasts exist; the frozen portfolio has not authorized "
                "scoring yet. Nothing was published."
            )
        else:
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
            staging = STAGING_ROOT / args.path.replace("/", "_")
            write_snapshot(snapshot, staging)
            content = staging.read_bytes()
            if args.dry_run:
                publish_verdict = "published"
            else:
                import requests

                result = commit_snapshot(
                    requests.Session(),
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
                publish_verdict = (
                    "published"
                    if result.action in {"created", "updated"}
                    else "skipped-identical"
                )

    record = assemble_status(
        finished_at=datetime.now(timezone.utc),
        is_live=args.is_live,
        ingest_gate=ingest_gate,
        ingest=ingest_obs,
        score=score_gate,
        realized=realized_gate,
        publish=publish_verdict,
        pipeline_run_id=args.job_run_id,
        job_id=args.job_id,
        pipeline_started_at=started_at,
    )

    if args.dry_run:
        print(
            f"[{TASK_NAME}] dry-run projection ({INGEST_STATUS_SCHEMA_VERSION}); "
            "nothing was committed to GitHub:"
        )
        print(record.model_dump_json(indent=2))
        return

    status_staging = STAGING_ROOT / args.status_path.replace("/", "_")
    write_status(record, status_staging)
    import requests

    session = requests.Session()
    token = resolve_publisher_token(args.secret_scope)
    status_result = commit_snapshot(
        session,
        repo=args.repo,
        branch=args.branch,
        path=args.status_path,
        token=token,
        content=status_staging.read_bytes(),
        message=f"ingest status from pipeline run {args.job_run_id or 'manual'}",
    )
    print(f"Status publish action: {status_result.action}")
    runs_result = append_run_line(
        session,
        repo=args.repo,
        branch=args.branch,
        path=args.runs_path,
        token=token,
        line=run_log_line(record),
        message=f"run record: pipeline run {args.job_run_id or 'manual'}",
    )
    print(f"Run-log publish action: {runs_result.action}")


if __name__ == "__main__":
    main()
