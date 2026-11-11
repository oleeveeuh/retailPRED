"""Per-run ingest-status record: contract ``2.2-status``.

A STATUS RECORD, not a forecast claim. The pipeline writes one JSON file
(``live/v2-ingest-status.json``) and one append-only run log
(``live/runs.jsonl``) on every completed publish-task run — including runs
where the forecast snapshot declines. The record carries run mechanics and
gate verdicts only: no predictions, no benchmark values, no model metrics.
Its presence or absence authorizes nothing; the fail-closed snapshot
contract is untouched and the snapshot still declines without gold
forecasts.

The record is assembled by the publish task from the run's own Delta reads
and the frozen portfolio decision — never from invented values. Databricks
job-identity macros are guarded: an unresolved ``{{...}}`` placeholder is
recorded as null, never as literal text.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

INGEST_STATUS_SCHEMA_VERSION = "2.2-status"
DEFAULT_STATUS_PATH = "live/v2-ingest-status.json"
DEFAULT_RUNS_PATH = "live/runs.jsonl"

# Keys that would turn this mechanical record into a forecast or metric
# claim. They are forbidden at every level of the payload; validation fails
# closed if any appears.
FORBIDDEN_CLAIM_KEYS = frozenset(
    {
        "prediction",
        "predictions",
        "forecast",
        "forecasts",
        "series",
        "points",
        "lower",
        "upper",
        "seasonal_naive_prediction",
        "evaluation_actual",
        "model_absolute_error",
        "seasonal_naive_absolute_error",
        "model_beat_seasonal_naive",
        "wape",
        "wrmsse",
        "mape",
        "accuracy",
    }
)


class IngestObservation(BaseModel):
    """What this run's merge left in Delta; null when the run cannot read it."""

    rows_in_delta: int | None = Field(default=None, ge=0)
    data_cutoff: str | None = None
    series_count: int | None = Field(default=None, ge=0)


class ScoreGate(BaseModel):
    """The frozen portfolio decision, verbatim from ``scoring_decision``."""

    model_config = ConfigDict(protected_namespaces=())

    status: Literal["authorized", "refused"]
    model: str
    model_status: str
    required_action: str


class RealizedGate(BaseModel):
    """Realized-error monitoring state; ``realized_rows`` is the immutable
    table's total row count, not a per-run append count."""

    status: Literal["evaluated", "skipped"]
    realized_rows: int = Field(ge=0)


class IngestStatusRecord(BaseModel):
    """Contract ``2.2-status``: mechanics and verdicts, nothing else."""

    model_config = ConfigDict(protected_namespaces=(), extra="forbid")

    schema_version: Literal["2.2-status"] = INGEST_STATUS_SCHEMA_VERSION
    pipeline_run_id: str | None = None
    job_id: str | None = None
    pipeline_started_at: datetime | None = None
    finished_at: datetime
    ingest: IngestObservation
    gates: "Gates"
    is_live: bool
    generated_at: datetime


class Gates(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ingest: Literal["success", "failed"]
    score: ScoreGate
    realized: RealizedGate
    publish: Literal["published", "declined", "skipped-identical"]


def clean_macro(value: str | None) -> str | None:
    """Return the value, or None when missing or an unresolved Databricks macro.

    Job-parameter references such as ``{{job.run_id}}`` are resolved by the
    scheduler; if one ever arrives literally, the record carries null rather
    than template text.
    """
    if value is None:
        return None
    stripped = value.strip()
    if not stripped or "{{" in stripped or "}}" in stripped:
        return None
    return stripped


def started_at_from_epoch_ms(value: str | None) -> datetime | None:
    """Parse the job run's ``{{start_time}}`` epoch-milliseconds macro."""
    cleaned = clean_macro(value)
    if cleaned is None or not cleaned.isdigit():
        return None
    return datetime.fromtimestamp(int(cleaned) / 1000, tz=timezone.utc)


def find_forbidden_claim_keys(payload: object) -> list[str]:
    """Every forbidden claim key anywhere in a decoded status payload."""
    found: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(key, str) and key in FORBIDDEN_CLAIM_KEYS:
                found.append(key)
            found.extend(find_forbidden_claim_keys(value))
    elif isinstance(payload, list):
        for item in payload:
            found.extend(find_forbidden_claim_keys(item))
    return found


def assemble_status(
    *,
    finished_at: datetime,
    is_live: bool,
    ingest_gate: Literal["success", "failed"],
    ingest: IngestObservation,
    score: ScoreGate,
    realized: RealizedGate,
    publish: str,
    pipeline_run_id: str | None = None,
    job_id: str | None = None,
    pipeline_started_at: datetime | None = None,
) -> IngestStatusRecord:
    """Assemble and validate the run's status record; fails closed.

    ``ingest_gate="failed"`` requires null observation fields — stale table
    state must never be presented as this run's merge — and ``"success"``
    requires the complete observation triple.
    """
    if publish not in {"published", "declined", "skipped-identical"}:
        raise ValueError(f"unknown publish verdict: {publish!r}")
    observed = (
        ingest.rows_in_delta is not None,
        ingest.data_cutoff is not None,
        ingest.series_count is not None,
    )
    if ingest_gate == "failed" and any(observed):
        raise ValueError("a failed ingest gate must carry null observations, not stale reads")
    if ingest_gate == "success" and not all(observed):
        raise ValueError("a successful ingest gate requires the complete observation triple")
    record = IngestStatusRecord(
        pipeline_run_id=clean_macro(pipeline_run_id),
        job_id=clean_macro(job_id),
        pipeline_started_at=pipeline_started_at,
        finished_at=finished_at,
        ingest=ingest,
        gates=Gates(
            ingest=ingest_gate,  # type: ignore[arg-type]
            score=score,
            realized=realized,
            publish=publish,  # type: ignore[arg-type]
        ),
        is_live=is_live,
        generated_at=finished_at,
    )
    claims = find_forbidden_claim_keys(record.model_dump(mode="json"))
    if claims:
        raise ValueError(f"status record carries forbidden claim keys: {sorted(set(claims))}")
    return record


def write_status(record: IngestStatusRecord, path: str | Path) -> Path:
    """Write the status record atomically so readers never see a partial file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    temporary.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


def run_log_line(record: IngestStatusRecord) -> str:
    """The single compact JSON line appended to ``live/runs.jsonl``."""
    return record.model_dump_json()
