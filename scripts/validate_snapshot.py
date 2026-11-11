#!/usr/bin/env python3
"""Validate published live-loop artifacts against their contracts, dependency-free.

Runs in CI before Pages deployment so an invalid artifact can never reach the
public site. Two artifact kinds are recognized by their own markers:

- the forecast snapshot (``contract_version`` 2.1): contract version, kind,
  live flag, freshness, series structure, point invariants, and
  benchmark-status consistency;
- the per-run ingest-status record (``schema_version`` "2.2-status") and the
  ``runs.jsonl`` log of such records: required gate fields, is_live
  consistency, and the absolute absence of forecast-claim keys — a status
  file makes no model claims, and its presence or absence authorizes
  nothing.

A deploy with a status file but no snapshot passes: the snapshot's absence
is the honest fail-closed state, not an error.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

STATUS_SCHEMA_VERSION = "2.2-status"

# Keys that would turn the mechanical status record into a forecast or
# metric claim. Forbidden at every level of the payload.
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


def validate(payload: object) -> list[str]:
    """Validate a forecast snapshot payload against contract 2.1."""
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["snapshot must be a JSON object"]

    if payload.get("contract_version") != "2.1":
        errors.append(f"contract_version must be '2.1', got {payload.get('contract_version')!r}")
    if payload.get("snapshot_kind") != "scheduled_batch_forecasts":
        errors.append(f"snapshot_kind must be 'scheduled_batch_forecasts', got {payload.get('snapshot_kind')!r}")
    if payload.get("dataset") != "mrts":
        errors.append(f"dataset must be 'mrts', got {payload.get('dataset')!r}")
    if not isinstance(payload.get("is_live"), bool):
        errors.append("is_live must be a boolean")
    for required in ("model_name", "model_version", "model_status", "uncertainty_status", "use_case"):
        if not isinstance(payload.get(required), str) or not payload.get(required):
            errors.append(f"{required} must be a non-empty string")

    freshness = payload.get("freshness")
    if not isinstance(freshness, dict):
        errors.append("freshness must be an object")
    else:
        if freshness.get("status") not in {"current", "stale"}:
            errors.append("freshness.status must be 'current' or 'stale'")
        for field in ("generated_at", "forecast_vintage_time", "data_cutoff"):
            if not isinstance(freshness.get(field), str) or not freshness.get(field):
                errors.append(f"freshness.{field} must be a non-empty string")
        if not isinstance(freshness.get("max_age_hours"), int) or freshness["max_age_hours"] < 1:
            errors.append("freshness.max_age_hours must be a positive integer")

    series = payload.get("series")
    if not isinstance(series, list) or not series:
        errors.append("series must be a non-empty list")
        return errors

    seen_ids: set[str] = set()
    for entry in series:
        if not isinstance(entry, dict):
            errors.append("each series entry must be an object")
            continue
        series_id = entry.get("series_id")
        if not isinstance(series_id, str) or not series_id:
            errors.append("series_id must be a non-empty string")
            continue
        if series_id in seen_ids:
            errors.append(f"duplicate series_id {series_id!r}")
        seen_ids.add(series_id)
        points = entry.get("points")
        if not isinstance(points, list) or not points:
            errors.append(f"series {series_id!r} must have a non-empty points list")
            continue
        for point in points:
            if not isinstance(point, dict):
                errors.append(f"series {series_id!r} has a non-object point")
                continue
            horizon = point.get("horizon")
            if not isinstance(horizon, int) or horizon < 1:
                errors.append(f"series {series_id!r} point horizon must be a positive integer")
            prediction = point.get("prediction")
            if not isinstance(prediction, (int, float)) or isinstance(prediction, bool) or prediction < 0:
                errors.append(f"series {series_id!r} point prediction must be a non-negative number")
            status = point.get("seasonal_naive_status")
            if status not in {"available", "unavailable"}:
                errors.append(
                    f"series {series_id!r} point seasonal_naive_status must be "
                    f"'available' or 'unavailable', got {status!r}"
                )
            elif status == "unavailable" and point.get("seasonal_naive_prediction") is not None:
                errors.append(
                    f"series {series_id!r} point marks the benchmark unavailable "
                    "but still carries a seasonal_naive_prediction"
                )
            elif status == "available" and not isinstance(
                point.get("seasonal_naive_prediction"), (int, float)
            ):
                errors.append(
                    f"series {series_id!r} point marks the benchmark available "
                    "without a seasonal_naive_prediction"
                )
    return errors


def _find_claim_keys(payload: object, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(key, str) and key in FORBIDDEN_CLAIM_KEYS:
                found.append(path)
            found.extend(_find_claim_keys(value, path))
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            found.extend(_find_claim_keys(item, f"{prefix}[{index}]"))
    return found


def validate_status_record(payload: object, source: str) -> list[str]:
    """Validate one 2.2-status record: gates present, zero forecast claims."""
    errors: list[str] = []
    if not isinstance(payload, dict):
        return [f"{source}: status record must be a JSON object"]

    if payload.get("schema_version") != STATUS_SCHEMA_VERSION:
        errors.append(
            f"{source}: schema_version must be '{STATUS_SCHEMA_VERSION}', "
            f"got {payload.get('schema_version')!r}"
        )
    for field in ("pipeline_run_id", "job_id"):
        if not isinstance(payload.get(field), (str, type(None))):
            errors.append(f"{source}: {field} must be a string or null")
    for field in ("pipeline_started_at", "finished_at", "generated_at"):
        value = payload.get(field)
        if field == "pipeline_started_at":
            if not isinstance(value, (str, type(None))):
                errors.append(f"{source}: {field} must be a string or null")
        elif not isinstance(value, str) or not value:
            errors.append(f"{source}: {field} must be a non-empty string")

    ingest = payload.get("ingest")
    if not isinstance(ingest, dict):
        errors.append(f"{source}: ingest must be an object")
    else:
        for field in ("rows_in_delta", "series_count"):
            value = ingest.get(field)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
                errors.append(f"{source}: ingest.{field} must be a non-negative integer or null")
        cutoff = ingest.get("data_cutoff")
        if cutoff is not None and (not isinstance(cutoff, str) or not cutoff):
            errors.append(f"{source}: ingest.data_cutoff must be a non-empty string or null")

    gates = payload.get("gates")
    if not isinstance(gates, dict):
        errors.append(f"{source}: gates must be an object with ingest, score, realized, publish")
    else:
        if gates.get("ingest") not in {"success", "failed"}:
            errors.append(f"{source}: gates.ingest must be 'success' or 'failed'")
        score = gates.get("score")
        if not isinstance(score, dict):
            errors.append(f"{source}: gates.score must be an object")
        else:
            if score.get("status") not in {"authorized", "refused"}:
                errors.append(f"{source}: gates.score.status must be 'authorized' or 'refused'")
            for field in ("model", "model_status", "required_action"):
                if not isinstance(score.get(field), str) or not score.get(field):
                    errors.append(f"{source}: gates.score.{field} must be a non-empty string")
        realized = gates.get("realized")
        if not isinstance(realized, dict):
            errors.append(f"{source}: gates.realized must be an object")
        else:
            if realized.get("status") not in {"evaluated", "skipped"}:
                errors.append(f"{source}: gates.realized.status must be 'evaluated' or 'skipped'")
            rows = realized.get("realized_rows")
            if not isinstance(rows, int) or isinstance(rows, bool) or rows < 0:
                errors.append(f"{source}: gates.realized.realized_rows must be a non-negative integer")
        if gates.get("publish") not in {"published", "declined", "skipped-identical"}:
            errors.append(
                f"{source}: gates.publish must be 'published', 'declined', or 'skipped-identical'"
            )

    is_live = payload.get("is_live")
    if not isinstance(is_live, bool):
        errors.append(f"{source}: is_live must be a boolean")
    elif is_live and isinstance(gates, dict):
        # Mirror the snapshot contract: is_live reflects the recurring
        # schedule having executed end to end — a record may only carry
        # is_live=true when this run ingested and completed publication.
        if gates.get("ingest") != "success" or gates.get("publish") not in {
            "published",
            "skipped-identical",
        }:
            errors.append(
                f"{source}: is_live=true requires a successful ingest and a completed "
                "publish; the schedule flag cannot precede its own evidence"
            )

    claims = _find_claim_keys(payload)
    if claims:
        errors.append(
            f"{source}: status record carries forbidden forecast-claim keys "
            f"{claims} — it is a status report, not a model claim"
        )
    return errors


def validate_runs_log(text: str, source: str) -> list[str]:
    errors: list[str] = []
    line_count = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        line_count += 1
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            errors.append(f"{source}: line {line_count} is not valid JSON ({error})")
            continue
        errors.extend(validate_status_record(payload, f"{source} line {line_count}"))
    if line_count == 0:
        errors.append(f"{source}: run log is empty")
    return errors


def _validate_path(path: Path) -> list[str]:
    try:
        if path.suffix == ".jsonl":
            return validate_runs_log(path.read_text(encoding="utf-8"), str(path))
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"{path} is unreadable: {error}"]
    if isinstance(payload, dict) and payload.get("schema_version") == STATUS_SCHEMA_VERSION:
        return validate_status_record(payload, str(path))
    errors = validate(payload)
    return [f"{path}: {error}" for error in errors]


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: validate_snapshot.py <artifact.json> [<more.json|.jsonl>...]", file=sys.stderr)
        return 1
    all_errors: list[str] = []
    for argument in sys.argv[1:]:
        all_errors.extend(_validate_path(Path(argument)))
    if all_errors:
        print("Live-loop artifact validation failed:")
        for error in all_errors:
            print(f"- {error}")
        return 1
    print(f"Validated {len(sys.argv) - 1} artifact(s) against their contracts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
