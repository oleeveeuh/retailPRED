#!/usr/bin/env python3
"""Validate a published snapshot JSON against the v2 contract, dependency-free.

Runs in CI before Pages deployment so an invalid snapshot can never reach the
public site. Schema-level only: contract version, kind, live flag, freshness,
series structure, point invariants, and benchmark-status consistency.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def validate(payload: object) -> list[str]:
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


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: validate_snapshot.py <snapshot.json>", file=sys.stderr)
        return 1
    path = Path(sys.argv[1])
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"Snapshot is unreadable: {error}")
        return 1
    errors = validate(payload)
    if errors:
        print(f"Snapshot {path} failed validation:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"Snapshot {path} matches contract 2.1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
