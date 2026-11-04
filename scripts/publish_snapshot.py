#!/usr/bin/env python3
"""Publish the v2 latest-forecast snapshot from gold forecasts in the local store.

Fails closed when no gold forecasts exist: until the frozen portfolio
authorizes MRTS scoring, there is nothing to publish. After a successful
build, upload the file to your public host and point the frontend at it via
``VITE_LATEST_SNAPSHOT_URL``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from retailpred.platform.local_store import LocalTableStore
from retailpred.platform.snapshot import build_snapshot, write_snapshot
from retailpred.platform.specs import GOLD_FORECASTS, GOLD_REALIZED_ERRORS, SILVER_MONTHLY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=Path("data/v2/store"))
    parser.add_argument("--output", type=Path, default=Path("data/v2/snapshots/v2-latest.json"))
    parser.add_argument("--max-age-hours", type=int, default=336)
    parser.add_argument("--next-run", type=datetime.fromisoformat, default=None)
    parser.add_argument("--job-run-id", default=None)
    parser.add_argument(
        "--is-live",
        action="store_true",
        help="Set only after the recurring schedule has executed end to end.",
    )
    return parser.parse_args()


def _code_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _manifest_sha256(store: LocalTableStore, layer: str, name: str) -> str | None:
    path = store.root / layer / f"{name}.manifest.json"
    if not path.exists():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return hashlib.sha256(f"{name}|{manifest['sha256']}".encode()).hexdigest()


def main() -> None:
    args = parse_args()
    store = LocalTableStore(args.store)
    forecasts = store.read(GOLD_FORECASTS)
    if forecasts.empty:
        raise SystemExit(
            "no gold forecasts exist in the store; the frozen portfolio has not "
            "authorized MRTS scoring yet, so snapshot publishing is unavailable"
        )

    provenance = {
        "code_revision": _code_revision(),
        "silver_manifest_sha256": _manifest_sha256(store, "silver", SILVER_MONTHLY.name) or "unknown",
        "forecasts_manifest_sha256": _manifest_sha256(store, "gold", GOLD_FORECASTS.name) or "unknown",
    }
    if args.job_run_id:
        provenance["job_run_id"] = args.job_run_id

    next_run = args.next_run
    if next_run is not None and next_run.tzinfo is None:
        raise SystemExit("--next-run must include a timezone offset")
    realized = store.read(GOLD_REALIZED_ERRORS)
    snapshot = build_snapshot(
        forecasts,
        generated_at=datetime.now(timezone.utc),
        is_live=args.is_live,
        max_age_hours=args.max_age_hours,
        next_scheduled_run=next_run,
        pipeline_run_id=args.job_run_id,
        provenance=provenance,
        realized=realized,
    )
    destination = write_snapshot(snapshot, args.output)
    print(f"Snapshot written to {destination}")
    print("Publish it to your public host, then set VITE_LATEST_SNAPSHOT_URL to its URL.")
    print(f"Freshness status: {snapshot.freshness.status}")


if __name__ == "__main__":
    main()
