#!/usr/bin/env python3
"""Run one live-loop ingestion against the local contract store.

Pulls the latest Census release into versioned silver and scores only if the
frozen portfolio authorizes it. Requires CENSUS_API_KEY in the environment;
failed pulls never substitute generated data.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

from forecasting.data.mrts import MRTSClient
from retailpred.live_loop.ingest import code_revision, run_live_ingest
from retailpred.platform.local_store import LocalTableStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=Path("data/v2/store"))
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument(
        "--end-year",
        type=int,
        default=datetime.now(timezone.utc).year,
        help="Latest Census year to request; defaults to the current year.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = MRTSClient(os.getenv("CENSUS_API_KEY"))
    store = LocalTableStore(args.store)
    outcome = run_live_ingest(
        store,
        client,
        start_year=args.start_year,
        end_year=args.end_year,
        now=datetime.now(timezone.utc),
        revision=code_revision(),
    )
    print(f"Silver rows after merge: {outcome.silver_rows:,}")
    print(f"Data cutoff: {outcome.data_cutoff}")
    scoring = outcome.scoring
    print(
        f"Scoring: {scoring['status']} (model={scoring['model']}, "
        f"model_status={scoring['model_status']}, "
        f"required_action={scoring['required_action']})"
    )
    if scoring["status"] != "authorized":
        print("No forecasts were generated; gold forecast tables remain untouched.")


if __name__ == "__main__":
    main()
