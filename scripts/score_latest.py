#!/usr/bin/env python3
"""Run governed live scoring against the local contract store.

Checks the frozen portfolio promotion decision first and refuses when the
candidate is not an authorized champion. Otherwise assembles point-in-time
gold at the scoring moment, refits the promoted model, freezes the
vintage-safe seasonal-naive benchmark beside every forecast, and snapshots
the gold forecast table.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from retailpred.live_loop.ingest import code_revision
from retailpred.live_loop.scoring import score_store
from retailpred.platform.local_store import LocalTableStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=Path("data/v2/store"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outcome = score_store(
        LocalTableStore(args.store),
        now=datetime.now(timezone.utc),
        revision=code_revision(),
    )
    if outcome.forecasts is None:
        decision = outcome.decision
        print(
            f"Scoring refused (model={decision['model']}, "
            f"model_status={decision['model_status']}, "
            f"required_action={decision['required_action']})."
        )
        print("No forecasts were generated; gold forecast tables remain untouched.")
        return
    print(f"Forecast rows written: {len(outcome.forecasts):,}")
    print(f"Data cutoff: {outcome.data_cutoff}")
    statuses = outcome.forecasts["seasonal_naive_status"].value_counts().to_dict()
    print(f"Seasonal-naive benchmark statuses: {statuses}")


if __name__ == "__main__":
    main()
