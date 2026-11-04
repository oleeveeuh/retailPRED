#!/usr/bin/env python3
"""Append realized-error rows for published forecasts against revised actuals."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from retailpred.live_loop.realized import run_realized_monitor
from retailpred.platform.local_store import LocalTableStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=Path("data/v2/store"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outcome = run_realized_monitor(
        LocalTableStore(args.store), now=datetime.now(timezone.utc)
    )
    print(f"Realized-error rows computed this run: {outcome.rows_computed:,}")
    print(f"Realized-error rows in gold table: {outcome.rows_in_table:,}")
    if outcome.rows_computed == 0:
        print(
            "No forecast targets have revised actuals yet; the table is unchanged."
        )


if __name__ == "__main__":
    main()
