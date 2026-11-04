#!/usr/bin/env python3
"""Download canonical native-monthly MRTS data for the v2 pipeline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from forecasting.data.mrts import MRTSClient
from forecasting.data.monthly_dataset import write_monthly_csv
from forecasting.registry import MRTS_SERIES, get_series


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument(
        "--series",
        action="append",
        choices=[item.series_id for item in MRTS_SERIES],
        help="Series to fetch; repeat the option. Defaults to all registered series.",
    )
    parser.add_argument("--output", type=Path, default=Path("data/v2/mrts_monthly.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = MRTSClient(os.getenv("CENSUS_API_KEY"))
    selected = args.series or [item.series_id for item in MRTS_SERIES]
    frames = [
        client.fetch_range(get_series(series_id), args.start_year, args.end_year)
        for series_id in selected
    ]
    result = pd.concat(frames, ignore_index=True).sort_values(
        ["series_id", "observation_time", "vintage_time"], ignore_index=True
    )
    write_monthly_csv(result, args.output)
    print(f"Wrote {len(result):,} independent monthly observations to {args.output}")


if __name__ == "__main__":
    main()
