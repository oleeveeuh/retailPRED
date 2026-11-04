#!/usr/bin/env python3
"""Run the v2 Seasonal Naive baseline on an official MRTS revision workbook."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from forecasting.data.mrts_workbook import load_revision_workbook
from forecasting.evaluation.backtest import rolling_origin_backtest
from forecasting.evaluation.metrics import bias, mae, rmsse, wape
from forecasting.models.seasonal_naive import SeasonalNaive
from forecasting.registry import MRTS_SERIES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--min-train-size", type=int, default=60)
    parser.add_argument(
        "--vintage-time",
        required=True,
        help="Timezone-aware ISO timestamp identifying when this workbook vintage was retrieved.",
    )
    parser.add_argument("--output", type=Path, default=Path("data/v2/mrts_seasonal_naive_folds.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vintage = datetime.fromisoformat(args.vintage_time)
    if vintage.tzinfo is None or vintage.utcoffset() is None:
        raise ValueError("--vintage-time must include a UTC offset, such as +00:00")
    fold_frames: list[pd.DataFrame] = []
    summary: list[dict[str, object]] = []
    for series in MRTS_SERIES:
        observations = load_revision_workbook(
            args.workbook,
            series,
            start_year=args.start_year,
            end_year=args.end_year,
            vintage_time=vintage,
        )
        folds = rolling_origin_backtest(
            observations[["series_id", "observation_time", "value"]],
            SeasonalNaive,
            horizons=(1, 3, 6, 12),
            min_train_size=args.min_train_size,
            data_vintage=vintage,
            actual_source="census_mrts_annual_revision",
        )
        fold_frames.append(folds)
        summary.append(
            {
                "series_id": series.series_id,
                "fold_rows": len(folds),
                "mae": mae(folds["actual"], folds["prediction"]),
                "wape": wape(folds["actual"], folds["prediction"]),
                "bias": bias(folds["actual"], folds["prediction"]),
                "rmsse": rmsse(
                    folds["actual"],
                    folds["prediction"],
                    observations["value"].iloc[: args.min_train_size],
                    season_length=12,
                ),
            }
        )
    all_folds = pd.concat(fold_frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_folds.to_csv(args.output, index=False)
    print(pd.DataFrame(summary).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nWrote {len(all_folds):,} fold rows to {args.output}")


if __name__ == "__main__":
    main()
