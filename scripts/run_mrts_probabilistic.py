#!/usr/bin/env python3
"""Evaluate native and conformalized LightGBM P10/P50/P90 forecasts."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd

from forecasting.data.mrts_workbook import load_revision_workbook
from forecasting.evaluation.backtest import rolling_origin_panel_quantile_backtest
from forecasting.evaluation.calibration import (
    ConformalIntervalCalibrator,
    temporal_calibration_split,
)
from forecasting.evaluation.metrics import (
    interval_coverage,
    interval_width,
    pinball_loss,
    weighted_interval_score,
)
from forecasting.models.global_lightgbm import GlobalLightGBM
from forecasting.registry import MRTS_SERIES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--vintage-time", required=True)
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--min-train-size", type=int, default=60)
    parser.add_argument("--origin-start", default="2017-01-01")
    parser.add_argument("--calibration-end", default="2021-12-31")
    parser.add_argument("--evaluation-start", default="2022-01-01")
    parser.add_argument("--n-estimators", type=int, default=250)
    parser.add_argument(
        "--output", type=Path, default=Path("data/v2/mrts_probabilistic_folds.csv")
    )
    parser.add_argument(
        "--summary", type=Path, default=Path("data/v2/mrts_probabilistic_summary.json")
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _metrics(
    frame: pd.DataFrame, *, lower: str, median: str, upper: str
) -> dict[str, float | int]:
    mean_scale = float(np.mean(np.abs(frame["actual"])))
    return {
        "rows": len(frame),
        "coverage_80": interval_coverage(frame["actual"], frame[lower], frame[upper]),
        "mean_interval_width": interval_width(frame[lower], frame[upper]),
        "mean_relative_width": float(
            np.mean((frame[upper] - frame[lower]) / np.abs(frame["actual"]))
        ),
        "normalized_wis": weighted_interval_score(
            frame["actual"], frame[median], [(0.2, frame[lower], frame[upper])]
        )
        / mean_scale,
        "normalized_pinball_p10": pinball_loss(frame["actual"], frame[lower], 0.1)
        / mean_scale,
        "normalized_pinball_p50": pinball_loss(frame["actual"], frame[median], 0.5)
        / mean_scale,
        "normalized_pinball_p90": pinball_loss(frame["actual"], frame[upper], 0.9)
        / mean_scale,
    }


def main() -> None:
    args = parse_args()
    vintage = datetime.fromisoformat(args.vintage_time)
    if vintage.tzinfo is None or vintage.utcoffset() is None:
        raise ValueError("--vintage-time must include a UTC offset")
    panel = pd.concat(
        [
            load_revision_workbook(
                args.workbook,
                series,
                start_year=args.start_year,
                end_year=args.end_year,
                vintage_time=vintage,
            )[["series_id", "observation_time", "value"]]
            for series in MRTS_SERIES
        ],
        ignore_index=True,
    )
    quantiles = (0.1, 0.5, 0.9)
    started = time.perf_counter()
    folds = rolling_origin_panel_quantile_backtest(
        panel,
        lambda: GlobalLightGBM(
            n_estimators=args.n_estimators, quantiles=quantiles
        ),
        quantiles=quantiles,
        horizons=(1, 3, 6, 12),
        min_train_size=args.min_train_size,
        data_vintage=vintage,
        actual_source="census_mrts_annual_revision",
        origin_start=args.origin_start,
    )
    runtime_seconds = time.perf_counter() - started
    calibration, evaluation = temporal_calibration_split(
        folds,
        calibration_end=args.calibration_end,
        evaluation_start=args.evaluation_start,
    )
    calibrator = ConformalIntervalCalibrator(
        alpha=0.2,
        group_columns=("series_id", "horizon"),
        method="multiplicative",
    ).fit(calibration, lower_column="q_10", upper_column="q_90")
    evaluated = calibrator.transform(
        evaluation,
        lower_column="q_10",
        upper_column="q_90",
        output_lower="q_10_calibrated",
        output_upper="q_90_calibrated",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    evaluated.to_csv(args.output, index=False)
    by_horizon = {
        str(horizon): {
            "native": _metrics(group, lower="q_10", median="q_50", upper="q_90"),
            "conformal": _metrics(
                group,
                lower="q_10_calibrated",
                median="q_50",
                upper="q_90_calibrated",
            ),
        }
        for horizon, group in evaluated.groupby("horizon", sort=True)
    }
    summary = {
        "data_version": _sha256(args.workbook),
        "code_revision": _code_revision(),
        "calibration": {
            "target_end": args.calibration_end,
            "rows": len(calibration),
            "method": "multiplicative_cqr_by_series_and_horizon",
            "adjustments": {
                f"{key[0]}|h{key[1]}": value
                for key, value in calibrator.adjustments.items()
            },
        },
        "evaluation": {
            "target_start": args.evaluation_start,
            "native": _metrics(evaluated, lower="q_10", median="q_50", upper="q_90"),
            "conformal": _metrics(
                evaluated,
                lower="q_10_calibrated",
                median="q_50",
                upper="q_90_calibrated",
            ),
            "by_horizon": by_horizon,
        },
        "runtime": {
            "elapsed_seconds": runtime_seconds,
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "lightgbm": lightgbm.__version__,
        },
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
