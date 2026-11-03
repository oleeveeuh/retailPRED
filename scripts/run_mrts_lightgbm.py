#!/usr/bin/env python3
"""Compare global LightGBM and Seasonal Naive on identical MRTS folds."""

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
import pandas as pd

from forecasting.data.mrts_workbook import load_revision_workbook
from forecasting.evaluation.backtest import (
    rolling_origin_backtest,
    rolling_origin_panel_backtest,
)
from forecasting.evaluation.metrics import bias, mae, wape
from forecasting.models.global_lightgbm import GlobalLightGBM
from forecasting.models.seasonal_naive import SeasonalNaive
from forecasting.registry import MRTS_SERIES
from forecasting.promotion import PromotionPolicy
from forecasting.tracking.mlflow import log_artifact_to_run, log_backtest_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--min-train-size", type=int, default=60)
    parser.add_argument("--vintage-time", required=True)
    parser.add_argument("--n-estimators", type=int, default=250)
    parser.add_argument("--output", type=Path, default=Path("data/v2/mrts_model_folds.csv"))
    parser.add_argument("--summary", type=Path, default=Path("data/v2/mrts_model_summary.json"))
    parser.add_argument("--tracking-uri")
    parser.add_argument("--experiment-name", default="retailpred-mrts")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _metrics(folds: pd.DataFrame) -> dict[str, float | int]:
    return {
        "fold_rows": len(folds),
        "wape": wape(folds["actual"], folds["prediction"]),
        "mae": mae(folds["actual"], folds["prediction"]),
        "bias": bias(folds["actual"], folds["prediction"]),
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
    common = {
        "horizons": (1, 3, 6, 12),
        "min_train_size": args.min_train_size,
        "data_vintage": vintage,
        "actual_source": "census_mrts_annual_revision",
    }
    started = time.perf_counter()
    challenger = rolling_origin_panel_backtest(
        panel,
        lambda: GlobalLightGBM(n_estimators=args.n_estimators),
        **common,
    )
    challenger_runtime = time.perf_counter() - started
    baseline = pd.concat(
        [
            rolling_origin_backtest(group, SeasonalNaive, **common)
            for _, group in panel.groupby("series_id", sort=True)
        ],
        ignore_index=True,
    )
    key = ["series_id", "forecast_origin", "target_time", "horizon"]
    if set(map(tuple, challenger[key].to_numpy())) != set(
        map(tuple, baseline[key].to_numpy())
    ):
        raise RuntimeError("challenger and baseline did not use identical folds")

    all_folds = pd.concat([baseline, challenger], ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_folds.to_csv(args.output, index=False)
    results = {
        model_name: _metrics(folds)
        for model_name, folds in all_folds.groupby("model_name", sort=True)
    }
    decision = PromotionPolicy().evaluate(
        baseline_wape=float(results["seasonal_naive"]["wape"]),
        challenger_wape=float(results["global_lightgbm"]["wape"]),
        runtime_seconds=challenger_runtime,
    )
    runtime = {
        "elapsed_seconds": challenger_runtime,
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "lightgbm": lightgbm.__version__,
    }
    summary = {
        "data_version": _sha256(args.workbook),
        "code_revision": _code_revision(),
        "results": results,
        "promotion": {
            "promoted": decision.promoted,
            "relative_wape_improvement": decision.relative_wape_improvement,
            "reasons": list(decision.reasons),
        },
        "runtime": runtime,
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.tracking_uri:
        model_metadata = GlobalLightGBM(n_estimators=args.n_estimators).metadata()
        summary["mlflow"] = log_backtest_run(
            challenger,
            experiment_name=args.experiment_name,
            run_name="global-lightgbm",
            tracking_uri=args.tracking_uri,
            data_version=summary["data_version"],
            code_revision=summary["code_revision"],
            model_metadata=model_metadata,
            runtime_metadata=runtime,
            artifacts=(args.output,),
        )
        args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        log_artifact_to_run(
            run_id=summary["mlflow"]["run_id"],
            tracking_uri=args.tracking_uri,
            artifact=args.summary,
            artifact_path="evidence",
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
