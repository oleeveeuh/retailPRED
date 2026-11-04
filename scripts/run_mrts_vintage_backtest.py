#!/usr/bin/env python3
"""Point-in-time MRTS champion gate: global LightGBM vs Seasonal Naive on ALFRED vintages.

Trains every model only on data knowable at each forecast origin and scores
against latest revised values. This is the promotion gate for live MRTS
serving; its criteria are predeclared in docs/v2/M3_VINTAGE_BACKTEST.md and
must not be adjusted after seeing results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from forecasting.data.monthly_dataset import read_monthly_csv
from forecasting.evaluation.metrics import bias, mae, wape
from forecasting.evaluation.vintage import vintage_panel_backtest
from forecasting.models.global_lightgbm import GlobalLightGBM
from forecasting.models.panel_adapter import PerSeriesPanelAdapter
from forecasting.models.seasonal_naive import SeasonalNaive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--availability-lag-days", type=int, default=45)
    parser.add_argument("--min-train-size", type=int, default=60)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 6, 12])
    parser.add_argument("--output", type=Path, default=Path("data/v2/vintage_backtest_folds.csv"))
    parser.add_argument("--summary", type=Path, default=Path("data/v2/vintage_backtest_summary.json"))
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


def _series_metrics(folds: pd.DataFrame) -> dict[str, float]:
    return {
        "wape": float(wape(folds["actual"], folds["prediction"])),
        "mae": float(mae(folds["actual"], folds["prediction"])),
        "bias": float(bias(folds["actual"], folds["prediction"])),
    }


def _evaluate(model_name: str, folds: pd.DataFrame) -> dict[str, object]:
    per_series = {
        str(series_id): _series_metrics(group)
        for series_id, group in folds.groupby("series_id", sort=True)
    }
    return {
        "model_name": model_name,
        "fold_rows": len(folds),
        "aggregate": _series_metrics(folds),
        "mean_category_wape": float(
            pd.Series({key: value["wape"] for key, value in per_series.items()}).mean()
        ),
        "by_series": per_series,
    }


def main() -> None:
    args = parse_args()
    panel = read_monthly_csv(args.input)
    distinct_vintages = panel["vintage_time"].nunique()
    if distinct_vintages < 2:
        raise SystemExit(
            "the panel contains a single vintage; point-in-time evaluation needs "
            "at least two monthly ALFRED vintages"
        )

    results = {}
    for label, factory in (
        ("seasonal_naive", lambda: PerSeriesPanelAdapter(SeasonalNaive)),
        ("global_lightgbm", GlobalLightGBM),
    ):
        outcome = vintage_panel_backtest(
            panel,
            factory,
            horizons=tuple(args.horizons),
            min_train_size=args.min_train_size,
            availability_lag_days=args.availability_lag_days,
        )
        results[label] = (outcome, _evaluate(label, outcome.folds))

    naive_outcome, naive_metrics = results["seasonal_naive"]
    lgbm_outcome, lgbm_metrics = results["global_lightgbm"]
    naive_origins = set(naive_outcome.folds["origin_as_of"])
    lgbm_origins = set(lgbm_outcome.folds["origin_as_of"])
    if naive_origins != lgbm_origins:
        raise SystemExit("models were not evaluated on identical origins")

    naive_by_series = naive_metrics["by_series"]
    lgbm_by_series = lgbm_metrics["by_series"]
    improvements = {
        series: naive_by_series[series]["wape"] - lgbm_by_series[series]["wape"]
        for series in sorted(naive_by_series)
    }
    regressions = {series: delta for series, delta in improvements.items() if delta <= 0}
    gate = {
        "mean_category_wape_improved": (
            lgbm_metrics["mean_category_wape"] < naive_metrics["mean_category_wape"]
        ),
        "improved_series_count": sum(1 for delta in improvements.values() if delta > 0),
        "series_count": len(improvements),
        "regressions": {
            series: {"wape_delta": delta, "naive_wape": naive_by_series[series]["wape"]}
            for series, delta in sorted(regressions.items())
        },
        "max_allowed_regressions": 1,
        "max_regression_pp_tolerance": 2.0,
    }
    gate["acceptable_regressions"] = all(
        naive_by_series[series]["wape"] - lgbm_by_series[series]["wape"] > -0.02
        for series in regressions
    )
    gate["passed"] = (
        gate["mean_category_wape_improved"]
        and gate["improved_series_count"] >= gate["series_count"] - 1
        and gate["acceptable_regressions"]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(
        [naive_outcome.folds, lgbm_outcome.folds], ignore_index=True
    ).sort_values(["model_name", "series_id", "forecast_origin", "horizon"], ignore_index=True)
    combined.to_csv(args.output, index=False)

    summary = {
        "status": "measured" if gate["passed"] else "gate_failed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_sha256": _sha256(args.input),
        "code_revision": _code_revision(),
        "protocol": {
            "availability_lag_days": args.availability_lag_days,
            "min_train_size": args.min_train_size,
            "horizons": args.horizons,
            "actual_source": "latest_revision_vintage",
        },
        "fold_rows": {"seasonal_naive": len(naive_outcome.folds), "global_lightgbm": len(lgbm_outcome.folds)},
        "excluded_origins": lgbm_outcome.excluded_origins,
        "seasonal_naive": naive_metrics,
        "global_lightgbm": lgbm_metrics,
        "wape_improvement_by_series": improvements,
        "promotion_gate": gate,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"Folds evaluated: {len(naive_outcome.folds):,} per model")
    print(f"Excluded origins: {len(lgbm_outcome.excluded_origins)}")
    print(f"Mean category WAPE - seasonal_naive: {naive_metrics['mean_category_wape']:.4f}")
    print(f"Mean category WAPE - global_lightgbm: {lgbm_metrics['mean_category_wape']:.4f}")
    verdict = "PASSED" if gate["passed"] else "FAILED"
    print(f"Point-in-time promotion gate: {verdict} (summary written to {args.summary})")


if __name__ == "__main__":
    main()
