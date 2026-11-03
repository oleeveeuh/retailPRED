"""Analyze frozen MRTS model folds across declared economic regimes."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from forecasting.evaluation.metrics import bias, mae, wape
from forecasting.evaluation.regimes import (
    MRTS_REGIME_BOUNDARIES,
    assign_mrts_regime,
)


KEY_COLUMNS = ("series_id", "forecast_origin", "target_time", "horizon")
EXPECTED_MODELS = ("global_lightgbm", "seasonal_naive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "folds", type=Path, nargs="?", default=Path("data/v2/mrts_model_folds.csv")
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("data/v2/evidence/mrts_regimes"),
    )
    return parser.parse_args()


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    return {
        "fold_rows": len(frame),
        "wape": wape(frame["actual"], frame["prediction"]),
        "mae": mae(frame["actual"], frame["prediction"]),
        "bias": bias(frame["actual"], frame["prediction"]),
    }


def main() -> None:
    args = parse_args()
    folds = pd.read_csv(
        args.folds,
        parse_dates=["forecast_origin", "target_time", "training_cutoff"],
    )
    required = {*KEY_COLUMNS, "model_name", "prediction", "actual"}
    if missing := required.difference(folds.columns):
        raise ValueError(f"fold evidence is missing columns: {sorted(missing)}")
    if set(folds["model_name"].unique()) != set(EXPECTED_MODELS):
        raise ValueError("regime analysis requires exactly the frozen model pair")
    if folds.duplicated([*KEY_COLUMNS, "model_name"]).any():
        raise ValueError("fold evidence contains duplicate model keys")
    keyed = {
        model: set(map(tuple, frame.loc[:, KEY_COLUMNS].to_numpy()))
        for model, frame in folds.groupby("model_name", sort=True)
    }
    if keyed[EXPECTED_MODELS[0]] != keyed[EXPECTED_MODELS[1]]:
        raise ValueError("models must have identical regime-evaluation rows")
    actual_counts = folds.groupby([*KEY_COLUMNS])["actual"].nunique()
    if (actual_counts != 1).any():
        raise ValueError("model rows disagree on realized actuals")

    folds["regime"] = assign_mrts_regime(folds["target_time"]).to_numpy()
    regime_rows: list[dict[str, object]] = []
    category_rows: list[dict[str, object]] = []
    horizon_rows: list[dict[str, object]] = []
    for (regime, model), group in folds.groupby(
        ["regime", "model_name"], sort=True
    ):
        regime_rows.append({"regime": regime, "model_name": model, **_metrics(group)})
    for (regime, model, series_id), group in folds.groupby(
        ["regime", "model_name", "series_id"], sort=True
    ):
        category_rows.append(
            {
                "regime": regime,
                "model_name": model,
                "series_id": series_id,
                **_metrics(group),
            }
        )
    for (regime, model, horizon), group in folds.groupby(
        ["regime", "model_name", "horizon"], sort=True
    ):
        horizon_rows.append(
            {
                "regime": regime,
                "model_name": model,
                "horizon": int(horizon),
                **_metrics(group),
            }
        )

    regime_metrics = pd.DataFrame(regime_rows)
    category_metrics = pd.DataFrame(category_rows)
    horizon_metrics = pd.DataFrame(horizon_rows)
    mean_category = (
        category_metrics.groupby(["regime", "model_name"], as_index=False)
        .agg(mean_category_wape=("wape", "mean"))
    )
    comparison = regime_metrics.merge(
        mean_category, on=["regime", "model_name"], validate="one_to_one"
    ).pivot(
        index="regime",
        columns="model_name",
        values=["fold_rows", "wape", "mae", "bias", "mean_category_wape"],
    )
    comparison.columns = [f"{metric}_{model}" for metric, model in comparison.columns]
    comparison = comparison.reset_index()
    comparison["relative_wape_improvement"] = (
        comparison["wape_seasonal_naive"] - comparison["wape_global_lightgbm"]
    ) / comparison["wape_seasonal_naive"]
    comparison["relative_mean_category_wape_improvement"] = (
        comparison["mean_category_wape_seasonal_naive"]
        - comparison["mean_category_wape_global_lightgbm"]
    ) / comparison["mean_category_wape_seasonal_naive"]
    category_wide = category_metrics.pivot(
        index=["regime", "series_id"], columns="model_name", values="wape"
    ).reset_index()
    category_wide["lightgbm_improved"] = (
        category_wide["global_lightgbm"] < category_wide["seasonal_naive"]
    )
    category_stability = (
        category_wide.groupby("regime", as_index=False)
        .agg(
            categories=("series_id", "size"),
            improved_categories=("lightgbm_improved", "sum"),
        )
    )
    comparison = comparison.merge(
        category_stability, on="regime", validate="one_to_one"
    )
    comparison["regime_gate_passed"] = (
        (comparison["relative_wape_improvement"] > 0)
        & (comparison["relative_mean_category_wape_improvement"] > 0)
        & (comparison["improved_categories"] == comparison["categories"])
    )

    args.output_directory.mkdir(parents=True, exist_ok=True)
    folds.to_csv(args.output_directory / "mrts_regime_folds.csv", index=False)
    comparison.to_csv(args.output_directory / "mrts_regime_summary.csv", index=False)
    category_metrics.to_csv(
        args.output_directory / "mrts_regime_by_category.csv", index=False
    )
    horizon_metrics.to_csv(
        args.output_directory / "mrts_regime_by_horizon.csv", index=False
    )
    summary = {
        "boundaries": {
            key: value.date().isoformat()
            for key, value in MRTS_REGIME_BOUNDARIES.items()
        },
        "code_revision": _git_revision(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fold_evidence_sha256": _sha256(args.folds),
        "gate": {
            "all_regimes_passed": bool(comparison["regime_gate_passed"].all()),
            "requirements": [
                "positive aggregate WAPE improvement in every regime",
                "positive mean-category WAPE improvement in every regime",
                "LightGBM improves every category within every regime",
            ],
        },
        "regimes": {
            str(row["regime"]): {
                "fold_rows_per_model": int(row["fold_rows_global_lightgbm"]),
                "global_lightgbm_wape": float(row["wape_global_lightgbm"]),
                "seasonal_naive_wape": float(row["wape_seasonal_naive"]),
                "relative_wape_improvement": float(row["relative_wape_improvement"]),
                "global_lightgbm_mean_category_wape": float(
                    row["mean_category_wape_global_lightgbm"]
                ),
                "seasonal_naive_mean_category_wape": float(
                    row["mean_category_wape_seasonal_naive"]
                ),
                "relative_mean_category_wape_improvement": float(
                    row["relative_mean_category_wape_improvement"]
                ),
                "improved_categories": int(row["improved_categories"]),
                "categories": int(row["categories"]),
                "gate_passed": bool(row["regime_gate_passed"]),
            }
            for _, row in comparison.iterrows()
        },
    }
    (args.output_directory / "mrts_regime_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
