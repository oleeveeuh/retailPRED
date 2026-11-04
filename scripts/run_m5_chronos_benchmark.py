"""Score zero-shot Chronos-2 on the complete official M5 holdout."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting.evaluation.m5_backtest import (
    hierarchy_rmsse_scales,
    official_m5_weights,
    recent_bottom_revenue,
    score_hierarchical_forecast,
    seasonal_naive_bottom_forecast,
)
from forecasting.reconciliation.m5 import M5Hierarchy
from retailpred.platform.m5_pipeline import M5_ID_COLUMNS, file_sha256


TRAIN_END_INDEX = 1_912
HORIZON = 28
WEIGHT_WINDOW = 28


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_directory", type=Path, nargs="?", default=Path("data/v2/source/m5")
    )
    parser.add_argument("--foundation-python", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-id", default="amazon/chronos-2")
    parser.add_argument(
        "--incumbent-summary",
        type=Path,
        default=Path("data/v2/evidence/m5_lightgbm/m5_lightgbm_summary.json"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("data/v2/evidence/m5_chronos_benchmark"),
    )
    return parser.parse_args()


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _peak_memory_mb() -> float:
    usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return usage / (1024 * 1024) if platform.system() == "Darwin" else usage / 1024


def _absolute_without_resolving_symlinks(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    source = args.source_directory.resolve()
    sales_path = source / "sales_train_evaluation.csv"
    calendar_path = source / "calendar.csv"
    prices_path = source / "sell_prices.csv"
    source_hashes = {
        path.name: file_sha256(path)
        for path in (calendar_path, sales_path, prices_path)
    }
    incumbent = json.loads(args.incumbent_summary.read_text())
    if incumbent.get("source_sha256") != source_hashes:
        raise ValueError("incumbent and Chronos benchmark source hashes differ")
    incumbent_score = float(incumbent["challenger_wrmsse"])

    calendar = pd.read_csv(calendar_path)
    observed_days = [f"d_{day}" for day in range(1, TRAIN_END_INDEX + 30)]
    dtypes: dict[str, object] = {day: "float32" for day in observed_days}
    dtypes.update({column: "string" for column in M5_ID_COLUMNS})
    sales = pd.read_csv(
        sales_path,
        usecols=list(M5_ID_COLUMNS) + observed_days,
        dtype=dtypes,
    )
    bottom = sales.loc[:, M5_ID_COLUMNS].rename(columns={"id": "series_id"})
    hierarchy = M5Hierarchy.from_bottom(bottom)
    hierarchy.validate_official_shape()
    ordered = sales.set_index("id").reindex(hierarchy.bottom["series_id"])
    values = ordered.loc[:, observed_days].to_numpy(dtype=np.float32, copy=False)
    del sales, ordered
    history = values[:, : TRAIN_END_INDEX + 1]
    bottom_actual = values[:, TRAIN_END_INDEX + 1 : TRAIN_END_INDEX + 29]

    args.output_directory.mkdir(parents=True, exist_ok=True)
    context_path = args.output_directory / "chronos_context.npy"
    raw_prediction_path = args.output_directory / "chronos_raw_predictions.npy"
    runtime_path = args.output_directory / "chronos_runtime.json"
    np.save(context_path, history)
    child_environment = os.environ.copy()
    child_environment["HF_HOME"] = str(
        Path("data/v2/cache/huggingface").resolve()
    )
    subprocess.run(
        [
            str(_absolute_without_resolving_symlinks(args.foundation_python)),
            "-m",
            "scripts.predict_chronos_matrix",
            str(context_path),
            str(raw_prediction_path),
            str(runtime_path),
            "--model-id",
            args.model_id,
            "--batch-size",
            str(args.batch_size),
        ],
        check=True,
        env=child_environment,
    )
    context_path.unlink()
    runtime = json.loads(runtime_path.read_text())
    bottom_prediction = np.load(raw_prediction_path).astype(np.float64)
    if bottom_prediction.shape != bottom_actual.shape or not np.isfinite(
        bottom_prediction
    ).all():
        raise RuntimeError("Chronos-2 returned an invalid forecast matrix")
    bottom_prediction = np.clip(bottom_prediction, 0.0, None)

    scales = hierarchy_rmsse_scales(hierarchy, history)
    weight_days = [f"d_{day}" for day in range(1_886, 1_914)]
    revenue = recent_bottom_revenue(
        bottom=hierarchy.bottom,
        recent_units=history[:, -WEIGHT_WINDOW:],
        recent_days=weight_days,
        calendar_path=calendar_path,
        prices_path=prices_path,
    )
    weights = official_m5_weights(hierarchy, revenue)
    actual = hierarchy.summing_matrix @ bottom_actual
    chronos = score_hierarchical_forecast(
        hierarchy,
        actual=actual,
        prediction=hierarchy.summing_matrix @ bottom_prediction,
        scales=scales,
        weights=weights,
    )
    baseline_bottom = seasonal_naive_bottom_forecast(history)
    baseline = score_hierarchical_forecast(
        hierarchy,
        actual=actual,
        prediction=hierarchy.summing_matrix @ baseline_bottom,
        scales=scales,
        weights=weights,
    )

    comparison = chronos.by_level.merge(
        baseline.by_level[["level", "level_wrmsse"]].rename(
            columns={"level_wrmsse": "baseline_level_wrmsse"}
        ),
        on="level",
        validate="one_to_one",
    )
    comparison["relative_improvement"] = (
        comparison["baseline_level_wrmsse"] - comparison["level_wrmsse"]
    ) / comparison["baseline_level_wrmsse"]
    comparison.to_csv(
        args.output_directory / "m5_chronos_by_level.csv", index=False
    )
    prediction_frame = pd.DataFrame(
        bottom_prediction,
        columns=[f"d_{day}" for day in range(1_914, 1_942)],
    )
    prediction_frame.insert(0, "series_id", hierarchy.bottom["series_id"].to_numpy())
    prediction_frame.to_parquet(
        args.output_directory / "m5_chronos_bottom_predictions.parquet", index=False
    )

    elapsed_seconds = time.perf_counter() - started
    improvement_vs_baseline = (
        baseline.overall_wrmsse - chronos.overall_wrmsse
    ) / baseline.overall_wrmsse
    improvement_vs_incumbent = (
        incumbent_score - chronos.overall_wrmsse
    ) / incumbent_score
    summary = {
        "baseline_wrmsse": baseline.overall_wrmsse,
        "challenger_wrmsse": chronos.overall_wrmsse,
        "code_revision": _git_revision(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": "cpu",
        "evaluation_days": ["d_1914", "d_1941"],
        "improvement_vs_incumbent": improvement_vs_incumbent,
        "incumbent_code_revision": incumbent["code_revision"],
        "incumbent_model": "global_lightgbm",
        "incumbent_wrmsse": incumbent_score,
        "model_id": args.model_id,
        "point_forecast": "Chronos-2 predictions output; no quantile selected",
        "relative_improvement_vs_baseline": improvement_vs_baseline,
        "runtime": {
            **runtime,
            "elapsed_seconds": elapsed_seconds,
            "machine": platform.machine(),
            "parent_peak_process_memory_mb": _peak_memory_mb(),
            "python": platform.python_version(),
        },
        "source_sha256": source_hashes,
        "zero_shot": True,
    }
    (args.output_directory / "m5_chronos_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
