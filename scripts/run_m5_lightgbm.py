"""Compare global LightGBM and Seasonal Naive on the frozen M5 holdout."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd

from forecasting.evaluation.m5_backtest import (
    hierarchy_rmsse_scales,
    official_m5_weights,
    recent_bottom_revenue,
    score_hierarchical_forecast,
    seasonal_naive_bottom_forecast,
)
from forecasting.models.m5_global_lightgbm import (
    FEATURE_COLUMNS,
    M5DirectFeatureBuilder,
    M5GlobalLightGBM,
    load_m5_price_vectors,
)
from forecasting.reconciliation.m5 import M5Hierarchy
from retailpred.platform.m5_pipeline import M5_ID_COLUMNS, file_sha256


TRAIN_END_INDEX = 1_912  # d_1913, zero based
VALIDATION_ORIGIN_INDEX = 1_884  # d_1885
HORIZONS = tuple(range(1, 29))
WEIGHT_WINDOW = 28


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_directory", type=Path, nargs="?", default=Path("data/v2/source/m5")
    )
    parser.add_argument("--training-windows", type=int, default=6)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("data/v2/evidence/m5_lightgbm"),
    )
    return parser.parse_args()


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    args = parse_args()
    if args.training_windows < 1:
        raise ValueError("--training-windows must be positive")
    started = time.perf_counter()
    source = args.source_directory.resolve()
    sales_path = source / "sales_train_evaluation.csv"
    calendar_path = source / "calendar.csv"
    prices_path = source / "sell_prices.csv"
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

    training_origins = tuple(
        VALIDATION_ORIGIN_INDEX - 28 * offset
        for offset in range(1, args.training_windows + 1)
    )
    feature_origins = (*training_origins, VALIDATION_ORIGIN_INDEX, TRAIN_END_INDEX)
    required_indices = set(feature_origins)
    for origin in feature_origins:
        required_indices.update(origin + horizon for horizon in HORIZONS)
    weeks = set(calendar.iloc[sorted(required_indices)]["wm_yr_wk"].astype(int))
    prices_by_week = load_m5_price_vectors(
        bottom=hierarchy.bottom, prices_path=prices_path, weeks=weeks
    )
    builder = M5DirectFeatureBuilder(
        bottom=hierarchy.bottom,
        calendar=calendar,
        prices_by_week=prices_by_week,
    )

    training_frames: list[pd.DataFrame] = []
    training_targets: list[np.ndarray] = []
    for origin in training_origins:
        features, target = builder.build(
            values, origin_index=origin, horizons=HORIZONS, include_target=True
        )
        training_frames.append(features)
        assert target is not None
        training_targets.append(target)
    x_train = pd.concat(training_frames, ignore_index=True)
    y_train = np.concatenate(training_targets)
    del training_frames, training_targets
    x_validation, y_validation = builder.build(
        values,
        origin_index=VALIDATION_ORIGIN_INDEX,
        horizons=HORIZONS,
        include_target=True,
    )
    assert y_validation is not None

    model = M5GlobalLightGBM(n_estimators=args.n_estimators)
    training_started = time.perf_counter()
    model.fit(
        x_train,
        y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        early_stopping_rounds=args.early_stopping_rounds,
    )
    training_seconds = time.perf_counter() - training_started
    train_rows = len(x_train)
    validation_rows = len(x_validation)
    del x_train, y_train, x_validation, y_validation
    gc.collect()

    evaluation_features, _ = builder.build(
        values,
        origin_index=TRAIN_END_INDEX,
        horizons=HORIZONS,
        include_target=False,
    )
    flat_prediction = model.predict(evaluation_features)
    bottom_prediction = flat_prediction.reshape(len(HORIZONS), len(hierarchy.bottom)).T
    bottom_actual = values[:, TRAIN_END_INDEX + 1 : TRAIN_END_INDEX + 29]
    history = values[:, : TRAIN_END_INDEX + 1]
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
    challenger_prediction = hierarchy.summing_matrix @ bottom_prediction
    challenger = score_hierarchical_forecast(
        hierarchy,
        actual=actual,
        prediction=challenger_prediction,
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

    args.output_directory.mkdir(parents=True, exist_ok=True)
    model_path = args.output_directory / "m5_global_lightgbm.pkl"
    comparison_path = args.output_directory / "m5_lightgbm_by_level.csv"
    importance_path = args.output_directory / "m5_lightgbm_feature_importance.csv"
    summary_path = args.output_directory / "m5_lightgbm_summary.json"
    model.save(model_path)
    comparison = challenger.by_level.merge(
        baseline.by_level[["level", "level_wrmsse"]].rename(
            columns={"level_wrmsse": "baseline_level_wrmsse"}
        ),
        on="level",
        validate="one_to_one",
    )
    comparison["relative_improvement"] = (
        comparison["baseline_level_wrmsse"] - comparison["level_wrmsse"]
    ) / comparison["baseline_level_wrmsse"]
    comparison.to_csv(comparison_path, index=False)
    booster = model.model
    assert booster is not None
    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "gain": booster.booster_.feature_importance(importance_type="gain"),
            "splits": booster.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("gain", ascending=False, ignore_index=True)
    importance.to_csv(importance_path, index=False)
    elapsed_seconds = time.perf_counter() - started
    best_iteration = int(booster.best_iteration_ or args.n_estimators)
    summary = {
        "baseline_wrmsse": baseline.overall_wrmsse,
        "best_iteration": best_iteration,
        "challenger_wrmsse": challenger.overall_wrmsse,
        "code_revision": _git_revision(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_days": ["d_1914", "d_1941"],
        "feature_count": len(FEATURE_COLUMNS),
        "lightgbm_version": lightgbm.__version__,
        "model": model.metadata(),
        "relative_improvement": (
            baseline.overall_wrmsse - challenger.overall_wrmsse
        )
        / baseline.overall_wrmsse,
        "runtime": {
            "elapsed_seconds": elapsed_seconds,
            "machine": platform.machine(),
            "python": platform.python_version(),
            "training_seconds": training_seconds,
        },
        "source_sha256": {
            path.name: file_sha256(path)
            for path in (calendar_path, sales_path, prices_path)
        },
        "train_rows": train_rows,
        "training_origin_days": [
            f"d_{origin + 1}" for origin in sorted(training_origins)
        ],
        "validation_origin_day": "d_1885",
        "validation_rows": validation_rows,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(comparison.to_string(index=False))
    print(importance.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
