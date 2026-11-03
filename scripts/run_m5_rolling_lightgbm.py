"""Audit global M5 LightGBM over nested 28-day rolling origins."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting.evaluation.m5_backtest import (
    hierarchy_rmsse_scales,
    m5_rolling_folds,
    official_m5_weights,
    recent_bottom_revenue,
    score_hierarchical_forecast,
    seasonal_naive_bottom_forecast,
)
from forecasting.models.m5_global_lightgbm import (
    M5DirectFeatureBuilder,
    M5GlobalLightGBM,
    load_m5_price_vectors,
)
from forecasting.reconciliation.m5 import M5Hierarchy
from retailpred.platform.m5_pipeline import M5_ID_COLUMNS, file_sha256


HORIZONS = tuple(range(1, 29))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_directory", type=Path, nargs="?", default=Path("data/v2/source/m5")
    )
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--training-windows", type=int, default=6)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("data/v2/evidence/m5_rolling_lightgbm"),
    )
    return parser.parse_args()


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _day(index: int) -> str:
    return f"d_{index + 1}"


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    folds = m5_rolling_folds(
        fold_count=args.fold_count, training_windows=args.training_windows
    )
    source = args.source_directory.resolve()
    sales_path = source / "sales_train_evaluation.csv"
    calendar_path = source / "calendar.csv"
    prices_path = source / "sell_prices.csv"
    calendar = pd.read_csv(calendar_path)
    final_target_index = folds[-1].evaluation_origin_index + max(HORIZONS)
    observed_days = [f"d_{day}" for day in range(1, final_target_index + 2)]
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

    all_origins = {
        origin
        for fold in folds
        for origin in (
            *fold.training_origin_indices,
            fold.validation_origin_index,
            fold.evaluation_origin_index,
        )
    }
    required_indices = set(all_origins)
    for origin in all_origins:
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

    fold_records: list[dict[str, object]] = []
    level_frames: list[pd.DataFrame] = []
    for fold_number, fold in enumerate(folds, start=1):
        fold_started = time.perf_counter()
        train_features: list[pd.DataFrame] = []
        train_targets: list[np.ndarray] = []
        for origin in fold.training_origin_indices:
            features, target = builder.build(
                values, origin_index=origin, horizons=HORIZONS, include_target=True
            )
            train_features.append(features)
            assert target is not None
            train_targets.append(target)
        x_train = pd.concat(train_features, ignore_index=True)
        y_train = np.concatenate(train_targets)
        x_validation, y_validation = builder.build(
            values,
            origin_index=fold.validation_origin_index,
            horizons=HORIZONS,
            include_target=True,
        )
        assert y_validation is not None
        model = M5GlobalLightGBM(n_estimators=args.n_estimators)
        model.fit(
            x_train,
            y_train,
            x_validation=x_validation,
            y_validation=y_validation,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        del x_train, y_train, x_validation, y_validation, train_features, train_targets
        gc.collect()

        evaluation_features, _ = builder.build(
            values,
            origin_index=fold.evaluation_origin_index,
            horizons=HORIZONS,
            include_target=False,
        )
        flat_prediction = model.predict(evaluation_features)
        bottom_prediction = flat_prediction.reshape(len(HORIZONS), len(hierarchy.bottom)).T
        del evaluation_features, flat_prediction
        history = values[:, : fold.evaluation_origin_index + 1]
        bottom_actual = values[
            :,
            fold.evaluation_origin_index + 1 : fold.evaluation_origin_index + 29,
        ]
        scales = hierarchy_rmsse_scales(
            hierarchy, history, require_positive=False
        )
        weight_start = fold.evaluation_origin_index - 27
        weight_days = [_day(index) for index in range(weight_start, fold.evaluation_origin_index + 1)]
        revenue = recent_bottom_revenue(
            bottom=hierarchy.bottom,
            recent_units=history[:, -28:],
            recent_days=weight_days,
            calendar_path=calendar_path,
            prices_path=prices_path,
        )
        weights = official_m5_weights(hierarchy, revenue)
        actual = hierarchy.summing_matrix @ bottom_actual
        challenger = score_hierarchical_forecast(
            hierarchy,
            actual=actual,
            prediction=hierarchy.summing_matrix @ bottom_prediction,
            scales=scales,
            weights=weights,
        )
        baseline = score_hierarchical_forecast(
            hierarchy,
            actual=actual,
            prediction=hierarchy.summing_matrix
            @ seasonal_naive_bottom_forecast(history),
            scales=scales,
            weights=weights,
        )
        relative_improvement = (
            baseline.overall_wrmsse - challenger.overall_wrmsse
        ) / baseline.overall_wrmsse
        booster = model.model
        assert booster is not None
        fold_records.append(
            {
                "fold": fold_number,
                "evaluation_origin_day": _day(fold.evaluation_origin_index),
                "evaluation_start_day": _day(fold.evaluation_origin_index + 1),
                "evaluation_end_day": _day(fold.evaluation_origin_index + 28),
                "baseline_wrmsse": baseline.overall_wrmsse,
                "challenger_wrmsse": challenger.overall_wrmsse,
                "relative_improvement": relative_improvement,
                "best_iteration": int(booster.best_iteration_ or args.n_estimators),
                "elapsed_seconds": time.perf_counter() - fold_started,
            }
        )
        levels = challenger.by_level.merge(
            baseline.by_level[["level", "level_wrmsse"]].rename(
                columns={"level_wrmsse": "baseline_level_wrmsse"}
            ),
            on="level",
            validate="one_to_one",
        )
        levels.insert(0, "fold", fold_number)
        levels["relative_improvement"] = (
            levels["baseline_level_wrmsse"] - levels["level_wrmsse"]
        ) / levels["baseline_level_wrmsse"]
        level_frames.append(levels)
        del model, booster, bottom_prediction, history, scales, revenue, weights
        gc.collect()
        print(
            f"Fold {fold_number}/{len(folds)} {_day(fold.evaluation_origin_index)}: "
            f"baseline={baseline.overall_wrmsse:.6f}, "
            f"lightgbm={challenger.overall_wrmsse:.6f}, "
            f"improvement={relative_improvement:.2%}",
            flush=True,
        )

    fold_results = pd.DataFrame.from_records(fold_records)
    level_results = pd.concat(level_frames, ignore_index=True)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    fold_results.to_csv(args.output_directory / "m5_rolling_folds.csv", index=False)
    level_results.to_csv(args.output_directory / "m5_rolling_levels.csv", index=False)
    level_stability = (
        level_results.groupby(["level", "level_name"], as_index=False)
        .agg(
            folds=("fold", "size"),
            improved_folds=("relative_improvement", lambda values: int((values > 0).sum())),
            mean_relative_improvement=("relative_improvement", "mean"),
            worst_relative_improvement=("relative_improvement", "min"),
        )
        .sort_values("level", ignore_index=True)
    )
    level_stability.to_csv(
        args.output_directory / "m5_rolling_level_stability.csv", index=False
    )
    summary = {
        "code_revision": _git_revision(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fold_count": len(folds),
        "improved_folds": int((fold_results["relative_improvement"] > 0).sum()),
        "mean_baseline_wrmsse": float(fold_results["baseline_wrmsse"].mean()),
        "mean_challenger_wrmsse": float(fold_results["challenger_wrmsse"].mean()),
        "mean_relative_improvement": float(fold_results["relative_improvement"].mean()),
        "median_relative_improvement": float(fold_results["relative_improvement"].median()),
        "worst_relative_improvement": float(fold_results["relative_improvement"].min()),
        "elapsed_seconds": time.perf_counter() - started,
        "source_sha256": {
            path.name: file_sha256(path)
            for path in (calendar_path, sales_path, prices_path)
        },
        "training_windows_per_fold": args.training_windows,
    }
    (args.output_directory / "m5_rolling_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(fold_results.to_string(index=False))
    print(level_stability.to_string(index=False))


if __name__ == "__main__":
    main()
