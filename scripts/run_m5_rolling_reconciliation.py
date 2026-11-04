"""Audit hierarchy-aware M5 reconciliation across rolling origins."""

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
)
from forecasting.models.m5_aggregate_lightgbm import (
    M5AggregateFeatureBuilder,
    M5AggregateLightGBM,
)
from forecasting.models.m5_global_lightgbm import (
    M5DirectFeatureBuilder,
    M5GlobalLightGBM,
    load_m5_price_vectors,
)
from forecasting.reconciliation.m5 import (
    M5Hierarchy,
    M5UpperHierarchy,
    reconcile_m5_upper_forecasts,
)
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
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("data/v2/evidence/m5_rolling_reconciliation"),
    )
    return parser.parse_args()


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _day(index: int) -> str:
    return f"d_{index + 1}"


def _training_set(
    builder: M5DirectFeatureBuilder | M5AggregateFeatureBuilder,
    history: np.ndarray,
    origins: tuple[int, ...],
) -> tuple[pd.DataFrame, np.ndarray]:
    frames: list[pd.DataFrame] = []
    targets: list[np.ndarray] = []
    for origin in origins:
        frame, target = builder.build(
            history, origin_index=origin, horizons=HORIZONS, include_target=True
        )
        frames.append(frame)
        assert target is not None
        targets.append(target)
    return pd.concat(frames, ignore_index=True), np.concatenate(targets)


def _reshape(values: np.ndarray, row_count: int) -> np.ndarray:
    return values.reshape(len(HORIZONS), row_count).T


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
    final_target = folds[-1].evaluation_origin_index + max(HORIZONS)
    days = [f"d_{day}" for day in range(1, final_target + 2)]
    dtypes: dict[str, object] = {day: "float32" for day in days}
    dtypes.update({column: "string" for column in M5_ID_COLUMNS})
    sales = pd.read_csv(
        sales_path, usecols=list(M5_ID_COLUMNS) + days, dtype=dtypes
    )
    bottom = sales.loc[:, M5_ID_COLUMNS].rename(columns={"id": "series_id"})
    hierarchy = M5Hierarchy.from_bottom(bottom)
    hierarchy.validate_official_shape()
    upper = M5UpperHierarchy.from_hierarchy(hierarchy)
    ordered = sales.set_index("id").reindex(hierarchy.bottom["series_id"])
    values = ordered.loc[:, days].to_numpy(dtype=np.float32, copy=False)
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
    price_vectors = load_m5_price_vectors(
        bottom=hierarchy.bottom, prices_path=prices_path, weeks=weeks
    )
    bottom_builder = M5DirectFeatureBuilder(
        bottom=hierarchy.bottom,
        calendar=calendar,
        prices_by_week=price_vectors,
    )
    level9_mask = hierarchy.nodes["level"].to_numpy() == 9
    group_history = hierarchy.summing_matrix[level9_mask] @ values
    upper_history = upper.summing_matrix @ group_history
    aggregate_builder = M5AggregateFeatureBuilder(
        nodes=upper.nodes, calendar=calendar
    )

    fold_records: list[dict[str, object]] = []
    level_frames: list[pd.DataFrame] = []
    for fold_number, fold in enumerate(folds, start=1):
        fold_started = time.perf_counter()
        bottom_x, bottom_y = _training_set(
            bottom_builder, values, fold.training_origin_indices
        )
        bottom_validation_x, bottom_validation_y = bottom_builder.build(
            values,
            origin_index=fold.validation_origin_index,
            horizons=HORIZONS,
            include_target=True,
        )
        assert bottom_validation_y is not None
        bottom_model = M5GlobalLightGBM(n_estimators=args.n_estimators).fit(
            bottom_x,
            bottom_y,
            x_validation=bottom_validation_x,
            y_validation=bottom_validation_y,
        )
        bottom_validation_prediction = _reshape(
            bottom_model.predict(bottom_validation_x), len(hierarchy.bottom)
        )
        bottom_evaluation_x, _ = bottom_builder.build(
            values,
            origin_index=fold.evaluation_origin_index,
            horizons=HORIZONS,
            include_target=False,
        )
        bottom_evaluation_prediction = _reshape(
            bottom_model.predict(bottom_evaluation_x), len(hierarchy.bottom)
        )
        bottom_booster = bottom_model.model
        assert bottom_booster is not None
        bottom_best_iteration = int(bottom_booster.best_iteration_ or args.n_estimators)
        del bottom_x, bottom_y, bottom_validation_x, bottom_validation_y
        del bottom_evaluation_x
        gc.collect()

        aggregate_x, aggregate_y = _training_set(
            aggregate_builder, upper_history, fold.training_origin_indices
        )
        aggregate_validation_x, aggregate_validation_y = aggregate_builder.build(
            upper_history,
            origin_index=fold.validation_origin_index,
            horizons=HORIZONS,
            include_target=True,
        )
        assert aggregate_validation_y is not None
        aggregate_model = M5AggregateLightGBM(
            n_estimators=args.n_estimators
        ).fit(
            aggregate_x,
            aggregate_y,
            x_validation=aggregate_validation_x,
            y_validation=aggregate_validation_y,
        )
        aggregate_validation_prediction = _reshape(
            aggregate_model.predict(aggregate_validation_x), len(upper.nodes)
        )
        aggregate_evaluation_x, _ = aggregate_builder.build(
            upper_history,
            origin_index=fold.evaluation_origin_index,
            horizons=HORIZONS,
            include_target=False,
        )
        aggregate_evaluation_prediction = _reshape(
            aggregate_model.predict(aggregate_evaluation_x), len(upper.nodes)
        )
        aggregate_booster = aggregate_model.model
        assert aggregate_booster is not None
        aggregate_best_iteration = int(
            aggregate_booster.best_iteration_ or args.n_estimators
        )
        del aggregate_x, aggregate_y, aggregate_validation_x, aggregate_validation_y
        del aggregate_evaluation_x
        gc.collect()

        validation_actual = upper_history[
            :,
            fold.validation_origin_index + 1 : fold.validation_origin_index + 29,
        ]
        fallback = values[
            :,
            fold.evaluation_origin_index - 27 : fold.evaluation_origin_index + 1,
        ].sum(axis=1)
        reconciled = reconcile_m5_upper_forecasts(
            upper,
            bottom_validation_forecast=bottom_validation_prediction,
            bottom_evaluation_forecast=bottom_evaluation_prediction,
            aggregate_validation_forecast=aggregate_validation_prediction,
            aggregate_evaluation_forecast=aggregate_evaluation_prediction,
            upper_validation_actual=validation_actual,
            fallback_shares=fallback,
        )
        history = values[:, : fold.evaluation_origin_index + 1]
        actual_bottom = values[
            :,
            fold.evaluation_origin_index + 1 : fold.evaluation_origin_index + 29,
        ]
        actual = hierarchy.summing_matrix @ actual_bottom
        scales = hierarchy_rmsse_scales(
            hierarchy, history, require_positive=False
        )
        weight_start = fold.evaluation_origin_index - 27
        revenue = recent_bottom_revenue(
            bottom=hierarchy.bottom,
            recent_units=history[:, -28:],
            recent_days=[
                _day(index)
                for index in range(weight_start, fold.evaluation_origin_index + 1)
            ],
            calendar_path=calendar_path,
            prices_path=prices_path,
        )
        weights = official_m5_weights(hierarchy, revenue)
        bottom_up_score = score_hierarchical_forecast(
            hierarchy,
            actual=actual,
            prediction=reconciled.bottom_up,
            scales=scales,
            weights=weights,
        )
        reconciled_score = score_hierarchical_forecast(
            hierarchy,
            actual=actual,
            prediction=reconciled.reconciled,
            scales=scales,
            weights=weights,
        )
        relative_improvement = (
            bottom_up_score.overall_wrmsse - reconciled_score.overall_wrmsse
        ) / bottom_up_score.overall_wrmsse
        fold_records.append(
            {
                "fold": fold_number,
                "evaluation_origin_day": _day(fold.evaluation_origin_index),
                "evaluation_start_day": _day(fold.evaluation_origin_index + 1),
                "evaluation_end_day": _day(fold.evaluation_origin_index + 28),
                "bottom_up_wrmsse": bottom_up_score.overall_wrmsse,
                "reconciled_wrmsse": reconciled_score.overall_wrmsse,
                "relative_improvement": relative_improvement,
                "bottom_best_iteration": bottom_best_iteration,
                "aggregate_best_iteration": aggregate_best_iteration,
                "base_coherence_max_abs": reconciled.base_coherence_max_abs,
                "reconciled_coherence_max_abs": reconciled.reconciled_coherence_max_abs,
                "elapsed_seconds": time.perf_counter() - fold_started,
            }
        )
        levels = bottom_up_score.by_level[
            ["level", "level_name", "level_wrmsse"]
        ].rename(columns={"level_wrmsse": "bottom_up_wrmsse"})
        levels["reconciled_wrmsse"] = reconciled_score.by_level["level_wrmsse"]
        levels["relative_improvement"] = (
            levels["bottom_up_wrmsse"] - levels["reconciled_wrmsse"]
        ) / levels["bottom_up_wrmsse"]
        levels.insert(0, "fold", fold_number)
        level_frames.append(levels)
        print(
            f"Fold {fold_number}/{len(folds)} {_day(fold.evaluation_origin_index)}: "
            f"bottom_up={bottom_up_score.overall_wrmsse:.6f}, "
            f"reconciled={reconciled_score.overall_wrmsse:.6f}, "
            f"improvement={relative_improvement:.2%}",
            flush=True,
        )
        del bottom_model, aggregate_model, bottom_booster, aggregate_booster
        del bottom_validation_prediction, bottom_evaluation_prediction
        del aggregate_validation_prediction, aggregate_evaluation_prediction
        del reconciled, history, scales, revenue, weights
        gc.collect()

    fold_results = pd.DataFrame.from_records(fold_records)
    level_results = pd.concat(level_frames, ignore_index=True)
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
    args.output_directory.mkdir(parents=True, exist_ok=True)
    fold_results.to_csv(args.output_directory / "m5_reconciliation_folds.csv", index=False)
    level_results.to_csv(args.output_directory / "m5_reconciliation_levels.csv", index=False)
    level_stability.to_csv(
        args.output_directory / "m5_reconciliation_level_stability.csv", index=False
    )
    summary = {
        "code_revision": _git_revision(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "fold_count": len(folds),
        "improved_folds": int((fold_results["relative_improvement"] > 0).sum()),
        "mean_bottom_up_wrmsse": float(fold_results["bottom_up_wrmsse"].mean()),
        "mean_reconciled_wrmsse": float(fold_results["reconciled_wrmsse"].mean()),
        "mean_relative_improvement": float(fold_results["relative_improvement"].mean()),
        "median_relative_improvement": float(fold_results["relative_improvement"].median()),
        "worst_relative_improvement": float(fold_results["relative_improvement"].min()),
        "maximum_reconciled_coherence_error": float(
            fold_results["reconciled_coherence_max_abs"].max()
        ),
        "source_sha256": {
            path.name: file_sha256(path)
            for path in (calendar_path, sales_path, prices_path)
        },
        "training_windows_per_fold": args.training_windows,
    }
    (args.output_directory / "m5_reconciliation_rolling_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(fold_results.to_string(index=False))
    print(level_stability.to_string(index=False))


if __name__ == "__main__":
    main()
