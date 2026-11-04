"""Compare bottom-up M5 LightGBM with validation-weighted reconciliation."""

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
)
from retailpred.platform.m5_pipeline import M5_ID_COLUMNS, file_sha256


EVALUATION_ORIGIN = 1_912
VALIDATION_ORIGIN = 1_884
HORIZONS = tuple(range(1, 29))
TRAINING_ORIGINS = tuple(range(1_716, 1_857, 28))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_directory", type=Path, nargs="?", default=Path("data/v2/source/m5")
    )
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("data/v2/evidence/m5_reconciliation"),
    )
    return parser.parse_args()


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _build_training_set(
    builder: M5DirectFeatureBuilder | M5AggregateFeatureBuilder,
    history: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    features: list[pd.DataFrame] = []
    targets: list[np.ndarray] = []
    for origin in TRAINING_ORIGINS:
        frame, target = builder.build(
            history, origin_index=origin, horizons=HORIZONS, include_target=True
        )
        features.append(frame)
        assert target is not None
        targets.append(target)
    return pd.concat(features, ignore_index=True), np.concatenate(targets)


def _reshape(prediction: np.ndarray, series_count: int) -> np.ndarray:
    return prediction.reshape(len(HORIZONS), series_count).T


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    source = args.source_directory.resolve()
    sales_path = source / "sales_train_evaluation.csv"
    calendar_path = source / "calendar.csv"
    prices_path = source / "sell_prices.csv"
    calendar = pd.read_csv(calendar_path)
    days = [f"d_{day}" for day in range(1, 1_942)]
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

    all_origins = (*TRAINING_ORIGINS, VALIDATION_ORIGIN, EVALUATION_ORIGIN)
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
    bottom_x, bottom_y = _build_training_set(bottom_builder, values)
    bottom_validation_x, bottom_validation_y = bottom_builder.build(
        values,
        origin_index=VALIDATION_ORIGIN,
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
        origin_index=EVALUATION_ORIGIN,
        horizons=HORIZONS,
        include_target=False,
    )
    bottom_evaluation_prediction = _reshape(
        bottom_model.predict(bottom_evaluation_x), len(hierarchy.bottom)
    )
    del bottom_x, bottom_y, bottom_validation_x, bottom_validation_y
    del bottom_evaluation_x, bottom_builder, price_vectors
    gc.collect()

    level9_mask = hierarchy.nodes["level"].to_numpy() == 9
    group_history = hierarchy.summing_matrix[level9_mask] @ values
    upper_history = upper.summing_matrix @ group_history
    aggregate_builder = M5AggregateFeatureBuilder(
        nodes=upper.nodes, calendar=calendar
    )
    aggregate_x, aggregate_y = _build_training_set(aggregate_builder, upper_history)
    aggregate_validation_x, aggregate_validation_y = aggregate_builder.build(
        upper_history,
        origin_index=VALIDATION_ORIGIN,
        horizons=HORIZONS,
        include_target=True,
    )
    assert aggregate_validation_y is not None
    aggregate_model = M5AggregateLightGBM(n_estimators=args.n_estimators).fit(
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
        origin_index=EVALUATION_ORIGIN,
        horizons=HORIZONS,
        include_target=False,
    )
    aggregate_evaluation_prediction = _reshape(
        aggregate_model.predict(aggregate_evaluation_x), len(upper.nodes)
    )
    del aggregate_x, aggregate_y, aggregate_validation_x, aggregate_validation_y
    del aggregate_evaluation_x, aggregate_builder
    gc.collect()

    bottom_up_validation = hierarchy.summing_matrix @ bottom_validation_prediction
    bottom_up_evaluation = hierarchy.summing_matrix @ bottom_evaluation_prediction
    upper_count = len(upper.nodes)
    upper_bottom_up_validation = bottom_up_validation[:upper_count]
    upper_bottom_up_evaluation = bottom_up_evaluation[:upper_count]
    upper_validation_actual = upper_history[
        :, VALIDATION_ORIGIN + 1 : VALIDATION_ORIGIN + 29
    ]

    blend_weights = np.empty(upper_count, dtype=np.float64)
    for level in range(1, 10):
        mask = upper.nodes["level"].to_numpy() == level
        aggregate_mse = np.mean(
            (aggregate_validation_prediction[mask] - upper_validation_actual[mask]) ** 2
        )
        bottom_up_mse = np.mean(
            (upper_bottom_up_validation[mask] - upper_validation_actual[mask]) ** 2
        )
        blend_weights[mask] = bottom_up_mse / (aggregate_mse + bottom_up_mse)
    validation_base = (
        blend_weights[:, None] * aggregate_validation_prediction
        + (1.0 - blend_weights[:, None]) * upper_bottom_up_validation
    )
    evaluation_base = (
        blend_weights[:, None] * aggregate_evaluation_prediction
        + (1.0 - blend_weights[:, None]) * upper_bottom_up_evaluation
    )
    residual_variance = np.mean(
        (validation_base - upper_validation_actual) ** 2, axis=1
    )
    for level in range(1, 10):
        mask = upper.nodes["level"].to_numpy() == level
        level_median = float(np.median(residual_variance[mask]))
        residual_variance[mask] = 0.5 * residual_variance[mask] + 0.5 * level_median
    positive_floor = max(float(np.median(residual_variance)) * 1e-6, 1e-6)
    residual_variance = np.maximum(residual_variance, positive_floor)
    variances = pd.Series(residual_variance, index=upper.nodes["node_id"])
    reconciled_upper = upper.reconcile(evaluation_base, variances)
    fallback = values[:, EVALUATION_ORIGIN - 27 : EVALUATION_ORIGIN + 1].sum(axis=1)
    reconciled_full = upper.disaggregate_to_bottom(
        reconciled_upper,
        bottom_evaluation_prediction,
        fallback_shares=fallback,
    )

    incoherent_full = bottom_up_evaluation.copy()
    incoherent_full[:upper_count] = evaluation_base
    actual_bottom = values[:, EVALUATION_ORIGIN + 1 : EVALUATION_ORIGIN + 29]
    actual = hierarchy.summing_matrix @ actual_bottom
    history = values[:, : EVALUATION_ORIGIN + 1]
    scales = hierarchy_rmsse_scales(hierarchy, history)
    revenue = recent_bottom_revenue(
        bottom=hierarchy.bottom,
        recent_units=history[:, -28:],
        recent_days=[f"d_{day}" for day in range(1_886, 1_914)],
        calendar_path=calendar_path,
        prices_path=prices_path,
    )
    weights = official_m5_weights(hierarchy, revenue)
    bottom_up_score = score_hierarchical_forecast(
        hierarchy,
        actual=actual,
        prediction=bottom_up_evaluation,
        scales=scales,
        weights=weights,
    )
    base_score = score_hierarchical_forecast(
        hierarchy,
        actual=actual,
        prediction=incoherent_full,
        scales=scales,
        weights=weights,
    )
    reconciled_score = score_hierarchical_forecast(
        hierarchy,
        actual=actual,
        prediction=reconciled_full,
        scales=scales,
        weights=weights,
    )
    base_coherence_error = float(
        np.max(np.abs(incoherent_full - bottom_up_evaluation))
    )
    reconciled_coherence_error = float(
        np.max(
            np.abs(
                reconciled_full
                - hierarchy.summing_matrix
                @ reconciled_full[hierarchy.nodes["level"].to_numpy() == 12]
            )
        )
    )

    comparison = bottom_up_score.by_level[
        ["level", "level_name", "level_wrmsse"]
    ].rename(columns={"level_wrmsse": "bottom_up_wrmsse"})
    comparison["incoherent_base_wrmsse"] = base_score.by_level["level_wrmsse"]
    comparison["reconciled_wrmsse"] = reconciled_score.by_level["level_wrmsse"]
    comparison["reconciled_relative_improvement"] = (
        comparison["bottom_up_wrmsse"] - comparison["reconciled_wrmsse"]
    ) / comparison["bottom_up_wrmsse"]
    args.output_directory.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output_directory / "m5_reconciliation_by_level.csv", index=False)
    pd.DataFrame(
        {
            "level": range(1, 10),
            "aggregate_blend_weight": [
                float(blend_weights[upper.nodes["level"].to_numpy() == level][0])
                for level in range(1, 10)
            ],
        }
    ).to_csv(args.output_directory / "m5_reconciliation_blend_weights.csv", index=False)
    summary = {
        "base_coherence_max_abs": base_coherence_error,
        "bottom_up_wrmsse": bottom_up_score.overall_wrmsse,
        "code_revision": _git_revision(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "incoherent_base_wrmsse": base_score.overall_wrmsse,
        "reconciled_coherence_max_abs": reconciled_coherence_error,
        "reconciled_relative_improvement": (
            bottom_up_score.overall_wrmsse - reconciled_score.overall_wrmsse
        )
        / bottom_up_score.overall_wrmsse,
        "reconciled_wrmsse": reconciled_score.overall_wrmsse,
        "source_sha256": {
            path.name: file_sha256(path)
            for path in (calendar_path, sales_path, prices_path)
        },
        "upper_nodes": upper_count,
        "upper_reconciliation_bottom_groups": len(upper.groups),
    }
    (args.output_directory / "m5_reconciliation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
