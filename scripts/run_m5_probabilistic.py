"""Evaluate training-only scaled conformal intervals for the M5 champion."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd

from forecasting.evaluation.m5_backtest import (
    hierarchy_rmsse_scales,
    m5_rolling_folds,
    official_m5_weights,
    recent_bottom_revenue,
    score_hierarchical_forecast,
)
from forecasting.evaluation.m5_probabilistic import (
    central_interval_metrics,
    finite_sample_quantile,
)
from forecasting.models.m5_global_lightgbm import (
    M5DirectFeatureBuilder,
    M5GlobalLightGBM,
    load_m5_price_vectors,
)
from forecasting.reconciliation.m5 import M5Hierarchy
from retailpred.platform.m5_pipeline import M5_ID_COLUMNS, file_sha256


HORIZONS = tuple(range(1, 29))
NOMINAL_COVERAGE = 0.8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_directory", type=Path, nargs="?", default=Path("data/v2/source/m5")
    )
    parser.add_argument("--calibration-folds", type=int, default=4)
    parser.add_argument("--training-windows", type=int, default=6)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("data/v2/evidence/m5_probabilistic"),
    )
    return parser.parse_args()


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _day(index: int) -> str:
    return f"d_{index + 1}"


def _fit_fold_model(
    *,
    builder: M5DirectFeatureBuilder,
    values: np.ndarray,
    training_origins: tuple[int, ...],
    validation_origin: int,
    n_estimators: int,
    early_stopping_rounds: int,
) -> M5GlobalLightGBM:
    frames: list[pd.DataFrame] = []
    targets: list[np.ndarray] = []
    for origin in training_origins:
        frame, target = builder.build(
            values, origin_index=origin, horizons=HORIZONS, include_target=True
        )
        frames.append(frame)
        assert target is not None
        targets.append(target)
    x_train = pd.concat(frames, ignore_index=True)
    y_train = np.concatenate(targets)
    x_validation, y_validation = builder.build(
        values,
        origin_index=validation_origin,
        horizons=HORIZONS,
        include_target=True,
    )
    assert y_validation is not None
    model = M5GlobalLightGBM(n_estimators=n_estimators).fit(
        x_train,
        y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        early_stopping_rounds=early_stopping_rounds,
    )
    del frames, targets, x_train, y_train, x_validation, y_validation
    gc.collect()
    return model


def _segment_metrics(
    *,
    labels: np.ndarray,
    actual: np.ndarray,
    point: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    scales: np.ndarray,
    segment_type: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label in sorted(np.unique(labels)):
        selected = labels == label
        metrics = central_interval_metrics(
            actual[selected],
            point[selected],
            lower[selected],
            upper[selected],
            scales[selected],
        )
        rows.append(
            {
                "segment_type": segment_type,
                "segment": str(label),
                "series": int(selected.sum()),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.calibration_folds < 1:
        raise ValueError("--calibration-folds must be positive")
    started = time.perf_counter()
    folds = m5_rolling_folds(
        fold_count=args.calibration_folds + 1,
        training_windows=args.training_windows,
    )
    calibration_folds = folds[:-1]
    evaluation_fold = folds[-1]
    source = args.source_directory.resolve()
    sales_path = source / "sales_train_evaluation.csv"
    calendar_path = source / "calendar.csv"
    prices_path = source / "sell_prices.csv"
    calendar = pd.read_csv(calendar_path)
    final_target_index = evaluation_fold.evaluation_origin_index + max(HORIZONS)
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
    prices = load_m5_price_vectors(
        bottom=hierarchy.bottom, prices_path=prices_path, weeks=weeks
    )
    builder = M5DirectFeatureBuilder(
        bottom=hierarchy.bottom, calendar=calendar, prices_by_week=prices
    )
    node_levels = hierarchy.nodes["level"].to_numpy(dtype=int)
    calibration_scores: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    fold_records: list[dict[str, object]] = []
    final_arrays: dict[str, object] = {}

    for fold_number, fold in enumerate(folds, start=1):
        fold_started = time.perf_counter()
        model = _fit_fold_model(
            builder=builder,
            values=values,
            training_origins=fold.training_origin_indices,
            validation_origin=fold.validation_origin_index,
            n_estimators=args.n_estimators,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        evaluation_x, _ = builder.build(
            values,
            origin_index=fold.evaluation_origin_index,
            horizons=HORIZONS,
            include_target=False,
        )
        bottom_point = model.predict(evaluation_x).reshape(
            len(HORIZONS), len(hierarchy.bottom)
        ).T
        history = values[:, : fold.evaluation_origin_index + 1]
        bottom_actual = values[
            :, fold.evaluation_origin_index + 1 : fold.evaluation_origin_index + 29
        ]
        node_actual = hierarchy.summing_matrix @ bottom_actual
        node_point = hierarchy.summing_matrix @ bottom_point
        scales = hierarchy_rmsse_scales(
            hierarchy,
            history,
            require_positive=fold is evaluation_fold,
        ).to_numpy(dtype=float)
        if fold in calibration_folds:
            valid_scale = np.isfinite(scales) & (scales > 0)
            for level in range(1, 13):
                selected = (node_levels == level) & valid_scale
                denominators = np.sqrt(scales[selected])
                for horizon_index, horizon in enumerate(HORIZONS):
                    calibration_scores[(level, horizon)].append(
                        np.abs(
                            node_actual[selected, horizon_index]
                            - node_point[selected, horizon_index]
                        )
                        / denominators
                    )
        else:
            final_arrays = {
                "actual": node_actual,
                "bottom_history": history,
                "bottom_point": bottom_point,
                "point": node_point,
                "scales": scales,
            }
        booster = model.model
        assert booster is not None
        fold_records.append(
            {
                "fold": fold_number,
                "role": "calibration" if fold in calibration_folds else "evaluation",
                "evaluation_origin_day": _day(fold.evaluation_origin_index),
                "evaluation_start_day": _day(fold.evaluation_origin_index + 1),
                "evaluation_end_day": _day(fold.evaluation_origin_index + 28),
                "best_iteration": int(booster.best_iteration_ or args.n_estimators),
                "elapsed_seconds": time.perf_counter() - fold_started,
            }
        )
        del model, booster, evaluation_x, bottom_point, bottom_actual
        del node_actual, node_point, scales, history
        gc.collect()
        print(
            f"Fold {fold_number}/{len(folds)} ({fold_records[-1]['role']}): "
            f"{fold_records[-1]['elapsed_seconds']:.2f}s",
            flush=True,
        )

    adjustment_rows: list[dict[str, object]] = []
    adjustments: dict[tuple[int, int], float] = {}
    for key, parts in sorted(calibration_scores.items()):
        scores = np.concatenate(parts)
        adjustment = finite_sample_quantile(scores, NOMINAL_COVERAGE)
        adjustments[key] = adjustment
        adjustment_rows.append(
            {
                "level": key[0],
                "horizon": key[1],
                "calibration_scores": len(scores),
                "scaled_adjustment": adjustment,
            }
        )

    actual = np.asarray(final_arrays["actual"], dtype=float)
    point = np.asarray(final_arrays["point"], dtype=float)
    scales = np.asarray(final_arrays["scales"], dtype=float)
    lower = np.empty_like(point)
    upper = np.empty_like(point)
    for level in range(1, 13):
        selected = node_levels == level
        scale = np.sqrt(scales[selected])
        for horizon_index, horizon in enumerate(HORIZONS):
            margin = adjustments[(level, horizon)] * scale
            lower[selected, horizon_index] = np.maximum(
                0.0, point[selected, horizon_index] - margin
            )
            upper[selected, horizon_index] = point[selected, horizon_index] + margin

    evaluation_origin = evaluation_fold.evaluation_origin_index
    recent_days = [
        _day(index) for index in range(evaluation_origin - 27, evaluation_origin + 1)
    ]
    bottom_history = np.asarray(final_arrays["bottom_history"], dtype=float)
    revenue = recent_bottom_revenue(
        bottom=hierarchy.bottom,
        recent_units=bottom_history[:, -28:],
        recent_days=recent_days,
        calendar_path=calendar_path,
        prices_path=prices_path,
    )
    weights = official_m5_weights(hierarchy, revenue).reindex(
        hierarchy.nodes["node_id"]
    ).to_numpy(dtype=float)
    overall = central_interval_metrics(
        actual, point, lower, upper, scales, weights=weights
    )
    point_result = score_hierarchical_forecast(
        hierarchy,
        actual=actual,
        prediction=point,
        scales=pd.Series(scales, index=hierarchy.nodes["node_id"]),
        weights=pd.Series(weights, index=hierarchy.nodes["node_id"]),
    )

    level_rows: list[dict[str, object]] = []
    for level in range(1, 13):
        selected = node_levels == level
        level_rows.append(
            {
                "level": level,
                "level_name": hierarchy.nodes.loc[selected, "level_name"].iloc[0],
                "nodes": int(selected.sum()),
                **central_interval_metrics(
                    actual[selected],
                    point[selected],
                    lower[selected],
                    upper[selected],
                    scales[selected],
                ),
            }
        )
    horizon_rows: list[dict[str, object]] = []
    for horizon_index, horizon in enumerate(HORIZONS):
        horizon_rows.append(
            {
                "horizon": horizon,
                **central_interval_metrics(
                    actual[:, horizon_index : horizon_index + 1],
                    point[:, horizon_index : horizon_index + 1],
                    lower[:, horizon_index : horizon_index + 1],
                    upper[:, horizon_index : horizon_index + 1],
                    scales,
                    weights=weights,
                ),
            }
        )

    bottom_selected = node_levels == 12
    bottom_actual_final = actual[bottom_selected]
    bottom_point_final = point[bottom_selected]
    bottom_lower = lower[bottom_selected]
    bottom_upper = upper[bottom_selected]
    bottom_scales = scales[bottom_selected]
    mean_units = bottom_history[:, -56:].mean(axis=1)
    nonzero_rate = (bottom_history[:, -56:] > 0).mean(axis=1)
    volume_labels = np.select(
        [mean_units <= 0.1, mean_units <= 0.5, mean_units <= 2.0],
        ["very_low", "low", "medium"],
        default="high",
    )
    intermittency_labels = np.select(
        [nonzero_rate <= 0.1, nonzero_rate <= 0.5],
        ["intermittent", "mixed"],
        default="regular",
    )
    segment_results = pd.concat(
        [
            _segment_metrics(
                labels=volume_labels,
                actual=bottom_actual_final,
                point=bottom_point_final,
                lower=bottom_lower,
                upper=bottom_upper,
                scales=bottom_scales,
                segment_type="recent_mean_units_56d",
            ),
            _segment_metrics(
                labels=intermittency_labels,
                actual=bottom_actual_final,
                point=bottom_point_final,
                lower=bottom_lower,
                upper=bottom_upper,
                scales=bottom_scales,
                segment_type="nonzero_rate_56d",
            ),
        ],
        ignore_index=True,
    )

    args.output_directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_records).to_csv(
        args.output_directory / "m5_probabilistic_folds.csv", index=False
    )
    pd.DataFrame(adjustment_rows).to_csv(
        args.output_directory / "m5_probabilistic_adjustments.csv", index=False
    )
    level_results = pd.DataFrame(level_rows)
    horizon_results = pd.DataFrame(horizon_rows)
    level_results.to_csv(
        args.output_directory / "m5_probabilistic_by_level.csv", index=False
    )
    horizon_results.to_csv(
        args.output_directory / "m5_probabilistic_by_horizon.csv", index=False
    )
    segment_results.to_csv(
        args.output_directory / "m5_probabilistic_by_segment.csv", index=False
    )
    interval_frame = pd.DataFrame(
        {
            "node_id": np.repeat(hierarchy.nodes["node_id"], len(HORIZONS)),
            "level": np.repeat(node_levels, len(HORIZONS)),
            "horizon": np.tile(HORIZONS, len(hierarchy.nodes)),
            "actual": actual.reshape(-1),
            "p10": lower.reshape(-1),
            "p50": point.reshape(-1),
            "p90": upper.reshape(-1),
        }
    )
    interval_frame.to_parquet(
        args.output_directory / "m5_probabilistic_intervals.parquet", index=False
    )

    level_coverage = level_results["coverage"]
    segment_coverage = segment_results["coverage"]
    calibration_gate_passed = bool(
        0.77 <= overall["coverage"] <= 0.83
        and level_coverage.between(0.70, 0.90).all()
        and (segment_coverage >= 0.70).all()
    )
    summary = {
        "calibration_fold_days": [
            _day(fold.evaluation_origin_index) for fold in calibration_folds
        ],
        "calibration_gate": {
            "overall_coverage_range": [0.77, 0.83],
            "per_level_coverage_range": [0.70, 0.90],
            "minimum_segment_coverage": 0.70,
            "passed": calibration_gate_passed,
        },
        "code_revision": _git_revision(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_days": [
            _day(evaluation_origin + 1),
            _day(evaluation_origin + 28),
        ],
        "lightgbm_version": lightgbm.__version__,
        "method": "level-and-horizon scaled split conformal around coherent LightGBM point forecasts",
        "nominal_coverage": NOMINAL_COVERAGE,
        "overall": overall,
        "point_wrmsse": point_result.overall_wrmsse,
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "source_sha256": {
            path.name: file_sha256(path)
            for path in (calendar_path, sales_path, prices_path)
        },
    }
    (args.output_directory / "m5_probabilistic_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(level_results.to_string(index=False))
    print(segment_results.to_string(index=False))


if __name__ == "__main__":
    main()
