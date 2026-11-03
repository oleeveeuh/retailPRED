"""Run the predeclared zero-shot Chronos-2 feasibility gate on M5."""

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

import lightgbm
import numpy as np
import pandas as pd

from forecasting.evaluation.m5_backtest import (
    recent_bottom_revenue,
    seasonal_naive_bottom_forecast,
)
from forecasting.evaluation.m5_feasibility import (
    bottom_rmsse_scales,
    foundation_model_gate,
    revenue_weighted_subset_rmsse,
    select_revenue_stratified_series,
)
from forecasting.models.m5_global_lightgbm import (
    M5DirectFeatureBuilder,
    M5GlobalLightGBM,
    load_m5_price_vectors,
)
from retailpred.platform.m5_pipeline import M5_ID_COLUMNS, file_sha256


TRAIN_END_INDEX = 1_912
VALIDATION_ORIGIN = 1_884
HORIZONS = tuple(range(1, 29))
TRAINING_ORIGINS = tuple(range(1_716, 1_857, 28))
TOTAL_BOTTOM_SERIES = 30_490


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_directory", type=Path, nargs="?", default=Path("data/v2/source/m5")
    )
    parser.add_argument("--foundation-python", type=Path, required=True)
    parser.add_argument("--per-store-category", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-id", default="amazon/chronos-2")
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("data/v2/evidence/m5_chronos_feasibility"),
    )
    return parser.parse_args()


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _peak_memory_mb() -> float:
    usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return usage / (1024 * 1024) if platform.system() == "Darwin" else usage / 1024


def _build_lightgbm_training(
    builder: M5DirectFeatureBuilder, values: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray]:
    frames: list[pd.DataFrame] = []
    targets: list[np.ndarray] = []
    for origin in TRAINING_ORIGINS:
        frame, target = builder.build(
            values, origin_index=origin, horizons=HORIZONS, include_target=True
        )
        frames.append(frame)
        assert target is not None
        targets.append(target)
    return pd.concat(frames, ignore_index=True), np.concatenate(targets)


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
    values = sales.loc[:, days].to_numpy(dtype=np.float32, copy=False)
    recent_days = [f"d_{day}" for day in range(1_886, 1_914)]
    revenue = recent_bottom_revenue(
        bottom=bottom,
        recent_units=values[:, TRAIN_END_INDEX - 27 : TRAIN_END_INDEX + 1],
        recent_days=recent_days,
        calendar_path=calendar_path,
        prices_path=prices_path,
    )
    selected = select_revenue_stratified_series(
        bottom, revenue, per_store_category=args.per_store_category
    )
    row_lookup = pd.Series(np.arange(len(bottom)), index=bottom["series_id"])
    selected_rows = row_lookup.reindex(selected["series_id"]).to_numpy(dtype=int)
    selected_values = values[selected_rows]
    selected_revenue = revenue.reindex(selected["series_id"]).to_numpy(dtype=float)
    del sales, values

    history = selected_values[:, : TRAIN_END_INDEX + 1]
    actual = selected_values[:, TRAIN_END_INDEX + 1 : TRAIN_END_INDEX + 29]
    scales = bottom_rmsse_scales(history)
    seasonal_prediction = seasonal_naive_bottom_forecast(history)
    seasonal_score = revenue_weighted_subset_rmsse(
        actual, seasonal_prediction, scales=scales, revenue=selected_revenue
    )

    all_origins = (*TRAINING_ORIGINS, VALIDATION_ORIGIN, TRAIN_END_INDEX)
    required_indices = set(all_origins)
    for origin in all_origins:
        required_indices.update(origin + horizon for horizon in HORIZONS)
    weeks = set(calendar.iloc[sorted(required_indices)]["wm_yr_wk"].astype(int))
    prices = load_m5_price_vectors(
        bottom=selected, prices_path=prices_path, weeks=weeks
    )
    builder = M5DirectFeatureBuilder(
        bottom=selected, calendar=calendar, prices_by_week=prices
    )
    training_x, training_y = _build_lightgbm_training(builder, selected_values)
    validation_x, validation_y = builder.build(
        selected_values,
        origin_index=VALIDATION_ORIGIN,
        horizons=HORIZONS,
        include_target=True,
    )
    assert validation_y is not None
    lightgbm_started = time.perf_counter()
    model = M5GlobalLightGBM().fit(
        training_x,
        training_y,
        x_validation=validation_x,
        y_validation=validation_y,
    )
    evaluation_x, _ = builder.build(
        selected_values,
        origin_index=TRAIN_END_INDEX,
        horizons=HORIZONS,
        include_target=False,
    )
    lightgbm_prediction = model.predict(evaluation_x).reshape(
        len(HORIZONS), len(selected)
    ).T
    lightgbm_seconds = time.perf_counter() - lightgbm_started
    lightgbm_score = revenue_weighted_subset_rmsse(
        actual, lightgbm_prediction, scales=scales, revenue=selected_revenue
    )

    context_dates = pd.to_datetime(
        calendar.iloc[: TRAIN_END_INDEX + 1]["date"]
    ).to_numpy()
    context = pd.DataFrame(
        {
            "series_id": np.repeat(
                selected["series_id"].astype(str), len(context_dates)
            ),
            "timestamp": np.tile(context_dates, len(selected)),
            "target": history.reshape(-1),
        }
    )
    args.output_directory.mkdir(parents=True, exist_ok=True)
    context_path = args.output_directory / "chronos_context.parquet"
    raw_prediction_path = args.output_directory / "chronos_raw_predictions.parquet"
    runtime_path = args.output_directory / "chronos_runtime.json"
    context.to_parquet(context_path, index=False)
    child_environment = os.environ.copy()
    child_environment["HF_HOME"] = str(
        (Path("data/v2/cache/huggingface")).resolve()
    )
    subprocess.run(
        [
            str(args.foundation_python.resolve()),
            "-m",
            "scripts.predict_chronos_subset",
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
    forecast = pd.read_parquet(raw_prediction_path)
    chronos_prediction = (
        forecast.pivot(index="series_id", columns="timestamp", values="predictions")
        .reindex(selected["series_id"].astype(str))
        .to_numpy(dtype=float)
    )
    if chronos_prediction.shape != actual.shape or not np.isfinite(
        chronos_prediction
    ).all():
        raise RuntimeError("Chronos-2 returned an invalid forecast matrix")
    chronos_prediction = np.clip(chronos_prediction, 0.0, None)
    chronos_score = revenue_weighted_subset_rmsse(
        actual, chronos_prediction, scales=scales, revenue=selected_revenue
    )

    target_dates = pd.to_datetime(
        calendar.iloc[TRAIN_END_INDEX + 1 : TRAIN_END_INDEX + 29]["date"]
    ).to_numpy()
    records: list[pd.DataFrame] = []
    for model_name, prediction in (
        ("seasonal_naive", seasonal_prediction),
        ("global_lightgbm", lightgbm_prediction),
        ("chronos_2_zero_shot", chronos_prediction),
    ):
        records.append(
            pd.DataFrame(
                {
                    "series_id": np.repeat(
                        selected["series_id"].astype(str), len(target_dates)
                    ),
                    "target_time": np.tile(target_dates, len(selected)),
                    "actual": actual.reshape(-1),
                    "prediction": prediction.reshape(-1),
                    "model": model_name,
                }
            )
        )
    pd.concat(records, ignore_index=True).to_csv(
        args.output_directory / "m5_chronos_subset_predictions.csv", index=False
    )
    selected[["series_id", "store_id", "cat_id", "recent_revenue"]].to_csv(
        args.output_directory / "m5_chronos_subset.csv", index=False
    )

    inference_seconds = float(runtime["inference_seconds"])
    full_scale_hours = inference_seconds * (TOTAL_BOTTOM_SERIES / len(selected)) / 3_600
    gate = foundation_model_gate(
        seasonal_naive_score=seasonal_score,
        incumbent_score=lightgbm_score,
        challenger_score=chronos_score,
        full_scale_runtime_hours=full_scale_hours,
    )
    summary = {
        "code_revision": _git_revision(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": "cpu",
        "full_scale_linear_runtime_estimate_hours": full_scale_hours,
        **gate,
        "lightgbm_seconds": lightgbm_seconds,
        "model_id": args.model_id,
        "point_forecast": "Chronos-2 predictions output; no quantile selected",
        "runtime": runtime,
        "scores": {
            "chronos_2_zero_shot": chronos_score,
            "global_lightgbm": lightgbm_score,
            "seasonal_naive": seasonal_score,
        },
        "selection": {
            "per_store_category": args.per_store_category,
            "series": len(selected),
            "store_category_strata": int(
                selected.groupby(["store_id", "cat_id"]).ngroups
            ),
        },
        "source_sha256": {
            path.name: file_sha256(path)
            for path in (calendar_path, sales_path, prices_path)
        },
        "versions": {
            "lightgbm": lightgbm.__version__,
            "python": platform.python_version(),
        },
        "warning": "Subset metric is revenue-weighted bottom-level RMSSE, not official 12-level WRMSSE.",
        "zero_shot": True,
    }
    (args.output_directory / "m5_chronos_feasibility_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Total elapsed seconds: {time.perf_counter() - started:.2f}")


if __name__ == "__main__":
    main()
