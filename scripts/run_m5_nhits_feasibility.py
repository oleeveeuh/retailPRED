"""Run a bounded, revenue-stratified NHITS feasibility gate on M5."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm
import neuralforecast
import numpy as np
import pandas as pd
import pytorch_lightning
import torch

from forecasting.evaluation.m5_backtest import (
    recent_bottom_revenue,
    seasonal_naive_bottom_forecast,
)
from forecasting.evaluation.m5_feasibility import (
    bottom_rmsse_scales,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_directory", type=Path, nargs="?", default=Path("data/v2/source/m5")
    )
    parser.add_argument("--per-store-category", type=int, default=20)
    parser.add_argument("--history-days", type=int, default=730)
    parser.add_argument("--input-size", type=int, default=112)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("data/v2/evidence/m5_nhits_feasibility"),
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
    if args.history_days < args.input_size + 28:
        raise ValueError("history-days must cover input-size plus validation")
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
        actual,
        seasonal_prediction,
        scales=scales,
        revenue=selected_revenue,
    )

    all_origins = (*TRAINING_ORIGINS, VALIDATION_ORIGIN, TRAIN_END_INDEX)
    required_indices = set(all_origins)
    for origin in all_origins:
        required_indices.update(origin + horizon for horizon in HORIZONS)
    weeks = set(calendar.iloc[sorted(required_indices)]["wm_yr_wk"].astype(int))
    prices = load_m5_price_vectors(
        bottom=selected, prices_path=prices_path, weeks=weeks
    )
    lightgbm_builder = M5DirectFeatureBuilder(
        bottom=selected, calendar=calendar, prices_by_week=prices
    )
    lightgbm_x, lightgbm_y = _build_lightgbm_training(
        lightgbm_builder, selected_values
    )
    lightgbm_validation_x, lightgbm_validation_y = lightgbm_builder.build(
        selected_values,
        origin_index=VALIDATION_ORIGIN,
        horizons=HORIZONS,
        include_target=True,
    )
    assert lightgbm_validation_y is not None
    lightgbm_started = time.perf_counter()
    lightgbm_model = M5GlobalLightGBM().fit(
        lightgbm_x,
        lightgbm_y,
        x_validation=lightgbm_validation_x,
        y_validation=lightgbm_validation_y,
    )
    lightgbm_evaluation_x, _ = lightgbm_builder.build(
        selected_values,
        origin_index=TRAIN_END_INDEX,
        horizons=HORIZONS,
        include_target=False,
    )
    lightgbm_prediction = lightgbm_model.predict(lightgbm_evaluation_x).reshape(
        len(HORIZONS), len(selected)
    ).T
    lightgbm_seconds = time.perf_counter() - lightgbm_started
    lightgbm_score = revenue_weighted_subset_rmsse(
        actual,
        lightgbm_prediction,
        scales=scales,
        revenue=selected_revenue,
    )

    nhits_start_index = TRAIN_END_INDEX + 1 - args.history_days
    train_dates = pd.to_datetime(
        calendar.iloc[nhits_start_index : TRAIN_END_INDEX + 1]["date"]
    ).to_numpy()
    nhits_history = history[:, nhits_start_index : TRAIN_END_INDEX + 1]
    train_frame = pd.DataFrame(
        {
            "unique_id": np.repeat(selected["series_id"].astype(str), len(train_dates)),
            "ds": np.tile(train_dates, len(selected)),
            "y": nhits_history.reshape(-1),
        }
    )
    args.output_directory.mkdir(parents=True, exist_ok=True)
    isolated_train_path = args.output_directory / "nhits_training_input.parquet"
    isolated_prediction_path = args.output_directory / "nhits_raw_predictions.parquet"
    isolated_metadata_path = args.output_directory / "nhits_runtime.json"
    train_frame.to_parquet(isolated_train_path, index=False)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.fit_nhits_subset",
            str(isolated_train_path),
            str(isolated_prediction_path),
            str(isolated_metadata_path),
            "--input-size",
            str(args.input_size),
            "--max-steps",
            str(args.max_steps),
        ],
        check=True,
    )
    forecast = pd.read_parquet(isolated_prediction_path)
    isolated_runtime = json.loads(isolated_metadata_path.read_text())
    nhits_seconds = float(isolated_runtime["elapsed_seconds"])
    isolated_train_path.unlink()
    prediction_columns = [
        column for column in forecast.columns if column not in {"unique_id", "ds"}
    ]
    if len(prediction_columns) != 1:
        raise RuntimeError(f"unexpected NHITS output columns: {prediction_columns}")
    nhits_prediction = (
        forecast.pivot(
            index="unique_id", columns="ds", values=prediction_columns[0]
        )
        .reindex(selected["series_id"].astype(str))
        .to_numpy(dtype=float)
    )
    if nhits_prediction.shape != actual.shape or not np.isfinite(nhits_prediction).all():
        raise RuntimeError("NHITS returned an invalid forecast matrix")
    nhits_prediction = np.clip(nhits_prediction, 0.0, None)
    nhits_score = revenue_weighted_subset_rmsse(
        actual,
        nhits_prediction,
        scales=scales,
        revenue=selected_revenue,
    )

    records: list[pd.DataFrame] = []
    target_dates = pd.to_datetime(
        calendar.iloc[TRAIN_END_INDEX + 1 : TRAIN_END_INDEX + 29]["date"]
    ).to_numpy()
    for model_name, prediction in (
        ("seasonal_naive", seasonal_prediction),
        ("global_lightgbm", lightgbm_prediction),
        ("nhits", nhits_prediction),
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
        args.output_directory / "m5_nhits_subset_predictions.csv", index=False
    )
    selection = selected[
        ["series_id", "store_id", "cat_id", "recent_revenue"]
    ].copy()
    selection.to_csv(args.output_directory / "m5_nhits_subset.csv", index=False)
    full_scale_linear_hours = nhits_seconds * (30_490 / len(selected)) / 3_600
    improvement_vs_naive = (seasonal_score - nhits_score) / seasonal_score
    improvement_vs_lightgbm = (lightgbm_score - nhits_score) / lightgbm_score
    full_scale_recommended = bool(
        improvement_vs_naive >= 0.05
        and improvement_vs_lightgbm > 0
        and full_scale_linear_hours <= 2.0
    )
    summary = {
        "code_revision": _git_revision(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": "cpu",
        "full_scale_linear_runtime_estimate_hours": full_scale_linear_hours,
        "full_scale_recommended": full_scale_recommended,
        "history_days": args.history_days,
        "input_size": args.input_size,
        "lightgbm_seconds": lightgbm_seconds,
        "max_steps": args.max_steps,
        "nhits_improvement_vs_lightgbm": improvement_vs_lightgbm,
        "nhits_improvement_vs_seasonal_naive": improvement_vs_naive,
        "nhits_seconds": nhits_seconds,
        "peak_process_memory_mb": max(
            _peak_memory_mb(), float(isolated_runtime["peak_process_memory_mb"])
        ),
        "scores": {
            "global_lightgbm": lightgbm_score,
            "nhits": nhits_score,
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
            "neuralforecast": neuralforecast.__version__,
            "pytorch_lightning": pytorch_lightning.__version__,
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "warning": "Subset metric is revenue-weighted bottom-level RMSSE, not official 12-level WRMSSE.",
    }
    (args.output_directory / "m5_nhits_feasibility_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Total elapsed seconds: {time.perf_counter() - started:.2f}")


if __name__ == "__main__":
    main()
