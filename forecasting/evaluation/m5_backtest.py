"""Leakage-safe M5 holdout utilities at the complete 12-level hierarchy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting.reconciliation.m5 import M5Hierarchy, m5_revenue_weights


@dataclass(frozen=True)
class M5BenchmarkResult:
    overall_wrmsse: float
    by_level: pd.DataFrame
    per_series: pd.DataFrame


@dataclass(frozen=True)
class M5RollingFold:
    evaluation_origin_index: int
    validation_origin_index: int
    training_origin_indices: tuple[int, ...]


def m5_rolling_folds(
    *,
    final_origin_index: int = 1_912,
    fold_count: int = 5,
    step: int = 28,
    training_windows: int = 6,
) -> tuple[M5RollingFold, ...]:
    """Declare nested, non-overlapping 28-day M5 train/validation/evaluation folds."""
    if fold_count < 1 or step < 1 or training_windows < 1:
        raise ValueError("fold count, step, and training windows must be positive")
    evaluation_origins = tuple(
        final_origin_index - step * offset
        for offset in reversed(range(fold_count))
    )
    folds: list[M5RollingFold] = []
    for evaluation_origin in evaluation_origins:
        validation_origin = evaluation_origin - step
        training_origins = tuple(
            validation_origin - step * offset
            for offset in range(training_windows, 0, -1)
        )
        if training_origins[0] < 363:
            raise ValueError("rolling fold does not contain 364 days of feature history")
        folds.append(
            M5RollingFold(
                evaluation_origin_index=evaluation_origin,
                validation_origin_index=validation_origin,
                training_origin_indices=training_origins,
            )
        )
    return tuple(folds)


def hierarchy_rmsse_scales(
    hierarchy: M5Hierarchy,
    bottom_history: np.ndarray,
    *,
    block_size: int = 64,
    require_positive: bool = True,
) -> pd.Series:
    """Compute naive-one MSE scales after each node's first non-zero sale."""
    history = np.asarray(bottom_history, dtype=np.float64)
    if history.ndim != 2 or history.shape[0] != len(hierarchy.bottom):
        raise ValueError("bottom history must have one row per hierarchy bottom series")
    if history.shape[1] < 2:
        raise ValueError("RMSSE scale requires at least two training observations")
    if not np.isfinite(history).all() or (history < 0).any():
        raise ValueError("bottom history must be finite and non-negative")

    node_count = len(hierarchy.nodes)
    started = np.zeros(node_count, dtype=bool)
    previous = np.zeros(node_count, dtype=np.float64)
    squared_differences = np.zeros(node_count, dtype=np.float64)
    difference_counts = np.zeros(node_count, dtype=np.int64)
    for start in range(0, history.shape[1], block_size):
        stop = min(start + block_size, history.shape[1])
        aggregated = hierarchy.summing_matrix @ history[:, start:stop]
        for offset in range(aggregated.shape[1]):
            current = aggregated[:, offset]
            continuing = started.copy()
            differences = current - previous
            squared_differences[continuing] += differences[continuing] ** 2
            difference_counts[continuing] += 1
            started |= current != 0
            previous[started] = current[started]

    scales = np.divide(
        squared_differences,
        difference_counts,
        out=np.full(node_count, np.nan, dtype=np.float64),
        where=difference_counts > 0,
    )
    if require_positive and (not np.isfinite(scales).all() or (scales <= 0).any()):
        invalid = hierarchy.nodes.loc[
            ~np.isfinite(scales) | (scales <= 0), "node_id"
        ].tolist()
        raise ValueError(f"M5 nodes require positive RMSSE scales: {invalid[:5]}")
    return pd.Series(scales, index=hierarchy.nodes["node_id"], name="scale")


def seasonal_naive_bottom_forecast(
    bottom_history: np.ndarray, *, horizon: int = 28, seasonal_period: int = 7
) -> np.ndarray:
    """Repeat the most recent complete seasonal cycle for every bottom series."""
    history = np.asarray(bottom_history, dtype=np.float64)
    if history.ndim != 2 or history.shape[1] < seasonal_period:
        raise ValueError("bottom history is shorter than the seasonal period")
    if horizon < 1 or seasonal_period < 1:
        raise ValueError("horizon and seasonal period must be positive")
    cycle = history[:, -seasonal_period:]
    repeats = int(np.ceil(horizon / seasonal_period))
    return np.tile(cycle, (1, repeats))[:, :horizon]


def score_hierarchical_forecast(
    hierarchy: M5Hierarchy,
    *,
    actual: np.ndarray,
    prediction: np.ndarray,
    scales: pd.Series,
    weights: pd.Series,
) -> M5BenchmarkResult:
    """Score node-by-horizon matrices using the fixed M5 WRMSSE contract."""
    actual_values = np.asarray(actual, dtype=np.float64)
    predicted_values = np.asarray(prediction, dtype=np.float64)
    if actual_values.ndim != 2:
        raise ValueError("actual and prediction must share node-by-horizon shape")
    expected_shape = (len(hierarchy.nodes), actual_values.shape[1])
    if actual_values.shape != expected_shape or predicted_values.shape != expected_shape:
        raise ValueError("actual and prediction must share node-by-horizon shape")
    if not np.isfinite(actual_values).all() or not np.isfinite(predicted_values).all():
        raise ValueError("actual and prediction values must be finite")
    node_ids = hierarchy.nodes["node_id"]
    aligned_scales = scales.reindex(node_ids)
    aligned_weights = weights.reindex(node_ids)
    if aligned_weights.isna().any() or (aligned_weights < 0).any():
        raise ValueError("every node requires a non-negative weight")
    if not np.isclose(aligned_weights.sum(), 1.0):
        raise ValueError("M5 weights must sum to one across all levels")
    invalid_scale = aligned_scales.isna() | (aligned_scales <= 0)
    if (aligned_weights[invalid_scale] > 0).any():
        raise ValueError("every positively weighted node requires a positive scale")

    rmse = np.sqrt(np.mean((actual_values - predicted_values) ** 2, axis=1))
    scale_values = aligned_scales.to_numpy(dtype=float)
    rmsse = np.divide(
        rmse,
        np.sqrt(scale_values),
        out=np.full_like(rmse, np.nan),
        where=np.isfinite(scale_values) & (scale_values > 0),
    )
    per_series = hierarchy.nodes[["node_id", "level", "level_name"]].copy()
    per_series["scale"] = aligned_scales.to_numpy()
    per_series["weight"] = aligned_weights.to_numpy()
    per_series["rmse"] = rmse
    per_series["rmsse"] = rmsse
    per_series["weighted_rmsse"] = np.where(
        per_series["weight"] == 0,
        0.0,
        per_series["weight"] * per_series["rmsse"],
    )
    by_level = (
        per_series.groupby(["level", "level_name"], as_index=False)
        .agg(
            series_count=("node_id", "size"),
            weight=("weight", "sum"),
            mean_rmsse=("rmsse", "mean"),
            wrmsse_contribution=("weighted_rmsse", "sum"),
        )
        .sort_values("level", ignore_index=True)
    )
    by_level["level_wrmsse"] = by_level["wrmsse_contribution"] / by_level["weight"]
    return M5BenchmarkResult(
        overall_wrmsse=float(per_series["weighted_rmsse"].sum()),
        by_level=by_level,
        per_series=per_series,
    )


def recent_bottom_revenue(
    *,
    bottom: pd.DataFrame,
    recent_units: np.ndarray,
    recent_days: list[str],
    calendar_path: str | Path,
    prices_path: str | Path,
    price_chunk_size: int = 500_000,
) -> pd.Series:
    """Calculate bottom-series dollar sales using only pre-horizon days."""
    units = np.asarray(recent_units, dtype=np.float64)
    if units.shape != (len(bottom), len(recent_days)):
        raise ValueError("recent units must align with bottom series and day labels")
    calendar = pd.read_csv(calendar_path, usecols=["d", "wm_yr_wk"])
    day_weeks = calendar.set_index("d")["wm_yr_wk"].reindex(recent_days)
    if day_weeks.isna().any():
        raise ValueError("calendar is missing a recent revenue-weight day")
    required_weeks = set(day_weeks.astype(int))
    price_parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        prices_path,
        usecols=["store_id", "item_id", "wm_yr_wk", "sell_price"],
        dtype={"store_id": "string", "item_id": "string", "sell_price": "float32"},
        chunksize=price_chunk_size,
    ):
        selected = chunk.loc[chunk["wm_yr_wk"].isin(required_weeks)]
        if not selected.empty:
            price_parts.append(selected)
    prices = pd.concat(price_parts, ignore_index=True)
    if prices.duplicated(["store_id", "item_id", "wm_yr_wk"]).any():
        raise ValueError("recent M5 prices contain duplicate keys")
    indexed_prices = prices.set_index(["store_id", "item_id", "wm_yr_wk"])["sell_price"]
    pair_index = pd.MultiIndex.from_frame(bottom[["store_id", "item_id"]])
    revenue = np.zeros(len(bottom), dtype=np.float64)
    for offset, week in enumerate(day_weeks.astype(int)):
        lookup = pd.MultiIndex.from_arrays(
            [
                pair_index.get_level_values(0),
                pair_index.get_level_values(1),
                np.full(len(bottom), week),
            ],
            names=["store_id", "item_id", "wm_yr_wk"],
        )
        day_prices = indexed_prices.reindex(lookup).to_numpy(dtype=np.float64)
        sold_without_price = (units[:, offset] > 0) & ~np.isfinite(day_prices)
        if sold_without_price.any():
            raise ValueError("positive recent unit sales are missing a sell price")
        revenue += units[:, offset] * np.nan_to_num(day_prices, nan=0.0)
    return pd.Series(revenue, index=bottom["series_id"], name="revenue")


def official_m5_weights(
    hierarchy: M5Hierarchy, recent_revenue: pd.Series
) -> pd.Series:
    """Expose the official equal-level/recent-dollar-sales weighting contract."""
    return m5_revenue_weights(hierarchy, recent_revenue)
