"""Selection and scoring utilities for bounded M5 model feasibility gates."""

from __future__ import annotations

import numpy as np
import pandas as pd


def select_revenue_stratified_series(
    bottom: pd.DataFrame,
    recent_revenue: pd.Series,
    *,
    per_store_category: int,
) -> pd.DataFrame:
    """Select top-revenue series within every observed store/category stratum."""
    if per_store_category < 1:
        raise ValueError("per_store_category must be positive")
    required = {"series_id", "store_id", "cat_id"}
    if missing := required.difference(bottom.columns):
        raise ValueError(f"bottom metadata is missing columns: {sorted(missing)}")
    frame = bottom.copy()
    frame["recent_revenue"] = recent_revenue.reindex(frame["series_id"]).to_numpy()
    if frame["recent_revenue"].isna().any() or (frame["recent_revenue"] < 0).any():
        raise ValueError("every series requires non-negative recent revenue")
    selected = (
        frame.sort_values(
            ["store_id", "cat_id", "recent_revenue", "series_id"],
            ascending=[True, True, False, True],
        )
        .groupby(["store_id", "cat_id"], observed=True, sort=True)
        .head(per_store_category)
        .sort_values("series_id", ignore_index=True)
    )
    return selected


def bottom_rmsse_scales(history: np.ndarray) -> np.ndarray:
    """Compute first-nonzero naive-one MSE scales for bottom series."""
    values = np.asarray(history, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("history must be a series-by-time matrix")
    scales = np.empty(values.shape[0], dtype=np.float64)
    for row_index, row in enumerate(values):
        nonzero = np.flatnonzero(row != 0)
        if not len(nonzero) or nonzero[0] == len(row) - 1:
            scales[row_index] = np.nan
            continue
        differences = np.diff(row[nonzero[0] :])
        scales[row_index] = float(np.mean(differences**2))
    return scales


def revenue_weighted_subset_rmsse(
    actual: np.ndarray,
    prediction: np.ndarray,
    *,
    scales: np.ndarray,
    revenue: np.ndarray,
) -> float:
    """Score a declared subset; this is not the official 12-level M5 WRMSSE."""
    actual_values = np.asarray(actual, dtype=np.float64)
    prediction_values = np.asarray(prediction, dtype=np.float64)
    scale_values = np.asarray(scales, dtype=np.float64)
    revenue_values = np.asarray(revenue, dtype=np.float64)
    if actual_values.shape != prediction_values.shape or actual_values.ndim != 2:
        raise ValueError("actual and prediction must share series-by-horizon shape")
    if scale_values.shape != (actual_values.shape[0],):
        raise ValueError("scales must align with series")
    if revenue_values.shape != (actual_values.shape[0],):
        raise ValueError("revenue must align with series")
    valid = np.isfinite(scale_values) & (scale_values > 0)
    if (revenue_values[~valid] > 0).any():
        raise ValueError("positively weighted series require a positive scale")
    weights = revenue_values.copy()
    weights[~valid] = 0.0
    if weights.sum() <= 0:
        raise ValueError("subset revenue weights must have positive total")
    weights /= weights.sum()
    rmse = np.sqrt(np.mean((actual_values - prediction_values) ** 2, axis=1))
    rmsse = np.divide(
        rmse,
        np.sqrt(scale_values),
        out=np.zeros_like(rmse),
        where=valid,
    )
    return float(np.sum(weights * rmsse))
