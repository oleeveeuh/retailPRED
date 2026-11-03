"""Scaled split-conformal utilities for hierarchical M5 intervals."""

from __future__ import annotations

import numpy as np


def finite_sample_quantile(scores: object, coverage: float = 0.8) -> float:
    """Return the conservative finite-sample conformal score quantile."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not 0 < coverage < 1:
        raise ValueError("coverage must be strictly between zero and one")
    if not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("scores must be a non-empty finite non-negative array")
    rank = min(len(values), int(np.ceil((len(values) + 1) * coverage)))
    return float(np.partition(values, rank - 1)[rank - 1])


def scaled_absolute_errors(
    actual: object, prediction: object, scales: object
) -> np.ndarray:
    """Normalize absolute errors by the RMSSE scale of each hierarchy node."""
    actual_values = np.asarray(actual, dtype=np.float64)
    predicted_values = np.asarray(prediction, dtype=np.float64)
    scale_values = np.asarray(scales, dtype=np.float64)
    if actual_values.shape != predicted_values.shape or actual_values.ndim != 2:
        raise ValueError("actual and prediction must share node-by-horizon shape")
    if scale_values.shape != (actual_values.shape[0],):
        raise ValueError("scales must align with hierarchy nodes")
    if (
        not np.isfinite(actual_values).all()
        or not np.isfinite(predicted_values).all()
        or not np.isfinite(scale_values).all()
        or (scale_values <= 0).any()
    ):
        raise ValueError("probabilistic calibration requires finite positive scales")
    return np.abs(actual_values - predicted_values) / np.sqrt(scale_values[:, None])


def central_interval_metrics(
    actual: object,
    point: object,
    lower: object,
    upper: object,
    scales: object,
    *,
    alpha: float = 0.2,
    weights: object | None = None,
) -> dict[str, float]:
    """Summarize coverage, scaled width, and scale-normalized WIS."""
    actual_values = np.asarray(actual, dtype=np.float64)
    point_values = np.asarray(point, dtype=np.float64)
    lower_values = np.asarray(lower, dtype=np.float64)
    upper_values = np.asarray(upper, dtype=np.float64)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one")
    if not (
        actual_values.shape
        == point_values.shape
        == lower_values.shape
        == upper_values.shape
    ) or actual_values.ndim != 2:
        raise ValueError("forecast arrays must share node-by-horizon shape")
    if not all(
        np.isfinite(values).all()
        for values in (actual_values, point_values, lower_values, upper_values)
    ):
        raise ValueError("forecast arrays must be finite")
    if (lower_values > upper_values).any():
        raise ValueError("lower interval cannot exceed upper interval")
    scale_values = np.asarray(scales, dtype=np.float64)
    if scale_values.shape != (actual_values.shape[0],) or (
        not np.isfinite(scale_values).all() or (scale_values <= 0).any()
    ):
        raise ValueError("scales must be finite, positive, and node-aligned")
    interval_score = (
        upper_values
        - lower_values
        + (2 / alpha) * (lower_values - actual_values) * (actual_values < lower_values)
        + (2 / alpha) * (actual_values - upper_values) * (actual_values > upper_values)
    )
    wis = (0.5 * np.abs(actual_values - point_values) + (alpha / 2) * interval_score) / 1.5
    coverage = ((actual_values >= lower_values) & (actual_values <= upper_values)).astype(float)
    scaled_width = (upper_values - lower_values) / np.sqrt(scale_values[:, None])
    normalized_wis = wis / np.sqrt(scale_values[:, None])
    if weights is None:
        cell_weights = np.full(actual_values.shape, 1 / actual_values.size)
    else:
        node_weights = np.asarray(weights, dtype=np.float64)
        if node_weights.shape != (actual_values.shape[0],) or (
            not np.isfinite(node_weights).all()
            or (node_weights < 0).any()
            or not np.isclose(node_weights.sum(), 1.0)
        ):
            raise ValueError("weights must be non-negative, node-aligned, and sum to one")
        cell_weights = np.repeat(
            (node_weights / actual_values.shape[1])[:, None],
            actual_values.shape[1],
            axis=1,
        )
    return {
        "coverage": float(np.sum(cell_weights * coverage)),
        "mean_scaled_width": float(np.sum(cell_weights * scaled_width)),
        "normalized_wis": float(np.sum(cell_weights * normalized_wis)),
    }
