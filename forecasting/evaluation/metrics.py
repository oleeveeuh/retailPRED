"""Point and probabilistic forecast metrics with explicit edge cases."""

from __future__ import annotations

import numpy as np


def _arrays(actual: object, prediction: object) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(actual, dtype=float)
    y_hat = np.asarray(prediction, dtype=float)
    if y.shape != y_hat.shape or y.size == 0:
        raise ValueError("actual and prediction must have the same non-empty shape")
    if not np.isfinite(y).all() or not np.isfinite(y_hat).all():
        raise ValueError("metrics require finite values")
    return y, y_hat


def mae(actual: object, prediction: object) -> float:
    y, y_hat = _arrays(actual, prediction)
    return float(np.mean(np.abs(y - y_hat)))


def wape(actual: object, prediction: object) -> float:
    y, y_hat = _arrays(actual, prediction)
    denominator = np.sum(np.abs(y))
    if denominator == 0:
        raise ValueError("WAPE is undefined when all actuals are zero")
    return float(np.sum(np.abs(y - y_hat)) / denominator)


def bias(actual: object, prediction: object) -> float:
    y, y_hat = _arrays(actual, prediction)
    return float(np.mean(y_hat - y))


def rmsse(actual: object, prediction: object, training: object, season_length: int = 1) -> float:
    y, y_hat = _arrays(actual, prediction)
    train = np.asarray(training, dtype=float)
    if season_length < 1 or train.size <= season_length:
        raise ValueError("training history must exceed season_length")
    scale = np.mean((train[season_length:] - train[:-season_length]) ** 2)
    if scale == 0:
        raise ValueError("RMSSE is undefined for constant seasonal training history")
    return float(np.sqrt(np.mean((y - y_hat) ** 2) / scale))


def pinball_loss(actual: object, quantile_prediction: object, quantile: float) -> float:
    if not 0 < quantile < 1:
        raise ValueError("quantile must be between zero and one")
    y, q_hat = _arrays(actual, quantile_prediction)
    error = y - q_hat
    return float(np.mean(np.maximum(quantile * error, (quantile - 1) * error)))


def interval_coverage(actual: object, lower: object, upper: object) -> float:
    y, lo = _arrays(actual, lower)
    _, hi = _arrays(actual, upper)
    if np.any(lo > hi):
        raise ValueError("lower interval cannot exceed upper interval")
    return float(np.mean((y >= lo) & (y <= hi)))


def interval_width(lower: object, upper: object) -> float:
    lo, hi = _arrays(lower, upper)
    if np.any(lo > hi):
        raise ValueError("lower interval cannot exceed upper interval")
    return float(np.mean(hi - lo))


def interval_score(actual: object, lower: object, upper: object, alpha: float) -> np.ndarray:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    y, lo = _arrays(actual, lower)
    _, hi = _arrays(actual, upper)
    if np.any(lo > hi):
        raise ValueError("lower interval cannot exceed upper interval")
    return (hi - lo) + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)


def weighted_interval_score(
    actual: object,
    median: object,
    intervals: list[tuple[float, object, object]],
) -> float:
    y, med = _arrays(actual, median)
    if not intervals:
        raise ValueError("at least one central interval is required")
    total = 0.5 * np.abs(y - med)
    for alpha, lower, upper in intervals:
        total = total + (alpha / 2) * interval_score(y, lower, upper, alpha)
    return float(np.mean(total / (len(intervals) + 0.5)))
