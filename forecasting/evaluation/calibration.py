"""Time-separated conformal calibration for central forecast intervals."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def enforce_non_crossing_quantiles(
    frame: pd.DataFrame, quantile_columns: tuple[str, ...]
) -> pd.DataFrame:
    """Apply monotone rearrangement without changing row membership."""
    if missing := set(quantile_columns).difference(frame.columns):
        raise ValueError(f"missing quantile columns: {sorted(missing)}")
    result = frame.copy()
    result.loc[:, quantile_columns] = np.sort(
        result.loc[:, quantile_columns].to_numpy(dtype=float), axis=1
    )
    return result


@dataclass
class ConformalIntervalCalibrator:
    """Calibrate an interval using earlier, disjoint forecast errors."""

    alpha: float = 0.2
    group_columns: tuple[str, ...] = ("horizon",)
    method: str = "multiplicative"
    adjustments: dict[tuple[object, ...], float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be strictly between zero and one")
        if not self.group_columns:
            raise ValueError("at least one calibration group is required")
        if self.method not in {"additive", "multiplicative"}:
            raise ValueError("method must be additive or multiplicative")

    def fit(
        self,
        calibration: pd.DataFrame,
        *,
        lower_column: str,
        upper_column: str,
    ) -> "ConformalIntervalCalibrator":
        required = {"actual", lower_column, upper_column, *self.group_columns}
        if missing := required.difference(calibration.columns):
            raise ValueError(f"calibration data is missing columns: {sorted(missing)}")
        if calibration.empty:
            raise ValueError("calibration data cannot be empty")
        self.adjustments = {}
        grouper: str | list[str]
        grouper = (
            self.group_columns[0]
            if len(self.group_columns) == 1
            else list(self.group_columns)
        )
        for raw_key, group in calibration.groupby(grouper, sort=True):
            key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
            actual = group["actual"].to_numpy(dtype=float)
            lower = group[lower_column].to_numpy(dtype=float)
            upper = group[upper_column].to_numpy(dtype=float)
            if np.any(lower > upper):
                raise ValueError("calibration intervals must be non-crossing")
            if self.method == "multiplicative":
                if np.any(actual <= 0) or np.any(lower <= 0) or np.any(upper <= 0):
                    raise ValueError("multiplicative calibration requires positive values")
                scores = np.maximum(np.log(lower / actual), np.log(actual / upper))
            else:
                scores = np.maximum(lower - actual, actual - upper)
            rank = min(len(scores), int(np.ceil((len(scores) + 1) * (1 - self.alpha))))
            adjustment = float(np.sort(scores)[rank - 1])
            self.adjustments[key] = max(0.0, adjustment)
        return self

    def transform(
        self,
        frame: pd.DataFrame,
        *,
        lower_column: str,
        upper_column: str,
        output_lower: str = "lower_calibrated",
        output_upper: str = "upper_calibrated",
    ) -> pd.DataFrame:
        if not self.adjustments:
            raise RuntimeError("fit must be called before transform")
        result = frame.copy()

        def adjustment(row: pd.Series) -> float:
            key = tuple(row[column] for column in self.group_columns)
            if key not in self.adjustments:
                raise ValueError(f"no conformal adjustment for group {key}")
            return self.adjustments[key]

        corrections = result.apply(adjustment, axis=1)
        if self.method == "multiplicative":
            result[output_lower] = result[lower_column] / np.exp(corrections)
            result[output_upper] = result[upper_column] * np.exp(corrections)
        else:
            result[output_lower] = result[lower_column] - corrections
            result[output_upper] = result[upper_column] + corrections
        return result


def temporal_calibration_split(
    frame: pd.DataFrame,
    *,
    calibration_end: str | pd.Timestamp,
    evaluation_start: str | pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by target time so calibration actuals cannot enter evaluation."""
    if "target_time" not in frame:
        raise ValueError("frame must contain target_time")
    end = pd.Timestamp(calibration_end)
    start = pd.Timestamp(evaluation_start)
    if end >= start:
        raise ValueError("calibration_end must precede evaluation_start")
    target = pd.to_datetime(frame["target_time"])
    calibration = frame.loc[target <= end].copy()
    evaluation = frame.loc[target >= start].copy()
    if calibration.empty or evaluation.empty:
        raise ValueError("both temporal partitions must contain rows")
    return calibration, evaluation
