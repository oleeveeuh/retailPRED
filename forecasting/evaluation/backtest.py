"""Rolling-origin evaluation at the series' native monthly frequency."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Protocol

import pandas as pd

from forecasting.features.monthly import validate_monthly_panel


class Forecaster(Protocol):
    name: str
    version: str

    def fit(self, history: pd.DataFrame) -> "Forecaster": ...
    def predict(self, horizon: int) -> pd.DataFrame: ...


def rolling_origin_backtest(
    frame: pd.DataFrame,
    model_factory: Callable[[], Forecaster],
    *,
    horizons: Sequence[int] = (1, 3, 6, 12),
    min_train_size: int = 36,
    step: int = 1,
    window: int | None = None,
    data_vintage: datetime,
    actual_source: str,
) -> pd.DataFrame:
    """Evaluate one series on expanding or fixed-length rolling windows."""
    if data_vintage.tzinfo is None or data_vintage.utcoffset() is None:
        raise ValueError("data_vintage must be timezone-aware")
    clean = validate_monthly_panel(frame)
    if clean["series_id"].nunique() != 1:
        raise ValueError("backtest expects exactly one series")
    requested = sorted(set(horizons))
    if not requested or requested[0] < 1:
        raise ValueError("horizons must contain positive integers")
    if min_train_size < 1 or step < 1:
        raise ValueError("min_train_size and step must be positive")
    if window is not None and window < min_train_size:
        raise ValueError("window cannot be smaller than min_train_size")

    max_horizon = requested[-1]
    if len(clean) < min_train_size + max_horizon:
        raise ValueError("series is too short for the requested training window and horizon")
    records: list[dict[str, object]] = []
    for cutoff_index in range(min_train_size - 1, len(clean) - max_horizon, step):
        start = 0 if window is None else max(0, cutoff_index + 1 - window)
        training = clean.iloc[start : cutoff_index + 1]
        model = model_factory().fit(training)
        predictions = model.predict(max_horizon).set_index("horizon")
        origin = pd.Timestamp(clean.iloc[cutoff_index]["observation_time"])
        for horizon in requested:
            actual_row = clean.iloc[cutoff_index + horizon]
            predicted = predictions.loc[horizon]
            target_time = pd.Timestamp(actual_row["observation_time"])
            if pd.Timestamp(predicted["target_time"]) != target_time:
                raise ValueError("model target timestamp does not match the evaluation panel")
            records.append(
                {
                    "series_id": str(actual_row["series_id"]),
                    "forecast_origin": origin,
                    "target_time": target_time,
                    "horizon": horizon,
                    "training_cutoff": origin,
                    "training_start": pd.Timestamp(training.iloc[0]["observation_time"]),
                    "data_vintage": data_vintage,
                    "model_name": model.name,
                    "model_version": model.version,
                    "prediction": float(predicted["prediction"]),
                    "actual": float(actual_row["value"]),
                    "actual_source": actual_source,
                }
            )
    return pd.DataFrame.from_records(records)
