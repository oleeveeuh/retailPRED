"""Deterministic seasonal-naive baseline."""

from __future__ import annotations

import pandas as pd


class SeasonalNaive:
    name = "seasonal_naive"
    version = "2.0.0"

    def __init__(self, season_length: int = 12) -> None:
        if season_length < 1:
            raise ValueError("season_length must be positive")
        self.season_length = season_length
        self._values: pd.Series | None = None

    def fit(self, history: pd.DataFrame) -> "SeasonalNaive":
        required = {"observation_time", "value"}
        if missing := required.difference(history.columns):
            raise ValueError(f"history is missing columns: {sorted(missing)}")
        ordered = history.copy()
        ordered["observation_time"] = pd.to_datetime(ordered["observation_time"])
        ordered = ordered.sort_values("observation_time")
        if ordered["observation_time"].duplicated().any():
            raise ValueError("history has duplicate timestamps")
        if len(ordered) < self.season_length:
            raise ValueError(f"seasonal naive requires at least {self.season_length} observations")
        periods = ordered["observation_time"].dt.to_period("M")
        self._values = pd.Series(ordered["value"].to_numpy(dtype=float), index=periods)
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        if self._values is None:
            raise RuntimeError("fit must be called before predict")
        if horizon < 1:
            raise ValueError("horizon must be positive")
        last_period = self._values.index[-1]
        records: list[dict[str, object]] = []
        for step in range(1, horizon + 1):
            target_period = last_period + step
            reference_period = target_period - self.season_length
            if reference_period not in self._values.index:
                raise ValueError("horizon exceeds the available seasonal reference history")
            records.append(
                {
                    "target_time": target_period.to_timestamp(how="start"),
                    "horizon": step,
                    "prediction": float(self._values.loc[reference_period]),
                }
            )
        return pd.DataFrame.from_records(records)
