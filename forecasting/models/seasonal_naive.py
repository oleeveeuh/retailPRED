"""Deterministic seasonal-naive baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


class SeasonalNaive:
    name = "seasonal_naive"
    version = "2.0.0"

    def __init__(self, season_length: int = 12) -> None:
        if season_length < 1:
            raise ValueError("season_length must be positive")
        self.season_length = season_length
        self._values: pd.Series | None = None
        self._series_id: str | None = None

    def fit(self, history: pd.DataFrame) -> "SeasonalNaive":
        required = {"observation_time", "value"}
        if missing := required.difference(history.columns):
            raise ValueError(f"history is missing columns: {sorted(missing)}")
        ordered = history.copy()
        ordered["observation_time"] = pd.to_datetime(ordered["observation_time"])
        ordered = ordered.sort_values("observation_time")
        if ordered["observation_time"].duplicated().any():
            raise ValueError("history has duplicate timestamps")
        if "series_id" in ordered:
            series_ids = ordered["series_id"].astype(str).unique()
            if len(series_ids) != 1:
                raise ValueError("seasonal naive expects exactly one series")
            self._series_id = str(series_ids[0])
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
            record: dict[str, object] = {
                "target_time": target_period.to_timestamp(how="start"),
                "horizon": step,
                "prediction": float(self._values.loc[reference_period]),
            }
            if self._series_id is not None:
                record["series_id"] = self._series_id
            records.append(record)
        return pd.DataFrame.from_records(records)

    def predict_quantiles(
        self, horizon: int, quantiles: tuple[float, ...]
    ) -> pd.DataFrame:
        if not quantiles or any(not 0 < quantile < 1 for quantile in quantiles):
            raise ValueError("quantiles must be strictly between zero and one")
        point = self.predict(horizon)
        return pd.concat(
            [point.assign(quantile=quantile) for quantile in quantiles],
            ignore_index=True,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "parameters": {"season_length": self.season_length},
            "series_ids": [] if self._series_id is None else [self._series_id],
        }

    def save(self, path: Path) -> None:
        if self._values is None:
            raise RuntimeError("fit must be called before save")
        payload = {
            "model": self.metadata(),
            "values": {
                str(period): float(value) for period, value in self._values.items()
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: Path) -> "SeasonalNaive":
        payload = json.loads(path.read_text())
        if payload["model"]["name"] != cls.name:
            raise ValueError("artifact is not a SeasonalNaive model")
        model = cls(season_length=int(payload["model"]["parameters"]["season_length"]))
        model._series_id = next(iter(payload["model"]["series_ids"]), None)
        values = payload["values"]
        model._values = pd.Series(
            [float(value) for value in values.values()],
            index=pd.PeriodIndex(values.keys(), freq="M"),
        )
        return model
