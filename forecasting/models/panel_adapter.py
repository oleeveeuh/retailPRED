"""Fit independent single-series forecasters across a panel with identical origins.

The shared evaluation protocol indexes predictions by ``(series_id, horizon)``,
which single-series models do not produce. This adapter fits one inner model
per panel series and concatenates their predictions so single-series
baselines can be compared against global models fold for fold.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd


class PerSeriesPanelAdapter:
    """Panel-compatible wrapper around any single-series Forecaster factory."""

    def __init__(self, factory: Callable[[], Any]) -> None:
        probe = factory()
        self.name = probe.name
        self.version = probe.version
        self._factory = factory
        self._models: dict[str, Any] = {}

    def fit(self, history: pd.DataFrame) -> "PerSeriesPanelAdapter":
        self._models = {}
        for series_id, group in history.groupby("series_id", sort=True):
            self._models[str(series_id)] = self._factory().fit(group)
        if not self._models:
            raise ValueError("panel history cannot be empty")
        return self

    def _require_fit(self) -> dict[str, Any]:
        if not self._models:
            raise RuntimeError("fit must be called before predict")
        return self._models

    def predict(self, horizon: int) -> pd.DataFrame:
        models = self._require_fit()
        frames = [model.predict(horizon) for model in models.values()]
        return pd.concat(frames, ignore_index=True).sort_values(
            ["series_id", "horizon"], ignore_index=True
        )

    def predict_quantiles(
        self, horizon: int, quantiles: tuple[float, ...]
    ) -> pd.DataFrame:
        models = self._require_fit()
        frames = [model.predict_quantiles(horizon, quantiles) for model in models.values()]
        return pd.concat(frames, ignore_index=True)

    def metadata(self) -> dict[str, Any]:
        models = self._require_fit()
        return {
            "name": self.name,
            "version": self.version,
            "strategy": "per_series_panel_adapter",
            "series_ids": sorted(models),
            "series_metadata": {
                series_id: json.loads(json.dumps(model.metadata()))
                for series_id, model in sorted(models.items())
            },
        }

    def save(self, path: Path) -> None:
        models = self._require_fit()
        root = path.parent / path.stem
        root.mkdir(parents=True, exist_ok=True)
        for series_id, model in models.items():
            model.save(root / f"{series_id}.json")
        (root.parent / f"{path.stem}.metadata.json").write_text(
            json.dumps(self.metadata(), indent=2, sort_keys=True) + "\n"
        )
