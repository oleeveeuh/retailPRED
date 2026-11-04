"""Shared model protocol and auditable metadata for v2 forecasters."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, Self, runtime_checkable

import pandas as pd


@runtime_checkable
class Forecaster(Protocol):
    """Minimum interface used by evaluation, tracking, and serving adapters."""

    name: str
    version: str

    def fit(self, history: pd.DataFrame) -> Self: ...

    def predict(self, horizon: int) -> pd.DataFrame: ...

    def predict_quantiles(
        self, horizon: int, quantiles: tuple[float, ...]
    ) -> pd.DataFrame: ...

    def metadata(self) -> dict[str, Any]: ...

    def save(self, path: Path) -> None: ...
