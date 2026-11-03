from pathlib import Path

import numpy as np
import pandas as pd

from forecasting.models.base import Forecaster
from forecasting.models.seasonal_naive import SeasonalNaive


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_id": "retail_total",
            "observation_time": pd.date_range("2020-01-01", periods=24, freq="MS"),
            "value": np.tile(np.arange(1, 13), 2),
        }
    )


def test_seasonal_naive_implements_common_protocol_and_round_trips(tmp_path: Path) -> None:
    model = SeasonalNaive().fit(_history())
    assert isinstance(model, Forecaster)
    assert model.metadata()["parameters"]["season_length"] == 12
    quantiles = model.predict_quantiles(2, (0.1, 0.5, 0.9))
    assert set(quantiles["quantile"]) == {0.1, 0.5, 0.9}

    artifact = tmp_path / "seasonal_naive.json"
    model.save(artifact)
    restored = SeasonalNaive.load(artifact)
    pd.testing.assert_frame_equal(model.predict(2), restored.predict(2))
