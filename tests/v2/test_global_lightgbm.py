from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecasting.evaluation.backtest import rolling_origin_panel_backtest
from forecasting.models.base import Forecaster
from forecasting.models.global_lightgbm import GlobalLightGBM


def _panel(periods: int = 48) -> pd.DataFrame:
    dates = pd.date_range("2019-01-01", periods=periods, freq="MS")
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "series_id": series_id,
                    "observation_time": dates,
                    "value": base + np.arange(periods) * trend + 5 * np.sin(
                        np.arange(periods) * 2 * np.pi / 12
                    ),
                }
            )
            for series_id, base, trend in (("a", 100, 1.0), ("b", 300, 2.0))
        ],
        ignore_index=True,
    )


def _model(*, quantiles: tuple[float, ...] = ()) -> GlobalLightGBM:
    return GlobalLightGBM(
        n_estimators=20,
        learning_rate=0.1,
        num_leaves=7,
        min_child_samples=3,
        quantiles=quantiles,
    )


def test_global_lightgbm_is_deterministic_and_recursive() -> None:
    first = _model().fit(_panel()).predict(3)
    second = _model().fit(_panel()).predict(3)
    assert isinstance(_model(), Forecaster)
    assert len(first) == 6
    assert set(first["series_id"]) == {"a", "b"}
    assert set(first["horizon"]) == {1, 2, 3}
    np.testing.assert_allclose(first["prediction"], second["prediction"])


def test_global_lightgbm_quantiles_and_serialization(tmp_path: Path) -> None:
    model = _model(quantiles=(0.1, 0.5, 0.9)).fit(_panel())
    forecasts = model.predict_quantiles(2, (0.1, 0.5, 0.9))
    assert len(forecasts) == 12
    assert set(forecasts["quantile"]) == {0.1, 0.5, 0.9}
    artifact = tmp_path / "global_lightgbm.pkl"
    model.save(artifact)
    restored = GlobalLightGBM.load(artifact)
    np.testing.assert_allclose(
        model.predict(2)["prediction"], restored.predict(2)["prediction"]
    )


def test_panel_backtest_uses_identical_folds_for_every_series() -> None:
    folds = rolling_origin_panel_backtest(
        _panel(),
        _model,
        horizons=(1, 3),
        min_train_size=24,
        step=12,
        data_vintage=datetime(2026, 1, 1, tzinfo=timezone.utc),
        actual_source="fixture",
    )
    assert len(folds) == 8
    assert set(folds["model_name"]) == {"global_lightgbm"}
    counts = folds.groupby(["forecast_origin", "horizon"])["series_id"].nunique()
    assert (counts == 2).all()


def test_global_lightgbm_rejects_unbalanced_panel() -> None:
    panel = _panel().drop(index=[0])
    with pytest.raises(ValueError, match="balanced"):
        _model().fit(panel)
