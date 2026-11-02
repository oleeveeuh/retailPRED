from datetime import datetime, timezone

import numpy as np
import pandas as pd

from forecasting.data.monthly_dataset import observations_as_of
from forecasting.evaluation.backtest import rolling_origin_backtest
from forecasting.evaluation.metrics import mae, wape
from forecasting.features.monthly import build_causal_features
from forecasting.models.seasonal_naive import SeasonalNaive


def test_canonical_monthly_pipeline_runs_end_to_end() -> None:
    dates = pd.date_range("2018-01-01", periods=72, freq="MS")
    vintage = datetime(2025, 1, 1, tzinfo=timezone.utc)
    canonical = pd.DataFrame(
        {
            "series_id": "retail_total",
            "observation_time": dates,
            "value": np.tile(np.arange(100, 112), 6).astype(float),
            "vintage_time": vintage,
            "ingested_at": vintage,
            "source": "test_fixture",
        }
    )

    selected = observations_as_of(canonical, vintage)
    features = build_causal_features(
        selected[["series_id", "observation_time", "value"]]
    )
    assert features.loc[12, "lag_12"] == features.loc[0, "value"]

    folds = rolling_origin_backtest(
        selected[["series_id", "observation_time", "value"]],
        SeasonalNaive,
        horizons=(1, 3, 6, 12),
        min_train_size=36,
        data_vintage=vintage,
        actual_source="test_fixture",
    )
    assert len(folds) == 100
    assert mae(folds["actual"], folds["prediction"]) == 0
    assert wape(folds["actual"], folds["prediction"]) == 0
