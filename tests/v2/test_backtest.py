from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from forecasting.evaluation.backtest import rolling_origin_backtest
from forecasting.models.seasonal_naive import SeasonalNaive


def _series() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_id": "retail_total",
            "observation_time": pd.date_range("2018-01-01", periods=60, freq="MS"),
            "value": np.tile(np.arange(1, 13), 5).astype(float),
        }
    )


def test_rolling_origin_records_complete_provenance() -> None:
    result = rolling_origin_backtest(
        _series(),
        SeasonalNaive,
        horizons=(1, 3, 12),
        min_train_size=36,
        step=12,
        data_vintage=datetime(2023, 1, 1, tzinfo=timezone.utc),
        actual_source="census_mrts_2023_revision",
    )
    required = {
        "forecast_origin",
        "horizon",
        "training_cutoff",
        "training_start",
        "data_vintage",
        "model_name",
        "model_version",
        "actual_source",
    }
    assert required.issubset(result.columns)
    assert len(result) == 6
    assert (result["prediction"] == result["actual"]).all()
    assert (result["forecast_origin"] == result["training_cutoff"]).all()


def test_sliding_window_records_later_training_start() -> None:
    result = rolling_origin_backtest(
        _series(),
        SeasonalNaive,
        horizons=(1,),
        min_train_size=24,
        step=12,
        window=24,
        data_vintage=datetime(2023, 1, 1, tzinfo=timezone.utc),
        actual_source="census_mrts_2023_revision",
    )
    assert result.iloc[1]["training_start"] > result.iloc[0]["training_start"]


def test_backtest_requires_timezone_aware_vintage() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        rolling_origin_backtest(
            _series(),
            SeasonalNaive,
            horizons=(1,),
            min_train_size=24,
            data_vintage=datetime(2023, 1, 1),
            actual_source="census_mrts_2023_revision",
        )
