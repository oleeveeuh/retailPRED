import numpy as np
import pandas as pd

from forecasting.models.seasonal_naive import SeasonalNaive


def test_seasonal_naive_repeats_last_year_deterministically() -> None:
    history = pd.DataFrame(
        {
            "observation_time": pd.date_range("2022-01-01", periods=24, freq="MS"),
            "value": np.arange(24, dtype=float),
        }
    )
    first = SeasonalNaive().fit(history).predict(3)
    second = SeasonalNaive().fit(history).predict(3)
    pd.testing.assert_frame_equal(first, second)
    assert first["prediction"].tolist() == [12.0, 13.0, 14.0]
