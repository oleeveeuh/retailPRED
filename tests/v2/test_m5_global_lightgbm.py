from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecasting.models.m5_global_lightgbm import (
    FEATURE_COLUMNS,
    M5DirectFeatureBuilder,
    M5GlobalLightGBM,
)


def _fixture() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict[int, np.ndarray]]:
    bottom = pd.DataFrame(
        {
            "series_id": ["A", "B"],
            "item_id": ["i1", "i2"],
            "dept_id": ["d", "d"],
            "cat_id": ["c", "c"],
            "store_id": ["CA_1", "TX_1"],
            "state_id": ["CA", "TX"],
        }
    )
    periods = 400
    calendar = pd.DataFrame(
        {
            "wm_yr_wk": 1000 + np.arange(periods) // 7,
            "wday": np.arange(periods) % 7 + 1,
            "month": np.arange(periods) % 12 + 1,
            "year": 2011 + np.arange(periods) // 365,
            "event_type_1": [None] * periods,
            "event_type_2": [None] * periods,
            "snap_CA": np.arange(periods) % 2,
            "snap_TX": (np.arange(periods) + 1) % 2,
            "snap_WI": np.zeros(periods),
        }
    )
    history = np.vstack([np.arange(periods) % 5, np.arange(periods) % 3]).astype(float)
    prices = {
        int(week): np.array([2.0, 3.0], dtype=np.float32)
        for week in calendar["wm_yr_wk"].unique()
    }
    return bottom, calendar, history, prices


def test_direct_features_are_origin_safe_and_horizon_aligned() -> None:
    bottom, calendar, history, prices = _fixture()
    builder = M5DirectFeatureBuilder(
        bottom=bottom, calendar=calendar, prices_by_week=prices
    )
    x, y = builder.build(
        history, origin_index=370, horizons=(1, 2), include_target=True
    )
    assert tuple(x.columns) == FEATURE_COLUMNS
    assert len(x) == 4
    np.testing.assert_array_equal(x["horizon"], [1, 1, 2, 2])
    np.testing.assert_array_equal(x.loc[:1, "lag_1"], history[:, 370])
    np.testing.assert_array_equal(y, np.concatenate([history[:, 371], history[:, 372]]))


def test_direct_features_reject_targets_beyond_history() -> None:
    bottom, calendar, history, prices = _fixture()
    builder = M5DirectFeatureBuilder(
        bottom=bottom, calendar=calendar, prices_by_week=prices
    )
    with pytest.raises(ValueError, match="training targets"):
        builder.build(history[:, :395], origin_index=390, horizons=(6,), include_target=True)


def test_m5_lightgbm_is_deterministic_and_serializable(tmp_path: Path) -> None:
    bottom, calendar, history, prices = _fixture()
    builder = M5DirectFeatureBuilder(
        bottom=bottom, calendar=calendar, prices_by_week=prices
    )
    x, y = builder.build(
        history, origin_index=370, horizons=tuple(range(1, 8)), include_target=True
    )
    first = M5GlobalLightGBM(
        n_estimators=10, num_leaves=7, min_child_samples=2
    ).fit(x, y)
    second = M5GlobalLightGBM(
        n_estimators=10, num_leaves=7, min_child_samples=2
    ).fit(x, y)
    np.testing.assert_allclose(first.predict(x), second.predict(x))
    artifact = tmp_path / "m5_lightgbm.pkl"
    first.save(artifact)
    np.testing.assert_allclose(first.predict(x), M5GlobalLightGBM.load(artifact).predict(x))
