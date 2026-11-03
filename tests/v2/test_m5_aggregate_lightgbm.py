import numpy as np
import pandas as pd

from forecasting.models.m5_aggregate_lightgbm import (
    FEATURE_COLUMNS,
    M5AggregateFeatureBuilder,
    M5AggregateLightGBM,
)


def _fixture() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    nodes = pd.DataFrame(
        {"node_id": ["total", "state"], "level": [1, 2]}
    )
    periods = 400
    calendar = pd.DataFrame(
        {
            "wday": np.arange(periods) % 7 + 1,
            "month": np.arange(periods) % 12 + 1,
            "year": 2011 + np.arange(periods) // 365,
            "event_type_1": [None] * periods,
            "event_type_2": [None] * periods,
        }
    )
    history = np.vstack(
        [100 + np.arange(periods) % 7, 50 + np.arange(periods) % 5]
    ).astype(float)
    return nodes, calendar, history


def test_aggregate_features_align_horizon_blocks() -> None:
    nodes, calendar, history = _fixture()
    builder = M5AggregateFeatureBuilder(nodes=nodes, calendar=calendar)
    features, target = builder.build(
        history, origin_index=370, horizons=(1, 2), include_target=True
    )
    assert tuple(features.columns) == FEATURE_COLUMNS
    np.testing.assert_array_equal(features["horizon"], [1, 1, 2, 2])
    np.testing.assert_array_equal(target, np.concatenate([history[:, 371], history[:, 372]]))


def test_aggregate_model_returns_nonnegative_predictions() -> None:
    nodes, calendar, history = _fixture()
    builder = M5AggregateFeatureBuilder(nodes=nodes, calendar=calendar)
    features, target = builder.build(
        history, origin_index=370, horizons=tuple(range(1, 8)), include_target=True
    )
    assert target is not None
    model = M5AggregateLightGBM(n_estimators=10).fit(
        features,
        target,
        x_validation=features,
        y_validation=target,
        early_stopping_rounds=3,
    )
    assert (model.predict(features) >= 0).all()
