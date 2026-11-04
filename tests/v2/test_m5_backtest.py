import numpy as np
import pandas as pd
import pytest

from forecasting.evaluation.m5_backtest import (
    hierarchy_rmsse_scales,
    m5_rolling_folds,
    score_hierarchical_forecast,
    seasonal_naive_bottom_forecast,
)
from forecasting.reconciliation.m5 import M5Hierarchy


def _hierarchy() -> M5Hierarchy:
    return M5Hierarchy.from_bottom(
        pd.DataFrame(
            {
                "series_id": ["A", "B"],
                "item_id": ["item_a", "item_b"],
                "dept_id": ["dept", "dept"],
                "cat_id": ["cat", "cat"],
                "store_id": ["store", "store"],
                "state_id": ["state", "state"],
            }
        )
    )


def test_scale_ignores_leading_zeros_and_uses_naive_one_differences() -> None:
    hierarchy = _hierarchy()
    history = np.array([[0, 0, 2, 4], [0, 1, 1, 3]], dtype=float)
    scales = hierarchy_rmsse_scales(hierarchy, history, block_size=2)
    bottom_nodes = hierarchy.nodes.loc[hierarchy.nodes["level"] == 12, "node_id"]
    assert scales.loc[bottom_nodes.iloc[0]] == pytest.approx(4.0)
    assert scales.loc[bottom_nodes.iloc[1]] == pytest.approx(2.0)


def test_weekly_seasonal_naive_repeats_the_last_cycle() -> None:
    history = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=float)
    prediction = seasonal_naive_bottom_forecast(
        history, horizon=5, seasonal_period=2
    )
    np.testing.assert_array_equal(prediction, [[3, 4, 3, 4, 3], [7, 8, 7, 8, 7]])


def test_score_reports_equal_level_contributions() -> None:
    hierarchy = _hierarchy()
    node_count = len(hierarchy.nodes)
    actual = np.ones((node_count, 2))
    prediction = np.zeros((node_count, 2))
    scales = pd.Series(1.0, index=hierarchy.nodes["node_id"])
    weights = pd.Series(
        1.0 / node_count, index=hierarchy.nodes["node_id"]
    )
    result = score_hierarchical_forecast(
        hierarchy,
        actual=actual,
        prediction=prediction,
        scales=scales,
        weights=weights,
    )
    assert result.overall_wrmsse == pytest.approx(1.0)
    assert result.by_level["wrmsse_contribution"].sum() == pytest.approx(1.0)


def test_rolling_folds_keep_training_validation_and_evaluation_disjoint() -> None:
    folds = m5_rolling_folds(
        final_origin_index=1_912, fold_count=3, step=28, training_windows=2
    )
    assert [fold.evaluation_origin_index for fold in folds] == [1_856, 1_884, 1_912]
    for fold in folds:
        assert fold.validation_origin_index + 28 == fold.evaluation_origin_index
        assert fold.training_origin_indices[-1] + 28 == fold.validation_origin_index
        assert all(
            right - left == 28
            for left, right in zip(
                fold.training_origin_indices,
                fold.training_origin_indices[1:],
                strict=False,
            )
        )


def test_zero_weight_prelaunch_node_may_have_undefined_scale() -> None:
    hierarchy = _hierarchy()
    node_count = len(hierarchy.nodes)
    actual = np.ones((node_count, 2))
    prediction = np.zeros((node_count, 2))
    scales = pd.Series(1.0, index=hierarchy.nodes["node_id"])
    weights = pd.Series(1.0 / (node_count - 1), index=hierarchy.nodes["node_id"])
    scales.iloc[-1] = np.nan
    weights.iloc[-1] = 0.0
    result = score_hierarchical_forecast(
        hierarchy,
        actual=actual,
        prediction=prediction,
        scales=scales,
        weights=weights,
    )
    assert result.overall_wrmsse == pytest.approx(1.0)
    weights.iloc[-1] = 0.01
    weights.iloc[0] -= 0.01
    with pytest.raises(ValueError, match="positively weighted"):
        score_hierarchical_forecast(
            hierarchy,
            actual=actual,
            prediction=prediction,
            scales=scales,
            weights=weights,
        )
