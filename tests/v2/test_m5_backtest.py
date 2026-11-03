import numpy as np
import pandas as pd
import pytest

from forecasting.evaluation.m5_backtest import (
    hierarchy_rmsse_scales,
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
