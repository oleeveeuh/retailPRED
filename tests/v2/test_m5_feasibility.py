import numpy as np
import pandas as pd
import pytest

from forecasting.evaluation.m5_feasibility import (
    bottom_rmsse_scales,
    revenue_weighted_subset_rmsse,
    select_revenue_stratified_series,
)


def test_revenue_selection_preserves_store_category_coverage() -> None:
    bottom = pd.DataFrame(
        {
            "series_id": ["a", "b", "c", "d"],
            "store_id": ["s1", "s1", "s2", "s2"],
            "cat_id": ["food", "food", "hobby", "hobby"],
        }
    )
    revenue = pd.Series([1.0, 3.0, 2.0, 4.0], index=bottom["series_id"])
    selected = select_revenue_stratified_series(
        bottom, revenue, per_store_category=1
    )
    assert set(selected["series_id"]) == {"b", "d"}


def test_subset_rmsse_uses_first_nonzero_scale_and_revenue_weights() -> None:
    history = np.array([[0, 1, 3], [0, 2, 4]], dtype=float)
    scales = bottom_rmsse_scales(history)
    np.testing.assert_allclose(scales, [4.0, 4.0])
    score = revenue_weighted_subset_rmsse(
        np.array([[2.0, 4.0], [3.0, 5.0]]),
        np.array([[1.0, 3.0], [1.0, 3.0]]),
        scales=scales,
        revenue=np.array([1.0, 3.0]),
    )
    assert score == pytest.approx(0.875)
