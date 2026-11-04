import numpy as np
import pandas as pd
import pytest

from forecasting.reconciliation.m5 import (
    M5Hierarchy,
    M5UpperHierarchy,
    M5_LEVELS,
    WeightedLeastSquaresReconciler,
    m5_revenue_weights,
    reconcile_m5_upper_forecasts,
    weighted_rmsse,
)


def _bottom() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("item_a_CA_1", "item_a", "dept_1", "cat_1", "CA_1", "CA"),
            ("item_b_CA_1", "item_b", "dept_2", "cat_2", "CA_1", "CA"),
            ("item_a_TX_1", "item_a", "dept_1", "cat_1", "TX_1", "TX"),
            ("item_b_TX_1", "item_b", "dept_2", "cat_2", "TX_1", "TX"),
        ],
        columns=["series_id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
    )


def _bottom_forecasts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_id": _bottom()["series_id"],
            "target_time": pd.Timestamp("2026-01-01"),
            "prediction": [1.0, 2.0, 3.0, 4.0],
        }
    )


def test_sparse_hierarchy_contains_all_twelve_levels() -> None:
    hierarchy = M5Hierarchy.from_bottom(_bottom())
    assert set(hierarchy.nodes["level"]) == set(range(1, 13))
    assert hierarchy.summing_matrix.shape == (len(hierarchy.nodes), 4)
    assert hierarchy.summing_matrix.nnz == len(_bottom()) * len(M5_LEVELS)
    assert hierarchy.nodes.loc[hierarchy.nodes["level"] == 1, "member_count"].item() == 4


def test_bottom_up_forecasts_are_exactly_coherent() -> None:
    hierarchy = M5Hierarchy.from_bottom(_bottom())
    coherent = hierarchy.aggregate_bottom(_bottom_forecasts())
    total = coherent.loc[coherent["level"] == 1, "prediction"].item()
    assert total == 10
    assert hierarchy.coherence_error(coherent) == 0


def test_weighted_reconciliation_projects_incoherent_nonnegative_forecasts() -> None:
    hierarchy = M5Hierarchy.from_bottom(_bottom())
    base = hierarchy.aggregate_bottom(_bottom_forecasts())
    base.loc[base["level"] == 1, "prediction"] = 30.0
    assert hierarchy.coherence_error(base) == 20.0
    variances = pd.Series(1.0, index=hierarchy.nodes["node_id"])
    reconciled = WeightedLeastSquaresReconciler(hierarchy).reconcile(base, variances)
    assert hierarchy.coherence_error(reconciled) < 1e-8
    assert (reconciled["prediction"] >= 0).all()


def test_upper_reconciliation_scales_bottom_to_coherent_store_departments() -> None:
    hierarchy = M5Hierarchy.from_bottom(_bottom())
    upper = M5UpperHierarchy.from_hierarchy(hierarchy)
    bottom = np.array([[1.0], [2.0], [3.0], [4.0]])
    base = upper.summing_matrix @ np.array([[1.5], [2.5], [3.5], [4.5]])
    base[0, 0] = 30.0
    reconciled = upper.reconcile(
        base, pd.Series(1.0, index=upper.nodes["node_id"])
    )
    full = upper.disaggregate_to_bottom(reconciled, bottom)
    np.testing.assert_allclose(
        full[: len(upper.nodes)], reconciled, rtol=1e-8, atol=1e-8
    )
    assert (full >= 0).all()


def test_validation_weighted_upper_reconciliation_returns_full_coherence() -> None:
    hierarchy = M5Hierarchy.from_bottom(_bottom())
    upper = M5UpperHierarchy.from_hierarchy(hierarchy)
    bottom_validation = np.array([[1.0], [2.0], [3.0], [4.0]])
    bottom_evaluation = np.array([[2.0], [3.0], [4.0], [5.0]])
    upper_actual = (upper.summing_matrix @ bottom_validation) * 0.95
    aggregate_validation = upper_actual * 1.05
    aggregate_evaluation = (upper.summing_matrix @ bottom_evaluation) * 1.05
    result = reconcile_m5_upper_forecasts(
        upper,
        bottom_validation_forecast=bottom_validation,
        bottom_evaluation_forecast=bottom_evaluation,
        aggregate_validation_forecast=aggregate_validation,
        aggregate_evaluation_forecast=aggregate_evaluation,
        upper_validation_actual=upper_actual,
        fallback_shares=np.ones(4),
    )
    assert result.base_coherence_max_abs > 0
    assert result.reconciled_coherence_max_abs < 1e-8
    assert result.reconciled.shape == (len(hierarchy.nodes), 1)


def test_m5_revenue_weights_allocate_one_twelfth_to_each_level() -> None:
    hierarchy = M5Hierarchy.from_bottom(_bottom())
    revenue = pd.Series(
        [10.0, 20.0, 30.0, 40.0], index=hierarchy.bottom["series_id"]
    )
    weights = m5_revenue_weights(hierarchy, revenue)
    assert weights.sum() == pytest.approx(1.0)
    by_level = hierarchy.nodes.assign(weight=weights.to_numpy()).groupby("level")["weight"].sum()
    np.testing.assert_allclose(by_level, np.repeat(1 / 12, 12))


def test_official_shape_validation_rejects_fixture() -> None:
    with pytest.raises(ValueError, match="30,490"):
        M5Hierarchy.from_bottom(_bottom()).validate_official_shape()


def test_wrmsse_uses_fixed_training_scales_and_revenue_weights() -> None:
    evaluation = pd.DataFrame(
        {
            "node_id": ["a", "a", "b", "b"],
            "target_time": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-01", "2026-01-02"]
            ),
            "actual": [10.0, 12.0, 20.0, 22.0],
            "prediction": [11.0, 13.0, 18.0, 20.0],
        }
    )
    score = weighted_rmsse(
        evaluation,
        scales=pd.Series({"a": 1.0, "b": 4.0}),
        weights=pd.Series({"a": 0.25, "b": 0.75}),
    )
    assert score == pytest.approx(1.0)
