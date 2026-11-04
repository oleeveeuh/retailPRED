import numpy as np
import pytest

from forecasting.evaluation.m5_probabilistic import (
    central_interval_metrics,
    finite_sample_quantile,
    scaled_absolute_errors,
)


def test_finite_sample_quantile_uses_conservative_rank() -> None:
    assert finite_sample_quantile([1, 2, 3, 4], coverage=0.8) == 4
    assert finite_sample_quantile(np.arange(1, 10), coverage=0.8) == 8


def test_scaled_errors_use_node_rmsse_denominator() -> None:
    result = scaled_absolute_errors(
        [[2.0, 4.0], [4.0, 8.0]],
        [[1.0, 2.0], [2.0, 4.0]],
        [1.0, 4.0],
    )
    np.testing.assert_allclose(result, [[1.0, 2.0], [1.0, 2.0]])


def test_central_interval_metrics_support_official_node_weights() -> None:
    result = central_interval_metrics(
        [[1.0, 3.0], [2.0, 4.0]],
        [[1.0, 2.0], [2.0, 3.0]],
        [[0.0, 1.0], [1.0, 2.0]],
        [[2.0, 3.0], [3.0, 4.0]],
        [1.0, 4.0],
        weights=[0.25, 0.75],
    )
    assert result["coverage"] == 1.0
    assert result["mean_scaled_width"] == pytest.approx(1.25)
    assert result["normalized_wis"] > 0
