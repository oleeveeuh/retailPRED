import numpy as np
import pytest

from forecasting.evaluation.metrics import (
    bias,
    interval_coverage,
    interval_width,
    mae,
    pinball_loss,
    rmsse,
    wape,
    weighted_interval_score,
)


def test_point_metrics_have_declared_units_and_direction() -> None:
    actual = [10, 20]
    prediction = [12, 18]
    assert mae(actual, prediction) == 2
    assert wape(actual, prediction) == pytest.approx(4 / 30)
    assert bias(actual, prediction) == 0
    assert rmsse(actual, prediction, training=[1, 2, 3, 4], season_length=1) == 2


def test_probabilistic_metrics() -> None:
    actual = np.array([10, 20])
    assert pinball_loss(actual, [8, 22], 0.5) == 1
    assert interval_coverage(actual, [8, 18], [12, 22]) == 1
    assert interval_width([8, 18], [12, 22]) == 4
    assert weighted_interval_score(actual, actual, [(0.2, [8, 18], [12, 22])]) == pytest.approx(0.2666666667)


def test_wape_rejects_zero_denominator() -> None:
    with pytest.raises(ValueError, match="undefined"):
        wape([0, 0], [1, 1])
