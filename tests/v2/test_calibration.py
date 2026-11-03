import pandas as pd
import pytest

from forecasting.evaluation.calibration import (
    ConformalIntervalCalibrator,
    enforce_non_crossing_quantiles,
    temporal_calibration_split,
)
from forecasting.evaluation.metrics import interval_coverage


def test_non_crossing_rearrangement_sorts_each_forecast() -> None:
    frame = pd.DataFrame({"q_10": [12.0], "q_50": [8.0], "q_90": [10.0]})
    result = enforce_non_crossing_quantiles(frame, ("q_10", "q_50", "q_90"))
    assert result.loc[0, ["q_10", "q_50", "q_90"]].tolist() == [8.0, 10.0, 12.0]


def test_conformal_calibration_uses_horizon_specific_errors() -> None:
    calibration = pd.DataFrame(
        {
            "horizon": [1, 1, 3, 3],
            "actual": [10, 20, 30, 40],
            "q_10": [9, 18, 25, 35],
            "q_90": [11, 19, 32, 42],
        }
    )
    calibrator = ConformalIntervalCalibrator(alpha=0.2, method="additive").fit(
        calibration, lower_column="q_10", upper_column="q_90"
    )
    evaluation = pd.DataFrame(
        {"horizon": [1, 3], "actual": [20, 40], "q_10": [18, 35], "q_90": [19, 42]}
    )
    result = calibrator.transform(
        evaluation, lower_column="q_10", upper_column="q_90"
    )
    assert interval_coverage(
        result["actual"], result["lower_calibrated"], result["upper_calibrated"]
    ) == 1
    assert calibrator.adjustments[(1,)] != calibrator.adjustments[(3,)]


def test_temporal_split_is_disjoint_by_target_time() -> None:
    frame = pd.DataFrame(
        {"target_time": pd.date_range("2021-12-01", periods=3, freq="MS")}
    )
    calibration, evaluation = temporal_calibration_split(
        frame, calibration_end="2021-12-31", evaluation_start="2022-01-01"
    )
    assert calibration["target_time"].max() < evaluation["target_time"].min()
    with pytest.raises(ValueError, match="precede"):
        temporal_calibration_split(
            frame, calibration_end="2022-01-01", evaluation_start="2022-01-01"
        )
