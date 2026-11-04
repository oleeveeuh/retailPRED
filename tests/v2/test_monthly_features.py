import numpy as np
import pandas as pd
import pytest

from forecasting.features.monthly import build_causal_features, validate_monthly_panel


def _panel(periods: int = 18) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_id": "retail_total",
            "observation_time": pd.date_range("2020-01-01", periods=periods, freq="MS"),
            "value": np.arange(1, periods + 1, dtype=float),
        }
    )


def test_current_and_future_targets_cannot_change_features_at_origin() -> None:
    base = _panel()
    changed = base.copy()
    changed.loc[changed.index >= 12, "value"] = 999_999
    columns = ["lag_1", "lag_12", "rolling_mean_3", "rolling_std_3", "diff_1", "pct_change_1", "month"]
    original_features = build_causal_features(base).loc[12, columns]
    changed_features = build_causal_features(changed).loc[12, columns]
    pd.testing.assert_series_equal(original_features, changed_features)


def test_rolling_features_shift_before_aggregating() -> None:
    features = build_causal_features(_panel())
    row = features.loc[3]
    assert row["lag_1"] == 3
    assert row["rolling_mean_3"] == 2
    assert row["diff_1"] == 1


def test_monthly_panel_rejects_repeated_and_missing_targets() -> None:
    duplicate = pd.concat([_panel(), _panel().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_monthly_panel(duplicate)
    with pytest.raises(ValueError, match="gaps"):
        validate_monthly_panel(_panel().drop(index=5))
