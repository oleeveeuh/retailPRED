from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
import shap

from forecasting.models.global_lightgbm import FEATURE_COLUMNS, GlobalLightGBM
from scripts.build_shap_analysis import build_shap_bundle

NOW = datetime(2026, 11, 20, tzinfo=timezone.utc)
REVISION = "a" * 40
DATA_SHA = "0" * 64


def _panel(periods: int = 36) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=periods, freq="MS")
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "series_id": series_id,
                    "observation_time": dates,
                    "value": base
                    + np.arange(periods) * trend
                    + 5 * np.sin(np.arange(periods) * 2 * np.pi / 12),
                }
            )
            for series_id, base, trend in (("a", 100, 1.0), ("b", 300, 2.0))
        ],
        ignore_index=True,
    )


def _model() -> GlobalLightGBM:
    return GlobalLightGBM(
        n_estimators=20, learning_rate=0.1, num_leaves=7, min_child_samples=3
    )


def _bundle(panel: pd.DataFrame) -> dict:
    return build_shap_bundle(
        panel, now=NOW, revision=REVISION, data_sha256=DATA_SHA, top_k=8
    )


def test_bundle_contract_pins_features_shares_and_provenance() -> None:
    payload = _bundle(_panel())
    provenance = payload["provenance"]
    assert payload["features"] == list(FEATURE_COLUMNS)
    assert provenance["model"] == "global_lightgbm"
    assert provenance["model_status"] == "development_challenger"
    assert provenance["explainer"].startswith("shap.TreeExplainer")
    assert provenance["target"] == "log(value / lag_12)"
    assert provenance["n_series"] == 2
    assert provenance["n_training_rows"] == len(_model().fit(_panel()).training_frame())
    assert provenance["data_cutoff"] == "2025-12-01"
    assert provenance["additivity_max_error"] < 1e-6

    shares = [row["share"] for row in payload["global_importance"]]
    assert len(payload["global_importance"]) == len(FEATURE_COLUMNS)
    assert abs(sum(shares) - 1.0) < 1e-9
    assert shares == sorted(shares, reverse=True)
    assert {row["feature"] for row in payload["global_importance"]} == set(FEATURE_COLUMNS)


def test_waterfalls_match_exact_shap_recomputation() -> None:
    panel = _panel()
    payload = _bundle(panel)

    # The bundle fit is deterministic, so refitting with the same default
    # constructor reproduces the identical booster and its exact SHAP values.
    model = GlobalLightGBM().fit(panel)
    frame = model.training_frame().reset_index(drop=True)
    design = frame.loc[:, FEATURE_COLUMNS]
    explanation = shap.TreeExplainer(model.point_model)(design)
    outputs = np.asarray(model.point_model.predict(design), dtype=float)
    np.testing.assert_allclose(
        np.asarray(explanation.base_values) + np.asarray(explanation.values).sum(axis=1),
        outputs,
        atol=1e-8,
    )

    assert len(payload["waterfalls"]) == 2
    assert {waterfall["series_id"] for waterfall in payload["waterfalls"]} == {"a", "b"}
    for waterfall in payload["waterfalls"]:
        rows = frame[frame["series_id"] == waterfall["series_id"]]
        index = rows["observation_time"].idxmax()
        assert waterfall["month"] == pd.Timestamp(rows.loc[index, "observation_time"]).date().isoformat()
        assert waterfall["model_output"] == pytest.approx(float(outputs[index]), abs=1e-9)
        assert waterfall["base_value"] == pytest.approx(float(explanation.base_values[index]), abs=1e-9)

        row_values = np.asarray(explanation.values)[index]
        top8 = {
            FEATURE_COLUMNS[position]
            for position in np.argsort(-np.abs(row_values))[:8]
        }
        assert {c["feature"] for c in waterfall["contributions"]} == top8
        assert all(
            c["shap"] == pytest.approx(float(row_values[FEATURE_COLUMNS.index(c["feature"])]), abs=1e-9)
            for c in waterfall["contributions"]
        )
        assert waterfall["month"] <= payload["provenance"]["data_cutoff"]


def test_training_frame_and_point_model_expose_the_fit_surface() -> None:
    model = _model().fit(_panel())
    frame = model.training_frame()
    assert {"series_id", "observation_time", "value"}.issubset(frame.columns)
    assert frame.loc[:, FEATURE_COLUMNS].notna().all().all()
    in_sample = model.point_model.predict(frame.loc[:, FEATURE_COLUMNS])
    assert len(in_sample) == len(frame)
    with pytest.raises(RuntimeError, match="fit must be called"):
        GlobalLightGBM().point_model
    with pytest.raises(RuntimeError, match="fit must be called"):
        GlobalLightGBM().training_frame()


def test_builder_refuses_additivity_violation(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenExplainer:
        def __init__(self, _model) -> None:
            pass

        def __call__(self, design):
            rows = len(design)

            class Explanation:
                values = np.zeros((rows, len(FEATURE_COLUMNS)))
                base_values = np.zeros(rows)

            return Explanation()

    class BrokenModel:
        def predict(self, design):
            return np.ones(len(design))

    real_fit = GlobalLightGBM.fit

    def fit_with_broken_explainer(self, history):
        real_fit(self, history)
        self._point_model = BrokenModel()
        return self

    monkeypatch.setattr(GlobalLightGBM, "fit", fit_with_broken_explainer)
    monkeypatch.setattr(shap, "TreeExplainer", BrokenExplainer)
    with pytest.raises(RuntimeError, match="additivity"):
        _bundle(_panel(15))
