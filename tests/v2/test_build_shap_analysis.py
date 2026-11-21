import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import shap

from forecasting.models.global_lightgbm import FEATURE_COLUMNS, GlobalLightGBM
from scripts.build_shap_analysis import (
    ADDITIVITY_TOLERANCE,
    build_shap_bundle,
    build_shift_attribution,
    build_shift_diagnosis,
    reference_metrics,
)

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

REPO_ROOT = Path(__file__).resolve().parents[2]
DEV_FOLDS = REPO_ROOT / "data/v2/mrts_model_folds.csv"
GATE_FOLDS = REPO_ROOT / "data/v2/vintage_backtest_folds.csv"
GATE_SUMMARY = REPO_ROOT / "data/v2/vintage_backtest_summary.json"
VINTAGE_CSV = REPO_ROOT / "data/v2/alfred_mrts_vintages.csv"


def _synthetic_dev_folds() -> pd.DataFrame:
    rows = []
    for series_id, actual, naive_err, lgbm_err in (("a", 100.0, 10.0, 5.0), ("b", 200.0, 40.0, 30.0)):
        for model_name, err in (("seasonal_naive", naive_err), ("global_lightgbm", lgbm_err)):
            rows.append(
                {
                    "series_id": series_id,
                    "model_name": model_name,
                    "prediction": actual - err,
                    "actual": actual,
                }
            )
    return pd.DataFrame(rows)


def _synthetic_gate_summary() -> dict:
    return {
        "global_lightgbm": {"by_series": {"a": {"wape": 0.10}, "b": {"wape": 0.14}}},
        "seasonal_naive": {"by_series": {"a": {"wape": 0.08}, "b": {"wape": 0.12}}},
    }


def test_reference_metrics_reproduce_the_fold_weighted_recipe() -> None:
    references = reference_metrics(_synthetic_dev_folds(), _synthetic_gate_summary())
    validation = references["development_validation"]
    # naive: (0.1 + 0.2) / 2, lightgbm: (0.05 + 0.15) / 2
    assert validation["naive_mean_category_wape"] == pytest.approx(0.15)
    assert validation["lightgbm_mean_category_wape"] == pytest.approx(0.10)
    assert validation["relative_improvement"] == pytest.approx(1 / 3)
    gate = references["vintage_gate"]
    assert gate["naive_mean_category_wape"] == pytest.approx(0.10)
    assert gate["lightgbm_mean_category_wape"] == pytest.approx(0.12)
    assert gate["relative_degradation"] == pytest.approx(0.20)


def test_shift_references_match_the_documented_measured_endpoints() -> None:
    references = reference_metrics(
        pd.read_csv(DEV_FOLDS), json.loads(GATE_SUMMARY.read_text())
    )
    # docs/v2/M3_LIGHTGBM.md: "equally weighted mean category WAPE by 29.2%"
    assert references["development_validation"]["relative_improvement"] == pytest.approx(
        0.2921, abs=5e-4
    )
    # docs/v2/M3_VINTAGE_BACKTEST.md: 8.49% vs 8.16% -> ~4.1% degradation, gate FAILED
    assert references["vintage_gate"]["relative_degradation"] == pytest.approx(0.0408, abs=2e-4)


def _synthetic_vintage_frame(periods: int = 30) -> pd.DataFrame:
    """Monthly vintages that keep revising toward the truth, like ALFRED.

    Thirteen monthly vintages per month shrink a +12% first-release bias
    toward zero, and a final revision (the exact value) lands at +420 days.
    An as-of near the cutoff therefore reads recent months from biased early
    vintages while old months have converged — the revised panel reads the
    final revision everywhere.
    """
    dates = pd.date_range("2023-01-01", periods=periods, freq="MS")
    rows = []
    for series_id, base, trend in (("a", 100.0, 1.0), ("b", 300.0, 2.0)):
        for index, month in enumerate(dates):
            value = base + index * trend + 5 * np.sin(index * 2 * np.pi / 12)
            for step in range(13):
                bias = 0.12 / (step + 1)
                vintage_time = pd.Timestamp(month) + pd.Timedelta(days=15 + 30 * step)
                rows.append(
                    {
                        "series_id": series_id,
                        "observation_time": month,
                        "value": value * (1.0 + bias),
                        "release_time": pd.NaT,
                        "vintage_time": vintage_time.tz_localize("UTC"),
                        "ingested_at": vintage_time.tz_localize("UTC"),
                        "source": "synthetic",
                    }
                )
            final_vintage = pd.Timestamp(month) + pd.Timedelta(days=420)
            rows.append(
                {
                    "series_id": series_id,
                    "observation_time": month,
                    "value": value,
                    "release_time": pd.NaT,
                    "vintage_time": final_vintage.tz_localize("UTC"),
                    "ingested_at": final_vintage.tz_localize("UTC"),
                    "source": "synthetic",
                }
            )
    return pd.DataFrame(rows)


def test_shift_attribution_moves_when_recent_months_stay_unrevised() -> None:
    frame = _synthetic_vintage_frame()
    cutoff = pd.Timestamp("2025-06-01")
    as_of = datetime(2025, 6, 21, tzinfo=timezone.utc)
    result = build_shift_attribution(frame, cutoff=cutoff, as_of=as_of)
    assert len(result["rows"]) == len(FEATURE_COLUMNS)
    drifts = [abs(row["drift_abs"]) for row in result["rows"]]
    assert drifts == sorted(drifts, reverse=True)
    assert result["attribution_drift_l1"] > 0
    fits = result.pop("fits")
    assert fits["revised_history"]["cutoff"] == fits["point_in_time"]["cutoff"] == "2025-06-01"
    assert fits["revised_history"]["residual"] < ADDITIVITY_TOLERANCE
    assert fits["point_in_time"]["residual"] < ADDITIVITY_TOLERANCE


def test_shift_diagnosis_on_the_frozen_gate_artifacts() -> None:
    import hashlib

    digest = hashlib.sha256(VINTAGE_CSV.read_bytes()).hexdigest()
    window_summary = json.loads(
        (REPO_ROOT / "data/v2/evidence/mrts_research_windows/mrts_window_summary.json").read_text()
    )
    assert digest == window_summary["provenance"]["data_sha256"], (
        "the diagnosis must run on the exact frozen panel the window summary pins"
    )
    payload = build_shift_diagnosis(
        pd.read_csv(VINTAGE_CSV),
        pd.read_csv(GATE_FOLDS),
        pd.read_csv(DEV_FOLDS),
        json.loads(GATE_SUMMARY.read_text()),
        now=NOW,
        revision=REVISION,
        data_sha256=digest,
    )
    provenance = payload["provenance"]
    assert pd.Timestamp(provenance["as_of"]) == pd.Timestamp("2025-05-15T00:00:00+00:00")
    assert provenance["common_cutoff"] == "2025-03-01"
    for fit in (provenance["revised_history"], provenance["point_in_time"]):
        assert fit["data_cutoff"] == "2025-03-01"
        assert fit["n_series"] == 11
        assert fit["additivity_max_error"] < ADDITIVITY_TOLERANCE
    assert provenance["revised_history"]["n_rows"] == provenance["point_in_time"]["n_rows"]
    assert payload["rows"] and len(payload["rows"]) == len(FEATURE_COLUMNS)
    drifts = [abs(row["drift_abs"]) for row in payload["rows"]]
    assert drifts == sorted(drifts, reverse=True)
    assert payload["attribution_drift_l1"] > 0
