"""Parsing and internal-consistency checks for the authoritative metrics file.

`training_outputs/validation_metrics.json` backs the README results table.
If its format or numbers drift, these tests should catch it.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
METRICS_PATH = REPO_ROOT / "training_outputs" / "validation_metrics.json"


def _load():
    return json.loads(METRICS_PATH.read_text())


def test_metrics_file_exists_and_parses():
    data = _load()
    assert data["total_models"] == len(data["models"]) == 6
    assert "generated_at" in data
    assert "description" in data


def test_expected_models_present():
    models = _load()["models"]
    assert set(models.keys()) == {
        "Lgbm", "Randomforest", "Seasonalnaive", "Patchtst", "Timesnet", "Autoarima"
    }


def test_avg_mape_reconciles_with_breakdown():
    """Reported avg_mape must equal the sample-weighted mean of the per-category rows."""
    for name, model in _load()["models"].items():
        breakdown = model["breakdown"]
        total_samples = sum(c["sample_count"] for c in breakdown)
        assert total_samples == model["total_samples"], (
            f"{name}: breakdown sample counts do not sum to total_samples"
        )
        weighted = (
            sum(c["avg_mape"] * c["sample_count"] for c in breakdown) / total_samples
        )
        assert abs(weighted - model["avg_mape"]) < 0.01, (
            f"{name}: avg_mape {model['avg_mape']} != weighted breakdown mean {weighted:.2f}"
        )
        assert len(breakdown) == model["categories"], (
            f"{name}: categories field != len(breakdown)"
        )


def test_documented_readme_numbers_match():
    """The README table quotes these values; fail if the metrics file changes."""
    models = _load()["models"]
    expected = {
        "Lgbm": 9.76,
        "Randomfarest": None,  # guard against typo'd keys below
        "Randomforest": 10.06,
        "Seasonalnaive": 14.11,
        "Patchtst": 17.89,
        "Timesnet": 18.61,
        "Autoarima": 34.75,
    }
    for name, value in expected.items():
        if value is None:
            continue
        assert abs(models[name]["avg_mape"] - value) < 0.01, (
            f"{name} avg_mape changed ({models[name]['avg_mape']}); update README table"
        )


def test_lgbm_has_all_11_categories():
    lgbm = _load()["models"]["Lgbm"]
    assert lgbm["categories"] == 11
    assert lgbm["total_samples"] == 429
