"""Model discovery and file-layout checks.

The serving code resolves models from the flat `backend/ml/models/` layout
first, then the legacy per-category layout. These tests verify both the
artifact set and (when pandas/joblib are installed) the resolution logic.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "backend" / "ml" / "models"

sys.path.insert(0, str(REPO_ROOT / "backend"))


def _model_names():
    return {p.name for p in MODELS_DIR.glob("*.pkl")}


def test_models_dir_exists():
    assert MODELS_DIR.is_dir()


def test_11_lgbm_and_11_random_forest_models():
    names = _model_names()
    lgbm = {n for n in names if n.endswith("_LGBM_model.pkl")}
    rf = {n for n in names if n.endswith("_RandomForest_model.pkl")}
    assert len(lgbm) == 11, f"expected 11 LGBM models, found {len(lgbm)}: {sorted(lgbm)}"
    assert 11 <= len(rf) <= 12, (
        f"expected 11-12 RandomForest models, found {len(rf)}: {sorted(rf)}"
    )


def test_expected_categories_present():
    expected_prefixes = {
        "total_sales",
        "automobile_dealers",
        "building_material_and_garden_equipment",
        "clothing_and_clothing_accessories_stores",
        "electronics_and_appliance_stores",
        "food_and_beverage_stores",
        "furniture_and_home_furnishings_stores",
        "gasoline_stations",
        "general_merchandise_stores",
        "health_and_personal_care_stores",
        "sporting_goods_hobby_and_musical_instrument_stores",
    }
    names = _model_names()
    for prefix in expected_prefixes:
        assert f"{prefix}_LGBM_model.pkl" in names, f"missing LGBM model for {prefix}"


def test_flat_path_resolution_matches_disk():
    """The inference resolver must find the tracked flat-layout models."""
    pytest.importorskip("pandas")
    from ml.multi_resolution_inference import get_model_file_path

    path = get_model_file_path("total_sales", "LGBM")
    assert path.exists(), f"resolver returned a non-existent path: {path}"
    assert path.parent == MODELS_DIR


def test_resolver_falls_back_when_missing():
    pytest.importorskip("pandas")
    from ml import multi_resolution_inference as mri

    # a category that exists nowhere must resolve to the flat (preferred)
    # candidate path without existing on disk
    path = mri.get_model_file_path("nonexistent_category", "LGBM")
    assert not path.exists()
    assert path.parent == MODELS_DIR
