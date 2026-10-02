"""Feature-schema validation for the tracked training CSVs.

Verifies the 73-feature contract: expected feature families exist, `year` is
present in the raw data but excluded by the training scripts.
"""

import csv
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = (
    REPO_ROOT / "project_root" / "data_multi_resolution" / "retail_total_sales_multi_resolution.csv"
)
TRAIN_SCRIPT = REPO_ROOT / "backend" / "ml" / "train_73_features.py"
TRAIN_SCRIPT_ROOT = REPO_ROOT / "train.py"


def _read_header():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        return next(csv.reader(f))


def test_csv_exists():
    assert CSV_PATH.exists(), f"missing training CSV: {CSV_PATH}"


def test_expected_feature_families_present():
    cols = _read_header()
    families = {
        "lag": 1,
        "rolling_mean": 3,
        "rolling_std": 3,
        "pct_change": 2,
        "diff": 2,
        "month_sin": 1,
        "momentum": 2,
        "yoy_change": 1,
    }
    for family, min_count in families.items():
        matching = [c for c in cols if c.startswith(family)]
        assert len(matching) >= min_count, (
            f"feature family '{family}': expected >= {min_count} columns, found {matching[:5]}"
        )


def test_year_column_exists_in_raw_csv():
    """`year` must exist in the raw CSV..."""
    assert "year" in _read_header(), "raw CSV should contain the year column"


def test_year_excluded_by_training_scripts():
    """...and be dropped by the training scripts (no leakage via calendar year)."""
    for script in (TRAIN_SCRIPT, TRAIN_SCRIPT_ROOT):
        source = script.read_text()
        assert re.search(
            r"exclude.*['\"]y['\"].*['\"]index['\"].*['\"]year['\"]", source, re.DOTALL
        ), f"{script.name} no longer excludes ['y', 'index', 'year']"


def test_target_and_index_columns_present():
    cols = _read_header()
    for col in ("y", "index", "year"):
        assert col in cols, f"expected column '{col}' in training CSV"
