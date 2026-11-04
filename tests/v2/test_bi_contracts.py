import pandas as pd
import pytest

from bi.contracts import DATA_FRESHNESS


def _freshness() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": "m5",
                "use_case": "operational_demand",
                "evaluation_cutoff": "2016-05-22",
                "as_of_date": "2026-11-03",
                "serving_mode": "offline_demo_until_refit_on_current_data",
                "freshness_status": "historical_evaluation_only",
                "point_status": "promoted_point_champion",
                "uncertainty_status": "research_only_failed_segment_gate",
                "evidence": "docs/v2/M4_M5_LIGHTGBM.md",
            }
        ]
    )


def test_bi_contract_accepts_exact_schema_and_unique_key() -> None:
    assert DATA_FRESHNESS.validate(_freshness()).equals(_freshness())


def test_bi_contract_rejects_schema_drift_and_duplicate_keys() -> None:
    with pytest.raises(ValueError, match="columns"):
        DATA_FRESHNESS.validate(_freshness().assign(extra="drift"))
    with pytest.raises(ValueError, match="duplicate"):
        DATA_FRESHNESS.validate(pd.concat([_freshness(), _freshness()]))
