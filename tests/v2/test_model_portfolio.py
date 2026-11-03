import pytest

from forecasting.portfolio import MODEL_PORTFOLIO, model_decision


def test_model_portfolio_has_one_fail_closed_decision_per_dataset() -> None:
    assert {decision.dataset for decision in MODEL_PORTFOLIO} == {"mrts", "m5"}
    assert len({decision.dataset for decision in MODEL_PORTFOLIO}) == len(MODEL_PORTFOLIO)
    with pytest.raises(KeyError):
        model_decision("unknown")


def test_serving_contract_preserves_evaluation_limits() -> None:
    mrts = model_decision("mrts")
    assert mrts.point_status == "development_challenger"
    assert mrts.serving_mode == "unavailable_until_point_in_time_validation"
    assert mrts.uncertainty_status == "research_only_not_calibrated"

    m5 = model_decision("m5")
    assert m5.point_status == "promoted_point_champion"
    assert m5.serving_mode == "offline_demo_until_refit_on_current_data"
    assert m5.uncertainty_status == "research_only_failed_segment_gate"


def test_model_portfolio_claims_link_to_tracked_evidence() -> None:
    for decision in MODEL_PORTFOLIO:
        assert decision.evidence
        assert 0 < decision.measured_value < 1
        assert all(path.startswith("docs/v2/") for path in decision.evidence)
