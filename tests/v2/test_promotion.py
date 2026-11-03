from forecasting.promotion import PromotionPolicy


def test_promotion_requires_declared_improvement_and_runtime() -> None:
    policy = PromotionPolicy(
        minimum_relative_wape_improvement=0.05,
        maximum_runtime_seconds=60,
    )
    accepted = policy.evaluate(
        baseline_wape=0.10,
        challenger_wape=0.09,
        runtime_seconds=30,
    )
    rejected = policy.evaluate(
        baseline_wape=0.10,
        challenger_wape=0.098,
        runtime_seconds=90,
    )
    assert accepted.promoted
    assert not rejected.promoted
    assert set(rejected.reasons) == {
        "minimum WAPE improvement not met",
        "runtime limit exceeded",
    }
