"""Frozen model-selection decisions handed from evaluation to serving."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelPortfolioDecision:
    use_case: str
    dataset: str
    point_model: str
    point_status: str
    serving_mode: str
    uncertainty_status: str
    evaluation_cutoff: str
    primary_metric: str
    measured_value: float
    evidence: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


MODEL_PORTFOLIO: tuple[ModelPortfolioDecision, ...] = (
    ModelPortfolioDecision(
        use_case="macro_planning",
        dataset="mrts",
        point_model="global_lightgbm",
        point_status="development_challenger",
        serving_mode="unavailable_until_point_in_time_validation",
        uncertainty_status="research_only_not_calibrated",
        evaluation_cutoff="2025-12-01",
        primary_metric="wape",
        measured_value=0.03154680188576103,
        evidence=(
            "docs/v2/M3_LIGHTGBM.md",
            "docs/v2/M3_PROBABILISTIC.md",
            "docs/v2/M3_MRTS_REGIMES.md",
        ),
    ),
    ModelPortfolioDecision(
        use_case="operational_demand",
        dataset="m5",
        point_model="m5_global_lightgbm",
        point_status="promoted_point_champion",
        serving_mode="offline_demo_until_refit_on_current_data",
        uncertainty_status="research_only_failed_segment_gate",
        evaluation_cutoff="2016-05-22",
        primary_metric="wrmsse",
        measured_value=0.813300679426698,
        evidence=(
            "docs/v2/M4_M5_LIGHTGBM.md",
            "docs/v2/M4_M5_STABILITY.md",
            "docs/v2/M4_M5_PROBABILISTIC.md",
        ),
    ),
)


def model_decision(dataset: str) -> ModelPortfolioDecision:
    """Return one frozen decision, failing closed for unknown datasets."""
    matches = [decision for decision in MODEL_PORTFOLIO if decision.dataset == dataset]
    if len(matches) != 1:
        raise KeyError(f"no unique model decision for dataset {dataset!r}")
    return matches[0]
