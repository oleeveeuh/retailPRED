"""Explicit champion/challenger promotion policy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromotionDecision:
    promoted: bool
    relative_wape_improvement: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class PromotionPolicy:
    minimum_relative_wape_improvement: float = 0.02
    maximum_runtime_seconds: float = 900.0
    minimum_interval_coverage: float | None = None

    def evaluate(
        self,
        *,
        baseline_wape: float,
        challenger_wape: float,
        runtime_seconds: float,
        interval_coverage: float | None = None,
    ) -> PromotionDecision:
        if baseline_wape <= 0 or challenger_wape < 0:
            raise ValueError("WAPE values must be non-negative and baseline must be positive")
        if runtime_seconds < 0:
            raise ValueError("runtime_seconds cannot be negative")
        improvement = (baseline_wape - challenger_wape) / baseline_wape
        reasons: list[str] = []
        if improvement < self.minimum_relative_wape_improvement:
            reasons.append("minimum WAPE improvement not met")
        if runtime_seconds > self.maximum_runtime_seconds:
            reasons.append("runtime limit exceeded")
        if self.minimum_interval_coverage is not None:
            if interval_coverage is None:
                reasons.append("interval coverage was not reported")
            elif interval_coverage < self.minimum_interval_coverage:
                reasons.append("minimum interval coverage not met")
        return PromotionDecision(
            promoted=not reasons,
            relative_wape_improvement=improvement,
            reasons=tuple(reasons),
        )
