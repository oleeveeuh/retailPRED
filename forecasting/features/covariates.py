"""Availability roles that prevent covariates from crossing forecast origins."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class CovariateRole(StrEnum):
    HISTORICAL = "historical"
    KNOWN_FUTURE = "known_future"
    STATIC = "static"
    UNAVAILABLE_AT_ORIGIN = "unavailable_at_origin"


@dataclass(frozen=True)
class CovariateSpec:
    name: str
    role: CovariateRole
    description: str


MONTHLY_COVARIATES = (
    CovariateSpec("lag_*", CovariateRole.HISTORICAL, "Lagged target values."),
    CovariateSpec(
        "rolling_*", CovariateRole.HISTORICAL, "Target summaries shifted by one month."
    ),
    CovariateSpec("month", CovariateRole.KNOWN_FUTURE, "Calendar month."),
    CovariateSpec("series_id", CovariateRole.STATIC, "Stable retail-series identity."),
    CovariateSpec(
        "unreleased_economic_value",
        CovariateRole.UNAVAILABLE_AT_ORIGIN,
        "Economic observations whose release time follows the forecast origin.",
    ),
)


def validate_covariate_roles(specs: tuple[CovariateSpec, ...]) -> None:
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("covariate names must be unique")
