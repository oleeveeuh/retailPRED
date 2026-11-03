import pytest

from forecasting.features.covariates import (
    M5_COVARIATES,
    MONTHLY_COVARIATES,
    CovariateRole,
    CovariateSpec,
    validate_covariate_roles,
)


def test_monthly_covariates_declare_availability_roles() -> None:
    validate_covariate_roles(MONTHLY_COVARIATES)
    assert {spec.role for spec in MONTHLY_COVARIATES} == set(CovariateRole)


def test_m5_covariates_declare_availability_roles() -> None:
    validate_covariate_roles(M5_COVARIATES)
    assert {spec.role for spec in M5_COVARIATES} == set(CovariateRole)


def test_covariate_names_must_be_unique() -> None:
    duplicate = CovariateSpec("month", CovariateRole.KNOWN_FUTURE, "duplicate")
    with pytest.raises(ValueError, match="unique"):
        validate_covariate_roles(MONTHLY_COVARIATES + (duplicate,))
