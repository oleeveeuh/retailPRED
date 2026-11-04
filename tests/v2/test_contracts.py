from datetime import date, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from forecasting.contracts import Forecast, Observation


NOW = datetime(2025, 1, 15, tzinfo=timezone.utc)


def test_observation_requires_timezone_aware_provenance() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        Observation(
            series_id="retail_total",
            observation_time=date(2024, 12, 1),
            value=600_000,
            vintage_time=datetime(2025, 1, 15),
            ingested_at=NOW,
            source="census_mrts",
        )


def test_observation_rejects_ingestion_before_vintage() -> None:
    with pytest.raises(ValidationError, match="cannot precede"):
        Observation(
            series_id="retail_total",
            observation_time=date(2024, 12, 1),
            value=600_000,
            vintage_time=NOW,
            ingested_at=NOW - timedelta(seconds=1),
            source="census_mrts",
        )


def test_forecast_rejects_training_after_origin() -> None:
    with pytest.raises(ValidationError, match="training_cutoff"):
        Forecast(
            series_id="retail_total",
            forecast_origin=date(2025, 1, 1),
            target_time=date(2025, 2, 1),
            horizon=1,
            training_cutoff=date(2025, 2, 1),
            data_vintage=NOW,
            model_name="seasonal_naive",
            model_version="2.0.0",
            prediction=1.0,
            actual_source="census_mrts",
        )
