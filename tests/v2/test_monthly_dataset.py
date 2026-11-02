from datetime import datetime, timezone

import pandas as pd
import pytest

from forecasting.data.monthly_dataset import (
    observations_as_of,
    read_monthly_csv,
    validate_monthly_observations,
    write_monthly_csv,
)


def _observations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_id": ["retail_total", "retail_total", "retail_total"],
            "observation_time": ["2024-01-01", "2024-01-01", "2024-02-01"],
            "value": [100.0, 101.0, 110.0],
            "vintage_time": [
                "2024-02-01T00:00:00Z",
                "2024-03-01T00:00:00Z",
                "2024-03-01T00:00:00Z",
            ],
            "ingested_at": [
                "2024-02-01T00:00:01Z",
                "2024-03-01T00:00:01Z",
                "2024-03-01T00:00:01Z",
            ],
            "source": ["census_mrts"] * 3,
        }
    )


def test_as_of_selection_uses_latest_available_vintage_only() -> None:
    result = observations_as_of(
        _observations(), datetime(2024, 2, 15, tzinfo=timezone.utc)
    )
    assert len(result) == 1
    assert result.iloc[0]["value"] == 100


def test_monthly_csv_round_trip_preserves_contract(tmp_path) -> None:
    path = tmp_path / "monthly.csv"
    write_monthly_csv(_observations(), path)
    result = read_monthly_csv(path)
    assert list(result.columns) == [
        "series_id",
        "observation_time",
        "value",
        "release_time",
        "vintage_time",
        "ingested_at",
        "source",
    ]
    assert len(result) == 3


def test_contract_rejects_non_month_start_and_duplicate_vintage_keys() -> None:
    invalid_date = _observations()
    invalid_date.loc[0, "observation_time"] = "2024-01-15"
    with pytest.raises(ValueError, match="first day"):
        validate_monthly_observations(invalid_date)

    invalid_time = _observations()
    invalid_time.loc[0, "observation_time"] = "2024-01-01T12:00:00"
    with pytest.raises(ValueError, match="midnight"):
        validate_monthly_observations(invalid_time)

    duplicate = pd.concat([_observations(), _observations().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_monthly_observations(duplicate)


def test_contract_rejects_malformed_release_timestamp() -> None:
    malformed = _observations().assign(release_time="not-a-timestamp")
    with pytest.raises(ValueError, match="release_time contains an invalid"):
        validate_monthly_observations(malformed)
