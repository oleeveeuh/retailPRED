from datetime import datetime, timezone

import pytest

from forecasting.data.mrts import (
    MRTSClient,
    MRTSResponseError,
    MissingAPIKeyError,
    parse_mrts_response,
)
from forecasting.registry import get_series


def _payload() -> list[list[str]]:
    return [
        [
            "data_type_code",
            "time_slot_id",
            "seasonally_adj",
            "category_code",
            "cell_value",
            "error_data",
            "time",
        ],
        ["SM", "1", "yes", "443", "7004", "", "2025"],
        ["SM", "2", "yes", "443", "7010", "", "2025"],
        ["SM", "1", "no", "443", "6549", "", "2025"],
        ["SM", "1", "yes", "444", "41000", "", "2025"],
        ["IM", "1", "yes", "443", "100", "", "2025"],
    ]


def test_parser_selects_native_monthly_sales_without_repetition() -> None:
    frame = parse_mrts_response(
        _payload(),
        get_series("electronics_and_appliances"),
        year=2025,
        vintage_time=datetime(2025, 3, 1, tzinfo=timezone.utc),
    )
    assert frame["observation_time"].dt.strftime("%Y-%m").tolist() == ["2025-01", "2025-02"]
    assert frame["value"].tolist() == [7004.0, 7010.0]
    assert frame[["series_id", "observation_time"]].duplicated().sum() == 0


def test_parser_rejects_missing_matching_rows() -> None:
    with pytest.raises(MRTSResponseError, match="no monthly sales rows"):
        parse_mrts_response(
            _payload(),
            get_series("retail_total"),
            year=2025,
            vintage_time=datetime(2025, 3, 1, tzinfo=timezone.utc),
        )


def test_client_never_falls_back_to_synthetic_data() -> None:
    with pytest.raises(MissingAPIKeyError, match="never substitutes synthetic"):
        MRTSClient(api_key=None)
