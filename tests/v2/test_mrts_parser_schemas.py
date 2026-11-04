"""Regression fixtures for documented Census MRTS response schema variants.

The timeseries API historically returned seasonally_adj as yes/no while MRTS
workbooks and later API revisions use SA/NSA style markers. The parser must
accept every documented encoding, and its failure for unknown encodings must
report the observed values so the next diagnosis is self-contained.
"""

from datetime import datetime, timezone

import pandas as pd
import pytest

from forecasting.data.mrts import MRTS_FIELDS, MRTSClient, MRTSResponseError, parse_mrts_response
from forecasting.registry import get_series

HEADER = [
    "data_type_code",
    "time_slot_id",
    "seasonally_adj",
    "category_code",
    "cell_value",
    "error_data",
    "time",
]
VINTAGE = datetime(2026, 11, 15, 14, 35, tzinfo=timezone.utc)


def _payload(adjustment_marker: str, opposite_marker: str) -> list[list[str]]:
    return [
        HEADER,
        ["SM", "1", adjustment_marker, "443", "7004", "", "2025"],
        ["SM", "2", adjustment_marker, "443", "7010", "", "2025"],
        ["SM", "1", opposite_marker, "443", "6549", "", "2025"],
        ["IM", "1", adjustment_marker, "443", "100", "", "2025"],
    ]


@pytest.mark.parametrize(
    ("adjusted_marker", "opposite_marker"),
    [
        ("yes", "no"),
        ("y", "n"),
        ("true", "false"),
        ("SA", "NSA"),
    ],
    ids=["yes-no", "y-n", "true-false", "sa-nsa"],
)
def test_parser_accepts_every_documented_adjustment_encoding(
    adjusted_marker: str, opposite_marker: str
) -> None:
    frame = parse_mrts_response(
        _payload(adjusted_marker, opposite_marker),
        get_series("electronics_and_appliances"),
        year=2025,
        vintage_time=VINTAGE,
    )
    assert frame["value"].tolist() == [7004.0, 7010.0]


def test_parser_failure_reports_observed_schema() -> None:
    with pytest.raises(Exception) as excinfo:
        parse_mrts_response(
            _payload("adjusted", "not_adjusted"),
            get_series("electronics_and_appliances"),
            year=2025,
            vintage_time=VINTAGE,
        )
    message = str(excinfo.value)
    assert "observed data_type_code=['IM', 'SM']" in message
    assert "observed seasonally_adj=['adjusted', 'not_adjusted']" in message


CURRENT_SCHEMA_HEADER = [
    "data_type_code",
    "time_slot_id",
    "seasonally_adj",
    "category_code",
    "cell_value",
    "time",
]


def _current_schema_payload(time_values: tuple[str, str]) -> list[list[str]]:
    """Mirror the observed 2026 API schema: ordinal slots, period in `time`."""
    return [
        CURRENT_SCHEMA_HEADER,
        ["SM", "0", "yes", "443", "7004", time_values[0]],
        ["SM", "0", "yes", "443", "7010", time_values[1]],
        ["SM", "0", "no", "443", "6549", time_values[0]],
        ["IM", "0", "yes", "443", "100", time_values[0]],
    ]



@pytest.mark.parametrize(
    ("period_a", "period_b"),
    [("2010-01", "2010-02"), ("Jan-10", "Feb-10"), ("Jan-2010", "Feb-2010")],
    ids=["iso-month", "mon-yy", "mon-yyyy"],
)
def test_parser_derives_months_from_period_column_with_ordinal_slots(
    period_a: str, period_b: str
) -> None:
    frame = parse_mrts_response(
        _current_schema_payload((period_a, period_b)),
        get_series("electronics_and_appliances"),
        year=2010,
        vintage_time=VINTAGE,
    )
    assert frame["observation_time"].dt.strftime("%Y-%m").tolist() == ["2010-01", "2010-02"]
    assert frame["value"].tolist() == [7004.0, 7010.0]


def test_unparseable_periods_fail_with_time_samples() -> None:
    with pytest.raises(Exception) as excinfo:
        parse_mrts_response(
            _current_schema_payload(("garbage-a", "garbage-b")),
            get_series("electronics_and_appliances"),
            year=2010,
            vintage_time=VINTAGE,
        )
    message = str(excinfo.value)
    assert "observed time_slot_id=['0']" in message
    assert "time samples=['garbage-a'" in message


def test_period_year_mismatch_with_requested_year_is_dropped() -> None:
    # The API was asked for 2010 but returned 2011 periods: those rows must
    # not silently become 2010 observations.
    with pytest.raises(Exception) as excinfo:
        parse_mrts_response(
            _current_schema_payload(("2011-01", "2011-02")),
            get_series("electronics_and_appliances"),
            year=2010,
            vintage_time=VINTAGE,
        )
    assert "time samples=['2011-01'" in str(excinfo.value)


def test_get_field_list_matches_the_observed_working_schema() -> None:
    # Requesting a variable the API no longer serves (error_data was removed
    # live) returns an HTML page instead of JSON. Guard the field list.
    fields = MRTS_FIELDS.split(",")
    assert "error_data" not in fields
    assert set(fields) == {
        "data_type_code",
        "time_slot_id",
        "seasonally_adj",
        "category_code",
        "cell_value",
    }


class _HTMLResponse:
    status_code = 200
    content = b"<html><title>Bad Request</title></html>"
    headers = {"Content-Type": "text/html"}

    def json(self):
        import requests

        raise requests.exceptions.JSONDecodeError("Expecting value", "<html>", 0)


class _HTMLSession:
    def get(self, url, params=None, timeout=None):
        return _HTMLResponse()


def test_non_json_response_failure_names_the_schema_cause() -> None:
    client = MRTSClient(api_key="key", session=_HTMLSession())
    with pytest.raises(MRTSResponseError) as excinfo:
        client.fetch_year(get_series("retail_total"), year=2010)
    message = str(excinfo.value)
    assert "content-type=text/html" in message
    assert "GET field list" in message
