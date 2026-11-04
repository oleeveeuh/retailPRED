"""Regression fixtures for documented Census MRTS response schema variants.

The timeseries API historically returned seasonally_adj as yes/no while MRTS
workbooks and later API revisions use SA/NSA style markers. The parser must
accept every documented encoding, and its failure for unknown encodings must
report the observed values so the next diagnosis is self-contained.
"""

from datetime import datetime, timezone

import pytest

from forecasting.data.mrts import parse_mrts_response
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
