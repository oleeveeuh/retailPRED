from scripts.diagnose_census import summarize_payload

PAYLOAD = [
    ["data_type_code", "time_slot_id", "seasonally_adj", "category_code", "cell_value"],
    ["SM", "1", "yes", "443", "7004"],
    ["SM", "2", "no", "443", "6549"],
    ["IM", "1", "yes", "44000", "99"],
]


def test_summary_reports_schema_without_cell_values() -> None:
    summary = summarize_payload(PAYLOAD)
    assert summary["shape"] == "list"
    assert summary["rows"] == 3
    assert summary["data_type_code"] == ["IM", "SM"]
    assert summary["seasonally_adj"] == ["no", "yes"]
    assert summary["category_code_count"] == 2
    serialized = str(summary)
    assert "7004" not in serialized
    assert "6549" not in serialized
    assert "99" not in serialized.split("time_slot_id")[0]


def test_summary_flags_unexpected_headers_and_shapes() -> None:
    assert summarize_payload([])["shape"] == "invalid"
    assert summarize_payload([["a", "b"]])["shape"] == "invalid"
    unknown = summarize_payload([["foo"], ["bar"]])
    assert unknown["shape"] == "unknown_header"
