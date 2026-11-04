#!/usr/bin/env python3
"""Capture Census MRTS API response schemas per year without exposing the key.

Prints schema-level information only: row counts and the distinct values of
the filtering columns. Cell values, the key, and full URLs are never printed.
Run this when ingestion fails so the observed schema can be compared against
the parser's documented expectations.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import requests

MRTS_API_URL = "https://api.census.gov/data/timeseries/eits/mrts"
MRTS_FIELDS = "data_type_code,time_slot_id,seasonally_adj,category_code,cell_value"


def summarize_payload(payload: Any) -> dict[str, object]:
    """Schema-level summary of a Census JSON payload (no cell values)."""
    if not isinstance(payload, list) or len(payload) < 2:
        return {"shape": "invalid", "detail": "payload is not a header+rows list"}
    header, *rows = payload
    if "data_type_code" not in header or "seasonally_adj" not in header:
        return {"shape": "unknown_header", "header": list(header)}
    index = {column: position for position, column in enumerate(header)}
    adjustments = sorted({str(row[index["seasonally_adj"]]) for row in rows})
    codes = sorted({str(row[index["data_type_code"]]) for row in rows})
    categories = sorted({str(row[index["category_code"]]) for row in rows})
    slots = sorted({str(row[index["time_slot_id"]]) for row in rows})
    return {
        "shape": "list",
        "rows": len(rows),
        "header": list(header),
        "data_type_code": codes,
        "seasonally_adj": adjustments,
        "category_code_count": len(categories),
        "time_slot_id": slots,
    }


def probe_year(session: requests.Session, api_key: str, year: int) -> dict[str, object]:
    response = session.get(
        MRTS_API_URL,
        params={"get": MRTS_FIELDS, "time": str(year), "key": api_key},
        timeout=30,
    )
    content_type = response.headers.get("Content-Type", "")
    result: dict[str, object] = {"year": year, "http_status": response.status_code, "content_type": content_type}
    if "json" in content_type:
        try:
            payload = response.json()
        except requests.JSONDecodeError:
            result["schema"] = {"shape": "non-json-decodable"}
        else:
            result["schema"] = summarize_payload(payload)
    else:
        # Never print raw bodies: they may echo parameters or contain markup.
        result["schema"] = {"shape": "non-json", "bytes": len(response.content)}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2010, 2015, 2020, 2024, 2025],
    )
    args = parser.parse_args()
    api_key = os.getenv("CENSUS_API_KEY")
    if not api_key:
        print("CENSUS_API_KEY must be set in the environment", file=sys.stderr)
        return 1
    session = requests.Session()
    for year in args.years:
        summary = probe_year(session, api_key, year)
        print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
