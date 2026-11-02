#!/usr/bin/env python3
"""Load one saved Census MRTS extract through local bronze, silver, and gold."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from retailpred.platform.gold import build_gold_monthly
from retailpred.platform.local_store import LocalTableStore
from retailpred.platform.mrts_pipeline import ExtractContext, ingest_extract
from retailpred.platform.specs import GOLD_MONTHLY, SILVER_MONTHLY


def _aware_timestamp(value: str) -> datetime:
    result = datetime.fromisoformat(value)
    if result.tzinfo is None or result.utcoffset() is None:
        raise argparse.ArgumentTypeError("timestamp must include a UTC offset")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("payload", type=Path, help="Saved Census API JSON response")
    parser.add_argument("--extract-id", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--vintage-time", required=True, type=_aware_timestamp)
    parser.add_argument("--ingested-at", required=True, type=_aware_timestamp)
    parser.add_argument("--release-time", type=_aware_timestamp)
    parser.add_argument("--as-of", type=_aware_timestamp)
    parser.add_argument("--root", type=Path, default=Path("data/v2/lakehouse"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.payload.read_text())
    context = ExtractContext(
        extract_id=args.extract_id,
        source_url=args.source_url,
        release_time=args.release_time,
        vintage_time=args.vintage_time,
        ingested_at=args.ingested_at,
    )
    store = LocalTableStore(args.root)
    bronze, silver = ingest_extract(store, payload, context)
    as_of = args.as_of or args.vintage_time
    gold = build_gold_monthly(store.read(SILVER_MONTHLY), as_of=as_of)
    store.replace_snapshot(GOLD_MONTHLY, gold)
    print(
        f"Loaded extract {args.extract_id}: "
        f"bronze={len(bronze):,}, silver={len(silver):,}, gold={len(gold):,}"
    )


if __name__ == "__main__":
    main()
