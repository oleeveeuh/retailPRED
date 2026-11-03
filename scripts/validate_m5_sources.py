"""Validate licensed M5 files and persist reproducible local evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from retailpred.platform.m5_source import validate_official_m5_sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_directory",
        type=Path,
        nargs="?",
        default=Path("data/v2/source/m5"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/v2/evidence/m5_source_validation.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_official_m5_sources(args.source_directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    print(f"Wrote validation evidence to {args.output}")


if __name__ == "__main__":
    main()
