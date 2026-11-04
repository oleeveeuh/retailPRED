"""Create declared-schema RetailPRED Delta tables in Unity Catalog."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from retailpred.platform.spark_delta import ensure_delta_table  # noqa: E402
from retailpred.platform.specs import ALL_TABLE_SPECS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    args = parser.parse_args()

    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    for spec in ALL_TABLE_SPECS:
        ensure_delta_table(spark, spec, catalog=args.catalog)
        print(f"Ready: {args.catalog}.{spec.layer}.{spec.name} v{spec.schema_version}")


if __name__ == "__main__":
    main()
