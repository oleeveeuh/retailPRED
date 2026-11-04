"""Create declared-schema RetailPRED Delta tables in Unity Catalog."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Serverless python tasks exec this file without defining __file__, so the
# bundle passes the synced repository root as an explicit parameter
# (${workspace.file_path}/files) and it is consumed before the imports below.
def _repository_root() -> str:
    argv = sys.argv
    if "--repo-root" in argv:
        return argv[argv.index("--repo-root") + 1]
    return "."

REPOSITORY_ROOT = Path(_repository_root())
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from retailpred.platform.spark_delta import ensure_delta_table  # noqa: E402
from retailpred.platform.specs import ALL_TABLE_SPECS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--repo-root", default=".", help="Synced bundle root; consumed before imports.")
    args = parser.parse_args()

    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    for spec in ALL_TABLE_SPECS:
        ensure_delta_table(spark, spec, catalog=args.catalog)
        print(f"Ready: {args.catalog}.{spec.layer}.{spec.name} v{spec.schema_version}")


if __name__ == "__main__":
    main()
