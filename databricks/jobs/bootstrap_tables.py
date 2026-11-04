"""Create declared-schema RetailPRED Delta tables in Unity Catalog."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Serverless python tasks exec this file without defining __file__, so the
# bundle passes the synced repository root as an explicit parameter
# (${workspace.file_path}/files), validated here, with a workspace-layout
# fallback and full diagnostics because the import depends on it.
def _repository_root() -> str:
    argv = sys.argv
    if "--repo-root" in argv:
        root = Path(argv[argv.index("--repo-root") + 1])
        if (root / "retailpred").is_dir():
            return str(root)
        print(f"[bootstrap] --repo-root given but no retailpred package inside: {root}")
    try:
        for candidate in sorted(Path("/Workspace/Users").glob("*/.bundle/*/dev/files")):
            if (candidate / "retailpred").is_dir():
                print(f"[bootstrap] fallback located repository root: {candidate}")
                return str(candidate)
    except OSError as error:
        print(f"[bootstrap] /Workspace/Users not listable: {error}")
    return "."


REPOSITORY_ROOT = Path(_repository_root())
print(
    f"[bootstrap] repository root: {REPOSITORY_ROOT} "
    f"exists={REPOSITORY_ROOT.exists()} "
    f"package={(REPOSITORY_ROOT / 'retailpred').is_dir()}"
)
if not (REPOSITORY_ROOT / "retailpred").is_dir():
    try:
        print(
            "[bootstrap] /Workspace/Users entries:",
            sorted(path.name for path in Path("/Workspace/Users").iterdir())[:10],
        )
    except OSError as error:
        print(f"[bootstrap] /Workspace/Users not listable: {error}")
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
