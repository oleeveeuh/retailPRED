from pathlib import Path

import pytest
import yaml

from retailpred.platform.spark_delta import (
    conflict_condition,
    create_table_sql,
    merge_condition,
    qualified_table,
    spark_sql_type,
)
from retailpred.platform.specs import (
    ALL_TABLE_SPECS,
    GOLD_M5_DEMAND,
    SILVER_MONTHLY,
)


def test_every_contract_generates_declared_delta_ddl() -> None:
    assert len(ALL_TABLE_SPECS) == len({spec.name for spec in ALL_TABLE_SPECS})
    for spec in ALL_TABLE_SPECS:
        sql = create_table_sql(spec, catalog="retailpred")
        assert f"`retailpred`.`{spec.layer}`.`{spec.name}`" in sql
        assert "USING DELTA" in sql
        assert f"'retailpred.schema_version' = '{spec.schema_version}'" in sql
        assert "delta.enableChangeDataFeed" in sql
        for column in spec.columns:
            assert f"`{column}`" in sql


def test_spark_types_and_nullable_fields_are_explicit() -> None:
    sql = create_table_sql(SILVER_MONTHLY, catalog="retailpred")
    assert "`observation_time` DATE NOT NULL" in sql
    assert "`value` DOUBLE NOT NULL" in sql
    assert "`release_time` TIMESTAMP" in sql
    assert "`release_time` TIMESTAMP NOT NULL" not in sql
    assert spark_sql_type(GOLD_M5_DEMAND, "units") == "BIGINT"
    assert spark_sql_type(GOLD_M5_DEMAND, "year") == "INT"


def test_immutable_merge_conditions_use_null_safe_equality() -> None:
    key_condition = merge_condition(SILVER_MONTHLY)
    value_condition = conflict_condition(SILVER_MONTHLY)
    assert "target.`series_id` <=> source.`series_id`" in key_condition
    assert "target.`observation_time` <=> source.`observation_time`" in key_condition
    assert "target.`vintage_time` <=> source.`vintage_time`" in key_condition
    assert "NOT (target.`value` <=> source.`value`)" in value_condition


def test_sql_identifiers_reject_injection() -> None:
    with pytest.raises(ValueError, match="invalid SQL identifier"):
        qualified_table("retailpred; DROP TABLE x", SILVER_MONTHLY)


def test_declarative_automation_bundle_has_deployable_job_shape() -> None:
    root = Path(__file__).resolve().parents[2]
    bundle = yaml.safe_load((root / "databricks.yml").read_text())
    resources = yaml.safe_load(
        (root / "databricks/resources/platform.job.yml").read_text()
    )
    assert bundle["bundle"]["name"] == "retailpred"
    assert bundle["targets"]["dev"]["mode"] == "development"
    job = resources["resources"]["jobs"]["platform_bootstrap"]
    task = job["tasks"][0]
    assert task["spark_python_task"]["python_file"] == "../jobs/bootstrap_tables.py"
    assert task["environment_key"] == "serverless_default"
    # Free Edition exposes no classic clusters; jobs run on serverless compute.
    assert "existing_cluster_id" not in task
    assert job["environments"][0]["spec"]["environment_version"] == "2"
