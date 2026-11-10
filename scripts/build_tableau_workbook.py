#!/usr/bin/env python3
"""Build the local Tableau Public workbook from governed BI extracts.

The workbook deliberately uses live text connections. The CSVs remain the
governed boundary, while this small generator keeps sheet definitions
reviewable and reproducible instead of committing opaque manual state.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import uuid
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr


@dataclass(frozen=True)
class DataSource:
    key: str
    caption: str
    filename: str
    schema: tuple[tuple[str, str], ...]

    @property
    def tableau_name(self) -> str:
        return f"federated.retailpred_{self.key}"

    @property
    def connection_name(self) -> str:
        return f"hyper.retailpred_{self.key}"


@dataclass(frozen=True)
class Field:
    name: str
    caption: str
    datatype: str
    role: str
    field_type: str
    derivation: str

    @property
    def instance(self) -> str:
        prefix = "none" if self.derivation == "None" else self.derivation.lower()
        suffix = "nk" if self.role == "dimension" else "qk"
        return f"[{prefix}:{self.name}:{suffix}]"


@dataclass(frozen=True)
class Sheet:
    name: str
    datasource_key: str
    rows: tuple[Field, ...]
    columns: tuple[Field, ...]
    color: Field | None = None
    label: Field | None = None
    mark: str = "Bar"


DATA_SOURCES = (
    DataSource(
        "performance",
        "Model Performance",
        "model_performance.csv",
        (
            ("dataset", "string"),
            ("use_case", "string"),
            ("model_name", "string"),
            ("decision_status", "string"),
            ("evaluation_start", "date"),
            ("evaluation_end", "date"),
            ("metric_name", "string"),
            ("metric_value", "real"),
            ("baseline_value", "real"),
            ("relative_improvement", "real"),
            ("code_revision", "string"),
            ("data_version", "string"),
        ),
    ),
    DataSource(
        "hierarchy",
        "M5 Hierarchy Accuracy",
        "m5_hierarchy_accuracy.csv",
        (
            ("dataset", "string"),
            ("model_name", "string"),
            ("evaluation_start", "date"),
            ("evaluation_end", "date"),
            ("level", "integer"),
            ("level_name", "string"),
            ("series_count", "integer"),
            ("weight", "real"),
            ("level_wrmsse", "real"),
            ("baseline_level_wrmsse", "real"),
            ("relative_improvement", "real"),
            ("code_revision", "string"),
        ),
    ),
    DataSource(
        "uncertainty",
        "M5 Uncertainty Audit",
        "m5_uncertainty.csv",
        (
            ("dataset", "string"),
            ("model_name", "string"),
            ("evaluation_start", "date"),
            ("evaluation_end", "date"),
            ("dimension_type", "string"),
            ("dimension_key", "string"),
            ("dimension_label", "string"),
            ("entity_count", "integer"),
            ("nominal_coverage", "real"),
            ("coverage", "real"),
            ("mean_scaled_width", "real"),
            ("normalized_wis", "real"),
            ("decision_status", "string"),
            ("code_revision", "string"),
        ),
    ),
    DataSource(
        "regimes",
        "MRTS Regime Accuracy",
        "mrts_regime_accuracy.csv",
        (
            ("dataset", "string"),
            ("regime", "string"),
            ("evaluation_start", "date"),
            ("evaluation_end", "date"),
            ("fold_rows_per_model", "integer"),
            ("lightgbm_wape", "real"),
            ("seasonal_naive_wape", "real"),
            ("relative_wape_improvement", "real"),
            ("mean_category_wape_lightgbm", "real"),
            ("mean_category_wape_seasonal_naive", "real"),
            ("relative_mean_category_wape_improvement", "real"),
            ("categories", "integer"),
            ("improved_categories", "integer"),
            ("regime_gate_passed", "boolean"),
            ("code_revision", "string"),
        ),
    ),
    DataSource(
        "freshness",
        "Data Status",
        "data_freshness.csv",
        (
            ("dataset", "string"),
            ("use_case", "string"),
            ("evaluation_cutoff", "date"),
            ("as_of_date", "date"),
            ("serving_mode", "string"),
            ("freshness_status", "string"),
            ("point_status", "string"),
            ("uncertainty_status", "string"),
            ("evidence", "string"),
        ),
    ),
    DataSource(
        "research",
        "MRTS Research Windows",
        "mrts_research_windows.csv",
        (
            ("dataset", "string"),
            ("model_name", "string"),
            ("decision_status", "string"),
            ("evaluation_window", "string"),
            ("actual_definition", "string"),
            ("evidence_class", "string"),
            ("origin_count", "integer"),
            ("evaluation_start", "date"),
            ("evaluation_end", "date"),
            ("metric_name", "string"),
            ("metric_value", "real"),
            ("baseline_value", "real"),
            ("relative_improvement", "real"),
            ("categories_beating_naive", "integer"),
            ("categories_total", "integer"),
            ("beats_naive_in_window", "boolean"),
            ("evidence_reference", "string"),
            ("code_revision", "string"),
            ("data_version", "string"),
        ),
    ),
    DataSource(
        "feature_ladder",
        "MRTS Feature Ladder",
        "mrts_feature_ladder.csv",
        (
            ("dataset", "string"),
            ("rung", "string"),
            ("feature_groups", "string"),
            ("compare_to", "string"),
            ("mean_category_delta_pp", "real"),
            ("improved_origins", "integer"),
            ("origins_total", "integer"),
            ("improved_origin_share", "real"),
            ("advanced", "boolean"),
            ("verdict", "string"),
            ("evidence_reference", "string"),
            ("code_revision", "string"),
            ("data_version", "string"),
        ),
    ),
    DataSource(
        "horizon_accuracy",
        "MRTS Horizon Accuracy",
        "mrts_horizon_accuracy.csv",
        (
            ("dataset", "string"),
            ("model_name", "string"),
            ("evaluation_window", "string"),
            ("actual_definition", "string"),
            ("horizon", "integer"),
            ("mean_category_wape", "real"),
            ("decision_status", "string"),
            ("evidence_reference", "string"),
            ("code_revision", "string"),
            ("data_version", "string"),
        ),
    ),
    DataSource(
        "candidate_comparisons",
        "MRTS Candidate Comparisons",
        "mrts_candidate_comparisons.csv",
        (
            ("dataset", "string"),
            ("model_name", "string"),
            ("reference_model", "string"),
            ("evaluation_window", "string"),
            ("actual_definition", "string"),
            ("origin_count", "integer"),
            ("mean_category_delta_pp", "real"),
            ("ci_low_pp", "real"),
            ("ci_high_pp", "real"),
            ("resolved", "boolean"),
            ("verdict", "string"),
            ("evidence_reference", "string"),
            ("code_revision", "string"),
            ("data_version", "string"),
        ),
    ),
)


def dimension(name: str, caption: str, datatype: str = "string") -> Field:
    return Field(name, caption, datatype, "dimension", "nominal", "None")


def measure(name: str, caption: str, datatype: str = "real") -> Field:
    return Field(name, caption, datatype, "measure", "quantitative", "Sum")


SHEETS = (
    Sheet(
        "01 Model Scorecard",
        "performance",
        (dimension("dataset", "Dataset"), dimension("model_name", "Model Name")),
        (measure("metric_value", "Metric Value"),),
        color=dimension("decision_status", "Decision Status"),
        label=measure("metric_value", "Metric Value"),
    ),
    Sheet(
        "02 M5 Hierarchy",
        "hierarchy",
        (dimension("level_name", "Hierarchy Level"),),
        (measure("level_wrmsse", "Level WRMSSE"),),
        color=measure("relative_improvement", "Relative Improvement"),
        label=measure("level_wrmsse", "Level WRMSSE"),
    ),
    Sheet(
        "03 Uncertainty Audit",
        "uncertainty",
        (
            dimension("dimension_type", "Diagnostic Type"),
            dimension("dimension_label", "Diagnostic Segment"),
        ),
        (measure("coverage", "Coverage"),),
        color=dimension("decision_status", "Decision Status"),
        label=measure("coverage", "Coverage"),
    ),
    Sheet(
        "04 MRTS Regimes",
        "regimes",
        (dimension("regime", "Economic Regime"),),
        (measure("lightgbm_wape", "LightGBM WAPE"),),
        color=dimension("regime_gate_passed", "Regime Gate Passed", "boolean"),
        label=measure("lightgbm_wape", "LightGBM WAPE"),
    ),
    Sheet(
        "05 Data Status",
        "freshness",
        (
            dimension("dataset", "Dataset"),
            dimension("use_case", "Use Case"),
            dimension("evaluation_cutoff", "Evaluation Cutoff", "date"),
            dimension("as_of_date", "Extract As-Of Date", "date"),
            dimension("serving_mode", "Serving Mode"),
            dimension("point_status", "Point-Forecast Status"),
            dimension("uncertainty_status", "Uncertainty Status"),
        ),
        (),
        color=dimension("freshness_status", "Freshness Status"),
        label=dimension("evidence", "Evidence"),
        mark="Text",
    ),
    Sheet(
        "06 MRTS Research Windows",
        "research",
        (
            dimension("model_name", "Model"),
            dimension("evaluation_window", "Evaluation Window"),
            dimension("actual_definition", "Actual Definition"),
        ),
        (measure("metric_value", "Mean-Category WAPE"),),
        color=dimension("decision_status", "Decision Status"),
        label=measure("metric_value", "Mean-Category WAPE"),
    ),
    Sheet(
        "07 MRTS Feature Ladder",
        "feature_ladder",
        (
            dimension("feature_groups", "Feature Rung"),
            dimension("improved_origins", "Origins Improved", "integer"),
            dimension("origins_total", "Origins Evaluated", "integer"),
            dimension("verdict", "Frozen Verdict"),
        ),
        (measure("mean_category_delta_pp", "Incremental Mean-Category WAPE Delta (pp)"),),
        color=dimension("advanced", "Advanced", "boolean"),
        label=measure("mean_category_delta_pp", "Incremental Mean-Category WAPE Delta (pp)"),
    ),
    Sheet(
        "08 MRTS Horizon Accuracy",
        "horizon_accuracy",
        (
            dimension("model_name", "Model"),
            dimension("horizon", "Forecast Horizon", "integer"),
        ),
        (measure("mean_category_wape", "Mean-Category WAPE"),),
        color=dimension("decision_status", "Decision Status"),
        label=measure("mean_category_wape", "Mean-Category WAPE"),
    ),
    Sheet(
        "09 MRTS Candidate Comparisons",
        "candidate_comparisons",
        (
            dimension("reference_model", "Reference Model"),
            dimension("evaluation_window", "Evaluation Window"),
            dimension("actual_definition", "Actual Definition"),
            dimension("ci_low_pp", "CI Low (pp)", "real"),
            dimension("ci_high_pp", "CI High (pp)", "real"),
            dimension("verdict", "Inference Verdict"),
        ),
        (measure("mean_category_delta_pp", "Mean-Category WAPE Delta (pp)"),),
        color=dimension("resolved", "Resolved", "boolean"),
        label=measure("mean_category_delta_pp", "Mean-Category WAPE Delta (pp)"),
    ),
)


def _q(value: str) -> str:
    return quoteattr(value, {"'": "&apos;"})


def _datasource_xml(source: DataSource, hyper_filename: str) -> str:
    table_name = f"[Extract].[{Path(source.filename).stem}]"
    return f"""    <datasource caption={_q(source.caption)} inline='true' name={_q(source.tableau_name)} version='18.1'>
      <connection class='federated'>
        <named-connections>
          <named-connection caption={_q(source.caption)} name={_q(source.connection_name)}>
            <connection authentication='auth-none' author-locale='en_US' class='hyper' dbname={_q(hyper_filename)} default-settings='yes' sslmode='' username='tableau_internal_user' />
          </named-connection>
        </named-connections>
        <relation connection={_q(source.connection_name)} name={_q(Path(source.filename).stem)} table={_q(table_name)} type='table' />
      </connection>
      <aliases enabled='yes' />
    </datasource>"""


def _field_definition(field: Field) -> str:
    return (
        f"            <column caption={_q(field.caption)} datatype={_q(field.datatype)} "
        f"name={_q(f'[{field.name}]')} role={_q(field.role)} type={_q(field.field_type)} />"
    )


def _field_instance(field: Field) -> str:
    return (
        f"            <column-instance column={_q(f'[{field.name}]')} derivation={_q(field.derivation)} "
        f"name={_q(field.instance)} pivot='key' type={_q(field.field_type)} />"
    )


def _qualified(source: DataSource, field: Field) -> str:
    return f"[{source.tableau_name}].{field.instance}"


def _shelf(source: DataSource, fields: tuple[Field, ...]) -> str:
    return " / ".join(_qualified(source, field) for field in fields)


def _worksheet_xml(sheet: Sheet, source: DataSource) -> str:
    unique_fields: dict[str, Field] = {}
    for field in (*sheet.rows, *sheet.columns, sheet.color, sheet.label):
        if field is not None:
            unique_fields[field.name] = field
    definitions = "\n".join(_field_definition(field) for field in unique_fields.values())
    instances = "\n".join(_field_instance(field) for field in unique_fields.values())
    encodings: list[str] = []
    if sheet.color:
        encodings.append(f"              <color column={_q(_qualified(source, sheet.color))} />")
    if sheet.label:
        encodings.append(f"              <text column={_q(_qualified(source, sheet.label))} />")
    encoding_xml = "\n".join(encodings)
    rows = escape(_shelf(source, sheet.rows))
    columns = escape(_shelf(source, sheet.columns))
    sheet_uuid = uuid.uuid5(uuid.NAMESPACE_URL, f"retailpred-tableau:{sheet.name}")
    return f"""    <worksheet name={_q(sheet.name)}>
      <table>
        <view>
          <datasources>
            <datasource caption={_q(source.caption)} name={_q(source.tableau_name)} />
          </datasources>
          <datasource-dependencies datasource={_q(source.tableau_name)}>
{definitions}
{instances}
          </datasource-dependencies>
          <aggregation value='true' />
        </view>
        <style />
        <panes>
          <pane selection-relaxation-option='selection-relaxation-allow'>
            <view><breakdown value='auto' /></view>
            <mark class={_q(sheet.mark)} />
            <encodings>
{encoding_xml}
            </encodings>
          </pane>
        </panes>
        <rows>{rows}</rows>
        <cols>{columns}</cols>
      </table>
      <simple-id uuid={_q('{' + str(sheet_uuid).upper() + '}')} />
    </worksheet>"""


def _window_xml(sheet: Sheet, source: DataSource) -> str:
    color_card = ""
    if sheet.color:
        color_card = (
            f"\n          <strip size='160'><card pane-specification-id='0' "
            f"param={_q(_qualified(source, sheet.color))} type='color' /></strip>"
        )
    window_uuid = uuid.uuid5(uuid.NAMESPACE_URL, f"retailpred-tableau-window:{sheet.name}")
    return f"""    <window class='worksheet' maximized='true' name={_q(sheet.name)}>
      <cards>
        <edge name='left'>
          <strip size='160'><card type='pages' /><card type='filters' /><card type='marks' /></strip>
        </edge>
        <edge name='top'>
          <strip size='2147483647'><card type='columns' /></strip>
          <strip size='2147483647'><card type='rows' /></strip>
          <strip size='31'><card type='title' /></strip>
        </edge>
        <edge name='right'>{color_card}
        </edge>
      </cards>
      <simple-id uuid={_q('{' + str(window_uuid).upper() + '}')} />
    </window>"""


def validate_sources(extract_dir: Path) -> None:
    for source in DATA_SOURCES:
        path = extract_dir / source.filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing governed Tableau extract: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            header = tuple(next(csv.reader(handle)))
        expected = tuple(name for name, _ in source.schema)
        if header != expected:
            raise ValueError(
                f"Schema drift in {source.filename}: expected {expected}, found {header}"
            )


def _hyper_value(raw: str, datatype: str) -> object | None:
    if raw == "":
        return None
    if datatype == "date":
        return dt.date.fromisoformat(raw)
    if datatype == "integer":
        return int(raw)
    if datatype == "real":
        return float(raw)
    if datatype == "boolean":
        normalized = raw.strip().lower()
        if normalized not in {"true", "false"}:
            raise ValueError(f"Invalid boolean value: {raw!r}")
        return normalized == "true"
    return raw


def build_hyper(extract_dir: Path, output: Path) -> None:
    try:
        from tableauhyperapi import (
            Connection,
            CreateMode,
            HyperProcess,
            Inserter,
            SqlType,
            TableDefinition,
            TableName,
            Telemetry,
        )
    except ImportError as exc:  # pragma: no cover - environment guidance
        raise RuntimeError(
            "Building the Tableau extract requires the official tableauhyperapi package"
        ) from exc

    sql_types = {
        "string": SqlType.text(),
        "date": SqlType.date(),
        "integer": SqlType.big_int(),
        "real": SqlType.double(),
        "boolean": SqlType.bool(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with HyperProcess(Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU) as hyper:
        with Connection(
            hyper.endpoint,
            str(output),
            CreateMode.CREATE_AND_REPLACE,
        ) as connection:
            connection.catalog.create_schema("Extract")
            for source in DATA_SOURCES:
                table = TableDefinition(
                    TableName("Extract", Path(source.filename).stem),
                    [
                        TableDefinition.Column(name, sql_types[datatype])
                        for name, datatype in source.schema
                    ],
                )
                connection.catalog.create_table(table)
                with (extract_dir / source.filename).open(
                    newline="", encoding="utf-8"
                ) as handle:
                    reader = csv.DictReader(handle)
                    with Inserter(connection, table) as inserter:
                        for record in reader:
                            inserter.add_row(
                                [
                                    _hyper_value(record[name], datatype)
                                    for name, datatype in source.schema
                                ]
                            )
                        inserter.execute()


def build_workbook(repo_root: Path, output: Path) -> None:
    extract_dir = (repo_root / "data/v2/bi/tableau").resolve()
    validate_sources(extract_dir)
    hyper_path = output.with_suffix(".hyper")
    build_hyper(extract_dir, hyper_path)
    by_key = {source.key: source for source in DATA_SOURCES}
    datasource_xml = "\n".join(
        _datasource_xml(source, hyper_path.name) for source in DATA_SOURCES
    )
    worksheet_xml = "\n".join(
        _worksheet_xml(sheet, by_key[sheet.datasource_key]) for sheet in SHEETS
    )
    window_xml = "\n".join(
        _window_xml(sheet, by_key[sheet.datasource_key]) for sheet in SHEETS
    )
    workbook = f"""<?xml version='1.0' encoding='utf-8' ?>
<!-- Generated by scripts/build_tableau_workbook.py. Do not edit manually. -->
<workbook original-version='18.1' source-build='2026.3.0' source-platform='mac' version='18.1'>
  <document-format-change-manifest>
    <ObjectModelEncapsulateLegacy />
    <ObjectModelExtractV2 />
    <ObjectModelTableType />
    <SchemaViewerObjectModel />
    <SheetIdentifierTracking />
    <WindowsPersistSimpleIdentifiers />
  </document-format-change-manifest>
  <preferences>
    <preference name='ui.encoding.shelf.height' value='24' />
    <preference name='ui.shelf.height' value='26' />
  </preferences>
  <datasources>
{datasource_xml}
  </datasources>
  <worksheets>
{worksheet_xml}
  </worksheets>
  <windows source-height='51'>
{window_xml}
  </windows>
</workbook>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(workbook, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root containing data/v2/bi/tableau",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Workbook path (defaults to bi/tableau/retailpred_v2.twb)",
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    output = args.output or repo_root / "bi/tableau/retailpred_v2.twb"
    build_workbook(repo_root, output.resolve())


if __name__ == "__main__":
    main()
