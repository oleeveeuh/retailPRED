"""Build governed Tableau-ready extracts from frozen evaluation evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from bi.contracts import CONTRACTS
from forecasting.portfolio import MODEL_PORTFOLIO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of-date", type=date.fromisoformat, required=True)
    parser.add_argument(
        "--output-directory", type=Path, default=Path("data/v2/bi/tableau")
    )
    parser.add_argument(
        "--gold-store",
        type=Path,
        default=None,
        help="Local contract store containing gold forecasts; adds live_forecasts.csv.",
    )
    return parser.parse_args()


def _json(path: str) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _model_performance() -> pd.DataFrame:
    mrts = _json("data/v2/mrts_model_summary.json")
    m5_baseline = _json("data/v2/evidence/m5_seasonal_naive/m5_seasonal_naive_summary.json")
    m5_lightgbm = _json("data/v2/evidence/m5_lightgbm/m5_lightgbm_summary.json")
    m5_reconciliation = _json("data/v2/evidence/m5_reconciliation/m5_reconciliation_summary.json")
    m5_chronos = _json("data/v2/evidence/m5_chronos_benchmark/m5_chronos_summary.json")
    mrts_results = mrts["results"]
    assert isinstance(mrts_results, dict)
    mrts_baseline = float(mrts_results["seasonal_naive"]["wape"])
    rows = []
    for model, status in (
        ("seasonal_naive", "baseline"),
        ("global_lightgbm", "development_challenger"),
    ):
        result = mrts_results[model]
        value = float(result["wape"])
        rows.append(
            {
                "dataset": "mrts",
                "use_case": "macro_planning",
                "model_name": model,
                "decision_status": status,
                "evaluation_start": "2015-01-01",
                "evaluation_end": "2025-12-01",
                "metric_name": "wape",
                "metric_value": value,
                "baseline_value": mrts_baseline,
                "relative_improvement": (mrts_baseline - value) / mrts_baseline,
                "code_revision": str(mrts["code_revision"]),
                "data_version": str(mrts["data_version"]),
            }
        )
    baseline = float(m5_baseline["overall_wrmsse"])
    m5_models = (
        ("seasonal_naive", "baseline", baseline, m5_baseline),
        ("m5_global_lightgbm", "promoted_point_champion", float(m5_lightgbm["challenger_wrmsse"]), m5_lightgbm),
        ("wls_reconciliation", "challenger_not_promoted", float(m5_reconciliation["reconciled_wrmsse"]), m5_reconciliation),
        ("chronos_2_zero_shot", "rejected", float(m5_chronos["challenger_wrmsse"]), m5_chronos),
    )
    for model, status, value, evidence in m5_models:
        hashes = evidence["source_sha256"]
        rows.append(
            {
                "dataset": "m5",
                "use_case": "operational_demand",
                "model_name": model,
                "decision_status": status,
                "evaluation_start": "2016-04-25",
                "evaluation_end": "2016-05-22",
                "metric_name": "wrmsse",
                "metric_value": value,
                "baseline_value": baseline,
                "relative_improvement": (baseline - value) / baseline,
                "code_revision": str(evidence["code_revision"]),
                "data_version": str(hashes["sales_train_evaluation.csv"]),
            }
        )
    return pd.DataFrame(rows, columns=CONTRACTS["model_performance.csv"].columns)


def _hierarchy_accuracy() -> pd.DataFrame:
    summary = _json("data/v2/evidence/m5_lightgbm/m5_lightgbm_summary.json")
    frame = pd.read_csv("data/v2/evidence/m5_lightgbm/m5_lightgbm_by_level.csv")
    frame.insert(0, "dataset", "m5")
    frame.insert(1, "model_name", "m5_global_lightgbm")
    frame.insert(2, "evaluation_start", "2016-04-25")
    frame.insert(3, "evaluation_end", "2016-05-22")
    frame["code_revision"] = summary["code_revision"]
    return frame.loc[:, CONTRACTS["m5_hierarchy_accuracy.csv"].columns]


def _uncertainty() -> pd.DataFrame:
    summary = _json("data/v2/evidence/m5_probabilistic/m5_probabilistic_summary.json")
    rows: list[dict[str, object]] = []
    levels = pd.read_csv("data/v2/evidence/m5_probabilistic/m5_probabilistic_by_level.csv")
    for row in levels.to_dict("records"):
        rows.append({"dimension_type": "hierarchy_level", "dimension_key": str(row["level"]), "dimension_label": row["level_name"], "entity_count": row["nodes"], **{key: row[key] for key in ("coverage", "mean_scaled_width", "normalized_wis")}})
    horizons = pd.read_csv("data/v2/evidence/m5_probabilistic/m5_probabilistic_by_horizon.csv")
    for row in horizons.to_dict("records"):
        rows.append({"dimension_type": "horizon", "dimension_key": str(row["horizon"]), "dimension_label": f"Day {row['horizon']}", "entity_count": 42840, **{key: row[key] for key in ("coverage", "mean_scaled_width", "normalized_wis")}})
    segments = pd.read_csv("data/v2/evidence/m5_probabilistic/m5_probabilistic_by_segment.csv")
    for row in segments.to_dict("records"):
        rows.append({"dimension_type": f"segment:{row['segment_type']}", "dimension_key": str(row["segment"]), "dimension_label": str(row["segment"]), "entity_count": row["series"], **{key: row[key] for key in ("coverage", "mean_scaled_width", "normalized_wis")}})
    frame = pd.DataFrame(rows)
    frame.insert(0, "dataset", "m5")
    frame.insert(1, "model_name", "m5_global_lightgbm")
    frame.insert(2, "evaluation_start", "2016-04-25")
    frame.insert(3, "evaluation_end", "2016-05-22")
    frame["nominal_coverage"] = summary["nominal_coverage"]
    frame["decision_status"] = "research_only_failed_segment_gate"
    frame["code_revision"] = summary["code_revision"]
    return frame.loc[:, CONTRACTS["m5_uncertainty.csv"].columns]


def _regimes() -> pd.DataFrame:
    summary = _json("data/v2/evidence/mrts_regimes/mrts_regime_summary.json")
    periods = {"pre_covid": ("2015-01-01", "2020-02-01"), "disruption": ("2020-03-01", "2021-06-01"), "post_disruption": ("2021-07-01", "2025-12-01")}
    rows = []
    for regime, result in summary["regimes"].items():
        start, end = periods[regime]
        rows.append({"dataset": "mrts", "regime": regime, "evaluation_start": start, "evaluation_end": end, "fold_rows_per_model": result["fold_rows_per_model"], "lightgbm_wape": result["global_lightgbm_wape"], "seasonal_naive_wape": result["seasonal_naive_wape"], "relative_wape_improvement": result["relative_wape_improvement"], "mean_category_wape_lightgbm": result["global_lightgbm_mean_category_wape"], "mean_category_wape_seasonal_naive": result["seasonal_naive_mean_category_wape"], "relative_mean_category_wape_improvement": result["relative_mean_category_wape_improvement"], "categories": result["categories"], "improved_categories": result["improved_categories"], "regime_gate_passed": result["gate_passed"], "code_revision": summary["code_revision"]})
    return pd.DataFrame(rows, columns=CONTRACTS["mrts_regime_accuracy.csv"].columns)


def _freshness(as_of: date) -> pd.DataFrame:
    rows = []
    for decision in MODEL_PORTFOLIO:
        rows.append({"dataset": decision.dataset, "use_case": decision.use_case, "evaluation_cutoff": decision.evaluation_cutoff, "as_of_date": as_of.isoformat(), "serving_mode": decision.serving_mode, "freshness_status": "historical_evaluation_only" if decision.dataset == "m5" else "revised_history_development", "point_status": decision.point_status, "uncertainty_status": decision.uncertainty_status, "evidence": decision.evidence[0]})
    return pd.DataFrame(rows, columns=CONTRACTS["data_freshness.csv"].columns)


def _live_forecasts(store_root: Path) -> pd.DataFrame:
    """Shape gold forecasts into the governed live extract; refuses when empty."""
    from retailpred.platform.local_store import LocalTableStore
    from retailpred.platform.specs import GOLD_FORECASTS

    forecasts = LocalTableStore(store_root).read(GOLD_FORECASTS)
    if forecasts.empty:
        raise SystemExit(
            "the store has no gold forecasts; live_forecasts.csv is unavailable"
        )
    from forecasting.portfolio import model_decision

    decision = model_decision(str(forecasts.iloc[0]["dataset"]))
    rows = [
        {
            "dataset": str(row.dataset),
            "series_id": str(row.series_id),
            "forecast_origin": pd.Timestamp(row.forecast_origin).date().isoformat(),
            "target_time": pd.Timestamp(row.target_time).date().isoformat(),
            "horizon": int(row.horizon),
            "prediction": float(row.prediction),
            "model_name": str(row.model_name),
            "model_version": str(row.model_version),
            "model_status": decision.point_status,
            "uncertainty_status": str(row.uncertainty_status),
            "freshness_status": "scheduled_batch_snapshot",
            "code_revision": str(row.code_revision),
        }
        for row in forecasts.itertuples()
    ]
    return pd.DataFrame(rows, columns=CONTRACTS["live_forecasts.csv"].columns)


def main() -> None:
    args = parse_args()
    frames = {
        "model_performance.csv": _model_performance(),
        "m5_hierarchy_accuracy.csv": _hierarchy_accuracy(),
        "m5_uncertainty.csv": _uncertainty(),
        "mrts_regime_accuracy.csv": _regimes(),
        "data_freshness.csv": _freshness(args.as_of_date),
    }
    if args.gold_store is not None:
        frames["live_forecasts.csv"] = _live_forecasts(args.gold_store)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    manifest_files = {}
    for filename, frame in frames.items():
        CONTRACTS[filename].validate(frame)
        path = args.output_directory / filename
        frame.to_csv(path, index=False)
        manifest_files[filename] = {"rows": len(frame), "sha256": _sha256(path), "key": list(CONTRACTS[filename].key)}
    manifest = {"schema_version": "1.0.0", "as_of_date": args.as_of_date.isoformat(), "created_at": datetime.now(timezone.utc).isoformat(), "code_revision": _git_revision(), "files": manifest_files}
    (args.output_directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
