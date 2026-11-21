#!/usr/bin/env python3
"""Build the frozen SHAP bundles behind the #/lab "why the model predicts" section.

Two governed artifacts in one bundle:

1. Today's challenger ledger: the portfolio's MRTS point model
   (GlobalLightGBM) refit on the local store through the exact scoring data
   path, explained in-sample by exact TreeExplainer values.
2. The distribution-shift diagnosis: the SAME architecture fit twice on the
   frozen ALFRED research panel — once on latest-revision values and once on
   point-in-time values of the identical months (through the final vintage
   gate origin). The drift between the two attribution vectors is the
   measurable account of the shift that reversed the challenger's measured
   29.2% revised-history improvement (docs/v2/M3_LIGHTGBM.md) into the
   gate's 4.1% degradation (docs/v2/M3_VINTAGE_BACKTEST.md, FAILED). The
   diagnosis explains; the documents measure. Nothing here re-runs the gate
   or touches a holdout.

Every payload is recomputed from governed data and frozen artifacts on
demand and committed with provenance; the browser never receives a model or
any runtime compute beyond display logic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting.data.monthly_dataset import (
    observations_as_of,
    validate_monthly_observations,
)
from forecasting.models.global_lightgbm import FEATURE_COLUMNS, GlobalLightGBM
from forecasting.portfolio import model_decision
from retailpred.platform.gold import build_gold_monthly
from retailpred.platform.local_store import LocalTableStore
from retailpred.platform.specs import SILVER_MONTHLY

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "frontend/public/data/lab/shap_analysis.json"
ADDITIVITY_TOLERANCE = 1e-6

DEVELOPMENT_FOLDS_PATH = REPO_ROOT / "data/v2/mrts_model_folds.csv"
GATE_FOLDS_PATH = REPO_ROOT / "data/v2/vintage_backtest_folds.csv"
GATE_SUMMARY_PATH = REPO_ROOT / "data/v2/vintage_backtest_summary.json"
VINTAGE_PANEL_PATH = REPO_ROOT / "data/v2/alfred_mrts_vintages.csv"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
        cwd=REPO_ROOT,
    ).stdout.strip()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _aware_timestamp(value: str) -> datetime:
    result = datetime.fromisoformat(value)
    if result.tzinfo is None or result.utcoffset() is None:
        raise argparse.ArgumentTypeError("timestamp must include a UTC offset")
    return result


def _explain_panel(model: GlobalLightGBM) -> dict:
    """Exact TreeExplainer values over a fitted model's own training rows.

    The filtered frame's index is reset so label-based lookups stay
    positional and explainer rows, design rows, and per-row records remain
    aligned.
    """
    frame = model.training_frame().reset_index(drop=True)
    design = frame.loc[:, FEATURE_COLUMNS]

    import shap  # deferred: keeps the heavy import off the platform import path

    explanation = shap.TreeExplainer(model.point_model)(design)
    values = np.asarray(explanation.values, dtype=float)
    base_values = np.asarray(explanation.base_values, dtype=float)
    outputs = np.asarray(model.point_model.predict(design), dtype=float)
    residual = float(np.max(np.abs(base_values + values.sum(axis=1) - outputs)))
    if residual > ADDITIVITY_TOLERANCE:
        raise RuntimeError(
            f"SHAP additivity violated: max |base + sum(shap) - output| = {residual}"
        )
    return {
        "frame": frame,
        "design": design,
        "values": values,
        "base_values": base_values,
        "outputs": outputs,
        "residual": residual,
    }


def reference_metrics(dev_folds: pd.DataFrame, gate_summary: dict) -> dict:
    """The two measured endpoints of the reversal, from frozen artifacts only.

    Mean category WAPE is fold-weighted per category (|error| mass over
    |actual| mass), then equally weighted across categories — the recipe
    behind docs/v2/M3_LIGHTGBM.md's 29.2% and the gate's 8.49% vs 8.16%.
    """
    folds = dev_folds.copy()
    folds["abs_err"] = (folds["prediction"] - folds["actual"]).abs()
    folds["abs_act"] = folds["actual"].abs()

    def _mean_category_wape(model_name: str) -> float:
        scoped = folds.loc[folds["model_name"] == model_name]
        grouped = scoped.groupby("series_id", sort=True)
        per_category = grouped["abs_err"].sum() / grouped["abs_act"].sum()
        return float(per_category.mean())

    naive_validation = _mean_category_wape("seasonal_naive")
    lightgbm_validation = _mean_category_wape("global_lightgbm")

    gate_by_series = gate_summary["global_lightgbm"]["by_series"]
    naive_by_series = gate_summary["seasonal_naive"]["by_series"]
    if set(gate_by_series) != set(naive_by_series):
        raise ValueError("gate summary by_series sets disagree between models")
    naive_gate = sum(v["wape"] for v in naive_by_series.values()) / len(naive_by_series)
    lightgbm_gate = sum(v["wape"] for v in gate_by_series.values()) / len(gate_by_series)

    return {
        "development_validation": {
            "label": "revised-history development backtest (M3)",
            "naive_mean_category_wape": naive_validation,
            "lightgbm_mean_category_wape": lightgbm_validation,
            "relative_improvement": 1.0 - lightgbm_validation / naive_validation,
            "source_artifact": "data/v2/mrts_model_folds.csv",
            "document": "docs/v2/M3_LIGHTGBM.md",
        },
        "vintage_gate": {
            "label": "point-in-time vintage gate (FAILED)",
            "naive_mean_category_wape": naive_gate,
            "lightgbm_mean_category_wape": lightgbm_gate,
            "relative_degradation": lightgbm_gate / naive_gate - 1.0,
            "source_artifact": "data/v2/vintage_backtest_summary.json",
            "document": "docs/v2/M3_VINTAGE_BACKTEST.md",
        },
    }


def build_shift_attribution(
    vintage_frame: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    as_of: datetime,
) -> dict:
    """Attribute the shift: one architecture, two draws of the same months.

    ``revised_history`` trains on the latest archived vintage of every
    series/month (the M3 development setting); ``point_in_time`` trains on
    what ``observations_as_of`` returns at the gate's final as-of (the
    vintage-gate setting). Both panels are clipped to the same cutoff so the
    only difference between them is revised versus point-in-time values.
    """
    canonical = validate_monthly_observations(vintage_frame)
    keys = ["series_id", "observation_time"]
    latest = (
        canonical.sort_values("vintage_time")
        .groupby(keys, sort=True)
        .tail(1)
        .reset_index(drop=True)
    )
    revised_panel = latest.loc[
        latest["observation_time"] <= cutoff, ["series_id", "observation_time", "value"]
    ].reset_index(drop=True)
    eligible = observations_as_of(canonical, as_of=as_of)
    vintage_panel = eligible.loc[
        eligible["observation_time"] <= cutoff, ["series_id", "observation_time", "value"]
    ].reset_index(drop=True)

    fits: dict[str, dict] = {}
    for name, panel in (("revised_history", revised_panel), ("point_in_time", vintage_panel)):
        explained = _explain_panel(GlobalLightGBM().fit(panel))
        mean_abs = np.abs(explained["values"]).mean(axis=0)
        fits[name] = {
            "mean_abs": mean_abs,
            "mean_signed": explained["values"].mean(axis=0),
            "mean_value": explained["design"].mean(axis=0).to_numpy(dtype=float),
            "residual": explained["residual"],
            "cutoff": pd.Timestamp(explained["frame"]["observation_time"].max()).date().isoformat(),
            "n_rows": int(len(explained["frame"])),
            "n_series": int(explained["frame"]["series_id"].nunique()),
        }

    drift_abs = fits["point_in_time"]["mean_abs"] - fits["revised_history"]["mean_abs"]
    drift_signed = fits["point_in_time"]["mean_signed"] - fits["revised_history"]["mean_signed"]
    rows = sorted(
        (
            {
                "feature": feature,
                "revised_history": {
                    "mean_abs_shap": float(fits["revised_history"]["mean_abs"][index]),
                    "mean_shap": float(fits["revised_history"]["mean_signed"][index]),
                    "mean_value": float(fits["revised_history"]["mean_value"][index]),
                },
                "point_in_time": {
                    "mean_abs_shap": float(fits["point_in_time"]["mean_abs"][index]),
                    "mean_shap": float(fits["point_in_time"]["mean_signed"][index]),
                    "mean_value": float(fits["point_in_time"]["mean_value"][index]),
                },
                "drift_abs": float(drift_abs[index]),
                "drift_signed": float(drift_signed[index]),
            }
            for index, feature in enumerate(FEATURE_COLUMNS)
        ),
        key=lambda row: abs(row["drift_abs"]),
        reverse=True,
    )
    return {
        "rows": rows,
        "attribution_drift_l1": float(np.abs(drift_abs).sum()),
        "fits": fits,
    }


def build_shift_diagnosis(
    vintage_frame: pd.DataFrame,
    gate_folds: pd.DataFrame,
    dev_folds: pd.DataFrame,
    gate_summary: dict,
    *,
    generated_at: str,
    revision: str,
    input_sha256s: dict[str, str],
) -> dict:
    """Pure core: frozen shift artifacts in, the diagnosis payload out.

    ``gate_folds`` supplies the final measured origin (its as-of and
    forecast origin define the point-in-time draw); ``dev_folds`` and
    ``gate_summary`` supply the two measured WAPE endpoints. Kept free of
    filesystem and clock access — every input, including the generation
    timestamp, arrives as a parameter — so the live publish task can adopt
    it verbatim.
    """
    as_of = pd.Timestamp(gate_folds["origin_as_of"].max())
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("gate fold origin_as_of must be timezone-aware")
    cutoff = pd.Timestamp(gate_folds["forecast_origin"].max())
    references = reference_metrics(dev_folds, gate_summary)
    attribution = build_shift_attribution(vintage_frame, cutoff=cutoff, as_of=as_of.to_pydatetime())
    fits = attribution.pop("fits")

    return {
        "provenance": {
            "generated_at": generated_at,
            "platform_code_revision": revision,
            "input_sha256_by_artifact": dict(input_sha256s),
            "source_artifacts": [
                "data/v2/alfred_mrts_vintages.csv",
                "data/v2/mrts_model_folds.csv",
                "data/v2/vintage_backtest_folds.csv",
                "data/v2/vintage_backtest_summary.json",
            ],
            "documents": [
                "docs/v2/M3_LIGHTGBM.md",
                "docs/v2/M3_VINTAGE_BACKTEST.md",
            ],
            "as_of": as_of.isoformat(),
            "common_cutoff": cutoff.date().isoformat(),
            "revised_history": {
                "n_rows": fits["revised_history"]["n_rows"],
                "n_series": fits["revised_history"]["n_series"],
                "data_cutoff": fits["revised_history"]["cutoff"],
                "additivity_max_error": fits["revised_history"]["residual"],
            },
            "point_in_time": {
                "n_rows": fits["point_in_time"]["n_rows"],
                "n_series": fits["point_in_time"]["n_series"],
                "data_cutoff": fits["point_in_time"]["cutoff"],
                "additivity_max_error": fits["point_in_time"]["residual"],
            },
            "references": references,
            "diagnosis": (
                "the identical GlobalLightGBM architecture fit on revised-history "
                "and point-in-time values of the same months; the attribution drift "
                "is the model-side account of the distribution shift. The WAPE "
                "endpoints are the documents' measurements, recomputed from their "
                "frozen artifacts — nothing here re-runs the gate."
            ),
        },
        "features": list(FEATURE_COLUMNS),
        "rows": attribution["rows"],
        "attribution_drift_l1": attribution["attribution_drift_l1"],
    }


def build_shap_bundle(
    panel: pd.DataFrame,
    *,
    now: datetime,
    generated_at: str,
    revision: str,
    data_sha256: str,
    top_k: int = 8,
) -> dict:
    """Pure core: governed gold monthly panel in, today's-challenger payload out.

    ``panel`` carries the gold monthly keys (series_id, observation_time,
    value). Kept free of store, filesystem, and clock access — every input,
    including the generation timestamp, arrives as a parameter — so the live
    publish task can adopt it verbatim if the portfolio ever promotes the
    champion.
    """
    decision = model_decision("mrts")
    model = GlobalLightGBM().fit(panel)
    explained = _explain_panel(model)
    frame = explained["frame"]
    design = explained["design"]
    values = explained["values"]
    base_values = explained["base_values"]
    outputs = explained["outputs"]

    mean_abs = np.abs(values).mean(axis=0)
    total = float(mean_abs.sum())
    global_importance = sorted(
        (
            {
                "feature": feature,
                "mean_abs_shap": float(mean_abs[index]),
                "share": float(mean_abs[index]) / total,
            }
            for index, feature in enumerate(FEATURE_COLUMNS)
        ),
        key=lambda row: row["mean_abs_shap"],
        reverse=True,
    )

    contributions_per_row = [
        sorted(
            (
                {
                    "feature": feature,
                    "shap": float(values[row_index, feature_index]),
                    "feature_value": float(design.iloc[row_index, feature_index]),
                }
                for feature_index, feature in enumerate(FEATURE_COLUMNS)
            ),
            key=lambda item: abs(item["shap"]),
            reverse=True,
        )[:top_k]
        for row_index in range(len(frame))
    ]
    latest_positions = frame.groupby("series_id", sort=False)[
        "observation_time"
    ].idxmax()
    waterfalls = []
    for series_id in sorted(frame["series_id"].astype(str).unique()):
        row_index = int(latest_positions.loc[series_id])
        waterfalls.append(
            {
                "series_id": series_id,
                "month": pd.Timestamp(frame.iloc[row_index]["observation_time"]).date().isoformat(),
                "base_value": float(base_values[row_index]),
                "model_output": float(outputs[row_index]),
                "contributions": contributions_per_row[row_index],
            }
        )

    return {
        "provenance": {
            "generated_at": generated_at,
            "as_of": pd.Timestamp(now).isoformat(),
            "platform_code_revision": revision,
            "data_sha256": data_sha256,
            "source_artifacts": [f"{SILVER_MONTHLY.layer}/{SILVER_MONTHLY.name}.parquet"],
            "model": decision.point_model,
            "model_version": GlobalLightGBM.version,
            "model_status": decision.point_status,
            "data_cutoff": pd.Timestamp(panel["observation_time"].max()).date().isoformat(),
            "n_training_rows": int(len(frame)),
            "n_series": int(frame["series_id"].nunique()),
            "explainer": "shap.TreeExplainer (exact, in-sample over training rows)",
            "target": "log(value / lag_12)",
            "additivity_max_error": explained["residual"],
        },
        "features": list(FEATURE_COLUMNS),
        "global_importance": global_importance,
        "waterfalls": waterfalls,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=Path("data/v2/store"))
    parser.add_argument("--as-of", type=_aware_timestamp, default=None)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    parser.add_argument("--vintage-csv", type=Path, default=VINTAGE_PANEL_PATH)
    parser.add_argument("--gate-folds", type=Path, default=GATE_FOLDS_PATH)
    parser.add_argument("--dev-folds", type=Path, default=DEVELOPMENT_FOLDS_PATH)
    parser.add_argument("--gate-summary", type=Path, default=GATE_SUMMARY_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for label, path in (
        ("ALFRED vintage panel", args.vintage_csv),
        ("vintage gate folds", args.gate_folds),
        ("development folds", args.dev_folds),
        ("vintage gate summary", args.gate_summary),
    ):
        if not path.exists():
            raise SystemExit(f"{label} not found at {path}; the frozen shift artifacts are required")

    silver = LocalTableStore(args.store).read(SILVER_MONTHLY)
    if silver.empty:
        raise SystemExit("no silver monthly observations in the store; run ingest first")
    data_sha256 = _sha(args.store / SILVER_MONTHLY.layer / f"{SILVER_MONTHLY.name}.parquet")
    now = args.as_of or datetime.now(timezone.utc)
    gold = build_gold_monthly(silver, as_of=now)
    input_sha256s = {
        "data/v2/alfred_mrts_vintages.csv": _sha(args.vintage_csv),
        "data/v2/mrts_model_folds.csv": _sha(args.dev_folds),
        "data/v2/vintage_backtest_folds.csv": _sha(args.gate_folds),
        "data/v2/vintage_backtest_summary.json": _sha(args.gate_summary),
    }
    generated_at = _now()
    revision = _git()
    payload = build_shap_bundle(
        gold,
        now=now,
        generated_at=generated_at,
        revision=revision,
        data_sha256=data_sha256,
        top_k=args.top_k,
    )
    payload["shift_diagnosis"] = build_shift_diagnosis(
        pd.read_csv(args.vintage_csv),
        pd.read_csv(args.gate_folds),
        pd.read_csv(args.dev_folds),
        json.loads(args.gate_summary.read_text()),
        generated_at=generated_at,
        revision=revision,
        input_sha256s=input_sha256s,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    size = args.output.stat().st_size
    provenance = payload["provenance"]
    shift = payload["shift_diagnosis"]
    print(f"  {args.output}  ({size / 1024:.1f} KB)")
    print(
        f"  cutoff={provenance['data_cutoff']} rows={provenance['n_training_rows']:,} "
        f"series={provenance['n_series']} additivity={provenance['additivity_max_error']:.2e}"
    )
    refs = shift["provenance"]["references"]
    print(
        f"  shift: validation +{(100 * refs['development_validation']['relative_improvement']):.2f}%"
        f" -> gate {(100 * refs['vintage_gate']['relative_degradation']):.2f}% degradation"
        f" | attribution drift L1 {shift['attribution_drift_l1']:.4f}"
    )


if __name__ == "__main__":
    main()
