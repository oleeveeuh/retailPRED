#!/usr/bin/env python3
"""Build the frozen SHAP bundle behind the #/lab "why the model predicts" section.

Refits the portfolio's MRTS point model (GlobalLightGBM) on the governed local
store through the exact scoring data path — point-in-time gold at the as-of
moment — then computes exact TreeExplainer SHAP values over the training
design matrix. This is an in-sample explanation of the fitted challenger: it
accounts for the model's internal arithmetic, it is not a performance claim
and it is not live inference. The bundle is recomputed from governed data on
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

from forecasting.models.global_lightgbm import FEATURE_COLUMNS, GlobalLightGBM
from forecasting.portfolio import model_decision
from retailpred.platform.gold import build_gold_monthly
from retailpred.platform.local_store import LocalTableStore
from retailpred.platform.specs import SILVER_MONTHLY

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "frontend/public/data/lab/shap_analysis.json"
ADDITIVITY_TOLERANCE = 1e-6


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
        cwd=REPO_ROOT,
    ).stdout.strip()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_shap_bundle(
    panel: pd.DataFrame,
    *,
    now: datetime,
    revision: str,
    data_sha256: str,
    top_k: int = 8,
) -> dict:
    """Pure core: governed gold monthly panel in, frozen SHAP payload out.

    ``panel`` carries the gold monthly keys (series_id, observation_time,
    value). Kept free of store and filesystem access so the live publish
    task can adopt it verbatim if the portfolio ever promotes the champion.
    """
    decision = model_decision("mrts")
    model = GlobalLightGBM().fit(panel)
    # training_frame() keeps the dropna-filtered index; resetting it makes
    # label-based lookups positional so explainer rows, design rows, and the
    # per-row records all stay aligned.
    frame = model.training_frame().reset_index(drop=True)
    design = frame.loc[:, FEATURE_COLUMNS]

    import shap  # deferred: keeps the heavy import off the platform import path

    explainer = shap.TreeExplainer(model.point_model)
    explanation = explainer(design)
    values = np.asarray(explanation.values, dtype=float)
    base_values = np.asarray(explanation.base_values, dtype=float)
    outputs = np.asarray(model.point_model.predict(design), dtype=float)
    residual = float(np.max(np.abs(base_values + values.sum(axis=1) - outputs)))
    if residual > ADDITIVITY_TOLERANCE:
        raise RuntimeError(
            f"SHAP additivity violated: max |base + sum(shap) - output| = {residual}"
        )

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
            "generated_at": _now(),
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
            "additivity_max_error": residual,
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
    return parser.parse_args()


def _aware_timestamp(value: str) -> datetime:
    result = datetime.fromisoformat(value)
    if result.tzinfo is None or result.utcoffset() is None:
        raise argparse.ArgumentTypeError("timestamp must include a UTC offset")
    return result


def main() -> None:
    args = parse_args()
    silver = LocalTableStore(args.store).read(SILVER_MONTHLY)
    if silver.empty:
        raise SystemExit("no silver monthly observations in the store; run ingest first")
    data_sha256 = _sha(args.store / SILVER_MONTHLY.layer / f"{SILVER_MONTHLY.name}.parquet")
    gold = build_gold_monthly(
        silver, as_of=args.as_of or datetime.now(timezone.utc)
    )
    payload = build_shap_bundle(
        gold,
        now=args.as_of or datetime.now(timezone.utc),
        revision=_git(),
        data_sha256=data_sha256,
        top_k=args.top_k,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    size = args.output.stat().st_size
    provenance = payload["provenance"]
    print(f"  {args.output}  ({size / 1024:.1f} KB)")
    print(
        f"  cutoff={provenance['data_cutoff']} rows={provenance['n_training_rows']:,} "
        f"series={provenance['n_series']} additivity={provenance['additivity_max_error']:.2e}"
    )


if __name__ == "__main__":
    main()
