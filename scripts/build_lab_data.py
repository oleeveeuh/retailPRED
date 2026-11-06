#!/usr/bin/env python3
"""Build the governed data bundles behind the #/lab interactive components.

Every bundle is recomputed from frozen artifacts only and carries its
provenance; the browser never receives anything but these JSON files —
no served model, no runtime compute beyond display logic.

Sources:
- data/v2/alfred_mrts_vintages.csv                      (this repo, frozen)
- data/v2/evidence/mrts_research_windows/...            (this repo, frozen)
- ../retailPRED-satellite/results/{policy_train,category_win_matrix,policy_report}.json
  (frozen satellite artifacts; --satellite-results overrides the path)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "frontend/public/data/lab"

DATA_SHA = "b754ac133667eed1dbb6a3e33d2d6a19730bddcbe727d55cb5aa08237a20ec86"
DEMO_MONTHS = (
    ("2020-04-01", "the crash month"),
    ("2021-03-01", "the inflationary rebound"),
    ("2023-06-01", "a settled month"),
)
MAX_VINTAGES = 12


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
        cwd=REPO_ROOT,
    ).stdout.strip()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(name: str, payload: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    path.write_text(json.dumps(payload, indent=2) + "\n")
    size = path.stat().st_size
    print(f"  {path.relative_to(REPO_ROOT)}  ({size / 1024:.1f} KB)")


def tau_sweep(satellite_results: Path) -> None:
    train = json.loads((satellite_results / "policy_train.json").read_text())
    report = json.loads((satellite_results / "policy_report.json").read_text())
    refs = train["tau_sweep_reference_mean_cat_wape"]
    payload = {
        "provenance": {
            "generated_at": _now(),
            "platform_code_revision": _git(),
            "data_sha256": DATA_SHA,
            "source_artifacts": [
                "retailPRED-satellite results/policy_train.json (tau_sweep_choose_oof)",
                "retailPRED-satellite results/policy_report.json",
            ],
            "note": (
                "CHOOSE out-of-fold sweep only. The REPORT window is a single frozen "
                "point at the frozen tau — it is never a curve, because sweeping tau "
                "on evaluation origins would be re-tuning."
            ),
        },
        "frozen_tau": train["tau"],
        "frozen_model": train["chosen_model"],
        "guardrail_max_loss_pp": 0.5,
        "choose_oof_brier": train["oof_brier_choose"],
        "n_choose_oof_pairs": train["provenance"]["n_train_rows"],
        "n_report_pairs": train["provenance"]["n_pairs_report"],
        "references": {k: round(v, 6) for k, v in refs.items()},
        "sweep": [
            {
                "tau": row["tau"],
                "policy_mean_cat_wape": round(row["policy_mean_cat_wape"], 6),
                "delta_vs_naive_pp": round(row["delta_vs_naive_pp"], 4),
                "theta_call_rate": round(row["theta_call_rate"], 4),
            }
            for row in train["tau_sweep_choose_oof"]
        ],
        "report_point": {
            "tau": report["frozen_policy"]["tau"],
            "policy_mean_cat_wape": report["secondary_mean_category_wape"]["policy"],
            "naive_mean_cat_wape": report["secondary_mean_category_wape"]["naive"],
            "categories_beating_naive": report["primary_categories_beating_naive"]["count"],
        },
    }
    _write("policy_tau_sweep.json", payload)


def vintage_revisions(panel_path: Path) -> None:
    panel = pd.read_csv(
        panel_path, parse_dates=["observation_time", "vintage_time"]
    )
    months = []
    for month, label in DEMO_MONTHS:
        categories = []
        for series_id, group in panel.loc[
            panel["observation_time"] == pd.Timestamp(month)
        ].groupby("series_id", sort=True):
            ordered = group.sort_values("vintage_time")
            vintages = [
                {"value": float(row.value), "vintage_date": row.vintage_time.date().isoformat()}
                for row in ordered.head(MAX_VINTAGES).itertuples()
            ]
            categories.append(
                {
                    "series_id": series_id,
                    "latest_revised": float(ordered["value"].iloc[-1]),
                    "n_vintages_total": int(len(ordered)),
                    "vintages": vintages,
                }
            )
        months.append({"month": month, "label": label, "categories": categories})
    payload = {
        "provenance": {
            "generated_at": _now(),
            "platform_code_revision": _git(),
            "data_sha256": _sha(panel_path),
            "source_artifact": "data/v2/alfred_mrts_vintages.csv",
            "rule": (
                "values exactly as published in each ALFRED vintage; "
                f"first {MAX_VINTAGES} vintages shown, total count retained"
            ),
            "availability_rule": (
                "a month's first value appears in the vintage published about "
                "45 days after month-end; every later vintage may revise it"
            ),
        },
        "months": months,
    }
    _write("vintage_revisions.json", payload)


def window_categories(satellite_results: Path) -> None:
    summary = json.loads(
        (REPO_ROOT / "data/v2/evidence/mrts_research_windows/mrts_window_summary.json")
        .read_text()
    )
    matrix = json.loads((satellite_results / "category_win_matrix.json").read_text())
    report = json.loads((satellite_results / "policy_report.json").read_text())

    window_keys = {
        "exploration_2017_2022": "full_exploration",
        "report_2021_2022": "report (2021-01..2022-10)",
    }
    model_map = {"naive": "seasonal_naive", "lightgbm": "global_lightgbm", "theta": "theta_method"}
    rows: list[dict] = []
    for window, matrix_key in window_keys.items():
        per_cat = matrix["windows"][matrix_key]["per_category_wape"]
        for series, values in per_cat.items():
            for model_key, model in model_map.items():
                rows.append(
                    {"window": window, "model": model, "series_id": series,
                     "wape": round(float(values[model_key]), 6)}
                )
    primary = report["primary_categories_beating_naive"]
    for series, wape in primary["policy_category_wape"].items():
        rows.append({"window": "report_2021_2022", "model": "regime_switch_policy",
                     "series_id": series, "wape": round(float(wape), 6)})

    means = {
        row["model_name"]: {
            w["evaluation_window"]: w["metric_value"]
            for w in summary["rows"] if w["model_name"] == row["model_name"]
        }
        for row in summary["rows"]
    }
    payload = {
        "provenance": {
            "generated_at": _now(),
            "platform_code_revision": _git(),
            "data_sha256": DATA_SHA,
            "source_artifacts": [
                "data/v2/evidence/mrts_research_windows/mrts_window_summary.json",
                "retailPRED-satellite results/category_win_matrix.json",
                "retailPRED-satellite results/policy_report.json",
            ],
            "note": (
                "policy is evaluated on the report window only (single frozen pass); "
                "holdout is absent by design — it is never re-derived"
            ),
        },
        "windows": list(window_keys),
        "models": ["seasonal_naive", "theta_method", "regime_switch_policy", "global_lightgbm"],
        "mean_category_wape": means,
        "rows": rows,
    }
    _write("window_categories.json", payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--satellite-results", type=Path,
        default=REPO_ROOT.parent / "retailPRED-satellite" / "results",
    )
    parser.add_argument("--panel", type=Path, default=REPO_ROOT / "data/v2/alfred_mrts_vintages.csv")
    args = parser.parse_args()
    if not args.satellite_results.exists():
        raise SystemExit(f"satellite results not found at {args.satellite_results}")
    digest = _sha(args.panel)
    if digest != DATA_SHA:
        raise SystemExit(f"panel hash mismatch: {digest}")
    print("building lab data bundles:")
    tau_sweep(args.satellite_results)
    vintage_revisions(args.panel)
    window_categories(args.satellite_results)
    print("done.")


if __name__ == "__main__":
    main()
