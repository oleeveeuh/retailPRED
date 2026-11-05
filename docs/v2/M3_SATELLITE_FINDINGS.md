# M3 satellite findings — the vintage-gate follow-up research (recorded evidence)

Status: **recorded research findings from the investigation satellite; no
platform serving, scoring, or gate decision changes.** The 2026-11-04
vintage gate verdict ([M3_VINTAGE_BACKTEST.md](M3_VINTAGE_BACKTEST.md))
stands: the real-time model decision stays fail-closed. This document
records what the satellite measured so the governed BI layer can display
it with explicit evaluation windows.

Source: [oleeveeuh/retailPRED-satellite](https://github.com/oleeveeuh/retailPRED-satellite)
(local investigation satellite; findings flow back as documents only).
Findings documents: satellite `FINDINGS.md` (gate forensics, hypotheses
H1–H5) and `POLICY_FINDINGS.md` (regime-switch policy layer). Protocol
documents: satellite `PROTOCOL.md`, `POLICY.md` (predeclared before
measurement; one commit per component).

## What was measured

1. **The theta method** (Assimakopoulos & Nikolopoulos 2000; custom NumPy,
   no learned parameters beyond one in-sample SES α) is the best measured
   real-time MRTS forecaster on every evaluated window — it beat seasonal
   naive in 9/11 categories on exploration origins (2017-10..2022-10),
   10/11 on the report window (2021-01..2022-10), and was carried to the
   single predeclared holdout run (2023-01..2025-03): mean-category WAPE
   4.98% vs naive 5.15% (+0.17pp), improving 7/11 categories — better than
   the frozen gate LightGBM slice (5.64%) but failing the ≥10/11 category
   criterion. The holdout was computed exactly once and is never re-read.
2. **A regime-switch policy** (capped LightGBM classifier, isotonic
   calibration on leave-one-origin-out CHOOSE scores, predeclared τ=0.50)
   choosing naive or theta per category and origin beats naive in 10/11
   categories on the report window (7.43% vs 10.51% mean-category WAPE,
   +3.08pp, origin-block bootstrap CI +0.70..+5.13) and beats the recorded
   0.5/0.5 blend — but not pure theta (7.02%). Its predeclared
   beat-everything guardrail failed, so no promotion document was issued;
   the measured lesson is that a switch pays only on regime-mixed
   evaluation windows and cross-regime probability calibration is the open
   defect (Brier 0.155 CHOOSE → 0.276 REPORT).

## Evidence lineage

- Frozen artifacts: satellite results JSONs (`category_win_matrix.json`,
  `holdout_theta.json`, `policy_report.json`), consolidated by the
  satellite into `results/mrts_window_summary.json` (satellite revision
  `4eb0261`, one commit per component; nothing on the spent holdout is
  re-derived — holdout numbers are copied from the frozen single run).
- Platform copy frozen at
  `data/v2/evidence/mrts_research_windows/mrts_window_summary.json`.
- Data: `alfred_mrts_vintages.csv`, SHA-256
  `b754ac133667eed1dbb6a3e33d2d6a19730bddcbe727d55cb5aa08237a20ec86`
  (byte-identical to this repo's `data/v2/alfred_mrts_vintages.csv` and to
  the failed gate's input; fold-for-fold gate reproduction verified exact).
- Environment: python 3.11.17, lightgbm 4.5.0, scikit-learn 1.5.2,
  pandas 2.2.3, numpy 2.1.3 (the pinned gate environment).

## Boundary statement

These are window-scoped research measurements on exploration/report/holdout
origin subsets. They are **not comparable to the official full-grid MRTS
gate metrics** in `model_performance.csv` and are therefore kept in a
separate extract (`mrts_research_windows.csv`) whose every row carries its
evaluation window, origin count, and source artifact. No candidate here is
promoted, authorized, or served; a future re-gate (theta as third
benchmark, per satellite FINDINGS §5) remains the owner's documented path
after vintage accumulation.
