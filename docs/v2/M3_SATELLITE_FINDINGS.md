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
H1–H5, the reporting-only holdout scorecard, and the H6/H7/H8 addenda),
`CHRONOS_MRTS.md`, `NHITS_MRTS.md`, `H7_FEATURE_LADDER.md`, and
`POLICY_FINDINGS.md` (regime-switch policy layer). Protocol documents:
satellite `PROTOCOL.md`, `POLICY.md`, the two benchmark predeclarations
(`CHRONOS_PREDECLARATION.md`, `NHITS_PREDECLARATION.md`), and the drafted
re-gate design `REGATE_DESIGN.md` (predeclared before measurement; one
commit per component).

## What was measured

1. **The theta method** (Assimakopoulos & Nikolopoulos 2000; custom NumPy,
   no learned parameters beyond one in-sample SES α) recorded the lowest
   mean-category WAPE of the three evaluated models on every evaluated
   window — it beat seasonal naive in 9/11 categories on exploration
   origins (2017-10..2022-10), 10/11 on the report window (2021-01..
   2022-10), and was carried to the single predeclared holdout run
   (2023-01..2025-03): mean-category WAPE 4.98% vs naive 5.15% (+0.17pp,
   origin-block CI −0.17..+0.54 — indistinguishable from naive), improving
   7/11 categories — better than the frozen gate LightGBM slice (5.64%;
   that gap is the one holdout ranking the CI resolves: +0.65,
   +0.46..+0.88) but failing the ≥10/11 category criterion. The holdout
   was computed exactly once; its fold rows were later recovered
   deterministically under a verification-locked protocol (satellite
   `scripts/holdout_theta_recovery.py` — all 34 frozen aggregates
   reproduced at diff 0.0) to enable the reporting-only scorecard
   (satellite `results/holdout_scorecard.md`: MASE, signed bias, horizon
   breakdown, first-release re-scoring). Under first-release actuals the
   same stored predictions give theta a wholly positive holdout margin
   (+1.19pp, CI +0.83..+1.57) — recorded as descriptive only; the gate's
   predeclared actual source (latest_revision_vintage) is unchanged, and
   criterion 2 still fails at 7/11. The scorecard also measured the
   aggregation split that the re-gate design now guards: on the holdout,
   mean-category WAPE puts theta at parity with naive while pooled
   (volume-weighted) WAPE resolves in naive's favor (−0.77pp, CI
   −1.24..−0.36).
2. **A regime-switch policy** (capped LightGBM classifier, isotonic
   calibration on leave-one-origin-out CHOOSE scores, predeclared τ=0.50)
   choosing naive or theta per category and origin beats naive in 10/11
   categories on the report window (7.43% vs 10.51% mean-category WAPE,
   +3.08pp, origin-block bootstrap CI +0.70..+5.13) and beats the recorded
   0.5/0.5 blend — but not pure theta (7.02%). Its predeclared
   beat-everything guardrail failed, so no promotion document was issued;
   the measured lesson is that a switch pays only on regime-mixed
   evaluation windows and cross-regime probability calibration is the open
   defect (Brier 0.155 CHOOSE → 0.276 REPORT). The layer was not carried
   forward.
3. **The H7 feature-group ladder** (predeclared; owner-directed) closed
   the archive-side enrichment question: direct per-horizon LightGBM with
   a growth-feature extension was the only advancing rung (+0.16pp, 13/15
   choose origins); trend, cross-category, revision (unstable, 8/15), and
   origin-constant external-macro context (+0.00pp, 4/15) were rejected;
   removal confirmed growth was load-bearing; an elastic-net arm trailed
   the LightGBM on identical features. The final challenger — direct
   LightGBM + growth — loses to theta with a wholly negative CI on every
   window under final-vintage actuals (full exploration −1.50pp, CI
   −2.45..−0.73; report −2.82pp, CI −4.69..−1.17), improves the recursive
   gate model by 1.42pp at h=12, and still trails theta at every horizon.
   No promotion; the report window was computed exactly once, after the
   ladder concluded.
4. **The H6 Chronos-2 benchmark** (predeclared, owner-directed):
   zero-shot Chronos-2 on identical origins scored 7.75% mean-category
   WAPE on exploration — second to theta (7.51%), ahead of naive (9.30%)
   and the gate LightGBM (9.55%) — and 5.00% on the single post-hoc,
   descriptive holdout pass. No confidence intervals, no selection or
   promotion input (predeclared reading rules 2–3). The M5 rejection does
   not transfer to MRTS and no cross-dataset performance claim is made;
   cross-learning and fine-tuned variants remain untested.
5. **The H8 NHITS benchmark** (predeclared, owner-directed): NHITS
   trained from scratch at every origin on the as-of panel scored 8.91%
   on exploration (behind theta and Chronos, ahead of naive on the point
   delta, 6/11 categories) and 6.90% on the single post-hoc descriptive
   holdout — behind naive (−1.75pp, 2/11), theta (−1.92pp, 1/11), and the
   gate LightGBM slice, degrading monotonically with horizon (5.20% →
   8.75%). Zero negative predictions; no clipping; no configuration
   search (predeclaration). With H6 and H8, the same protocol has
   evaluated all three modern families beside the classical baselines;
   theta ranks first on both windows and Chronos second. The full
   three-family table is in satellite `NHITS_MRTS.md` §"The three-family
   statement".

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
evaluation window, origin count, source artifact, evidence class, and
actual definition — the Chronos and NHITS holdout rows are explicitly
labeled post-hoc and descriptive, and the H7 challenger rows carry their
predeclared evidence classes. The 2026-11-10 scorecard's additional
metrics (MASE, signed bias, horizon breakdown, first-release re-scoring)
remain in the satellite artifacts only and are not added to the governed
extract beyond the actual-definition split the extract declares. No
candidate here is promoted, authorized, or served; a future re-gate (theta
as third benchmark, per satellite FINDINGS §5 and its drafted
predeclaration `REGATE_DESIGN.md`) remains the owner's documented path
after vintage accumulation, with owner-confirmed thresholds recorded
before the window opens.
