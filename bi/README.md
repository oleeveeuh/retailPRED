# Governed BI extracts

The BI boundary converts frozen evaluation evidence into nine Tableau-ready CSV
tables. It never reads frontend demo calculations or reconstructs metrics from
screenshots.

## Build

```bash
make bi-extracts AS_OF_DATE=2026-11-03
```

Outputs are written under ignored `data/v2/bi/tableau/`. The as-of date is
required so freshness reporting is reproducible rather than silently depending
on the machine clock.

## Tables

### `model_performance.csv`

Grain: dataset, model, metric, and evaluation window.

Contains comparable official results only: MRTS WAPE and complete-hierarchy M5
WRMSSE. The bounded NHITS and Chronos feasibility scores are intentionally
excluded because their subset metric is not comparable to official WRMSSE.
Decision status distinguishes baseline, promoted champion, non-promoted
challenger, and rejected models.

### `m5_hierarchy_accuracy.csv`

Grain: model, evaluation window, and official M5 hierarchy level.

Contains level weight, series count, LightGBM and baseline WRMSSE, and relative
improvement for drill-down from total demand to item/store.

### `m5_uncertainty.csv`

Grain: model, evaluation window, diagnostic dimension type, and key.

Unifies hierarchy-level, horizon, volume-segment, and intermittency-segment
coverage, scaled width, and normalized WIS. Every row retains the failed
serving decision so an 80.38% overall coverage card cannot hide the failed
high-volume or horizon diagnostics.

### `mrts_regime_accuracy.csv`

Grain: declared economic regime.

Contains baseline and LightGBM aggregate and mean-category WAPE, relative
improvement, category win count, and the strict regime gate result.

### `mrts_research_windows.csv`

Grain: model and evaluation window.

Window-scoped satellite research measurements — seasonal naive, global
LightGBM, theta, and the regime-switch policy on exploration (2017–2022),
report (2021–2022), and the single frozen holdout run (2023–2025) origin
windows. Every row carries its origin count and source artifact; subset
windows are never mixed with the official full-grid gate metrics in
`model_performance.csv`. Evidence:
[docs/v2/M3_SATELLITE_FINDINGS.md](../docs/v2/M3_SATELLITE_FINDINGS.md).

The table also carries the predeclared MRTS Chronos benchmark, the H7 final
challenger, and an explicit actual definition. Chronos holdout evidence remains
visibly post-hoc and descriptive.

### `mrts_feature_ladder.csv`

Grain: H7 feature rung on the choose window. Contains incremental
mean-category WAPE change, origin stability, and the frozen advance/no-advance
verdict. Rejected rungs have no report or holdout scores.

### `mrts_horizon_accuracy.csv`

Grain: model and horizon on the full exploration window. Compares seasonal
naive, recursive gate LightGBM, direct LightGBM + growth, and theta under
final-vintage actuals.

### `mrts_candidate_comparisons.csv`

Grain: final H7 challenger, reference model, evaluation window, and actual
definition. Stores paired origin-block point deltas and confidence bounds so
the resolved losses to theta remain visible rather than being inferred from
close-looking bars.

### `data_freshness.csv`

Grain: dataset and declared extract as-of date.

Contains evaluation cutoff, serving mode, point and uncertainty status, and the
tracked evidence reference. This table drives visible historical/development
labels; it must not be interpreted as a live-source freshness measurement.

## Integrity manifest

`manifest.json` records schema version, as-of date, Git revision, row count,
key, and SHA-256 for every CSV. Build-time contracts reject schema drift, empty
tables, null or duplicate keys, and invalid non-negative metrics before any
file is published.

## Tableau workbook

The tracked local workbook is `bi/tableau/retailpred_v2.twb`, backed by the
adjacent `retailpred_v2.hyper` extract. Its nine governed views are:

1. **Model scorecard** — model performance, baseline comparison, decision
   status, evaluation dates, and dataset filter.
2. **M5 hierarchy** — level WRMSSE and relative improvement with level-order
   drill-down.
3. **Uncertainty audit** — coverage versus the 80% reference, width and WIS,
   with dimension-type filters and failed-gate annotation.
4. **MRTS regimes** — aggregate and mean-category WAPE across pre-COVID,
   disruption, and post-disruption periods.
5. **Data status** — cutoff, as-of date, serving restriction, and evidence link
   displayed on every dashboard.
6. **MRTS research windows** — window-scoped mean-category WAPE for the
   satellite's measured models, including zero-shot Chronos, trained NHITS,
   and the final H7 challenger, split by actual definition and colored by
   evidence status.
7. **MRTS feature ladder** — every choose-window rung, incremental WAPE delta,
   origin stability, and frozen advancement verdict.
8. **MRTS horizon accuracy** — final-vintage exploration WAPE at h=1/3/6/12
   across the classical, foundation, neural, learned, and baseline families,
   exposing the recursion improvement and every challenger's theta deficit.
9. **MRTS candidate comparisons** — paired deltas and confidence bounds for
   the final H7 challenger versus naive and theta on both actual definitions.

Tableau calculations must format stored decimal rates as percentages; they must
not recompute WAPE or WRMSSE from differently filtered data. React and Tableau
therefore display the same governed metric values and decision semantics.

See `bi/tableau/README.md` for the view contract, deterministic rebuild command,
and the explicit boundary between a local artifact and public publishing.
