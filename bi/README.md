# Governed BI extracts

The BI boundary converts frozen evaluation evidence into five Tableau-ready CSV
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

## Recommended Tableau workbook

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

Tableau calculations must format stored decimal rates as percentages; they must
not recompute WAPE or WRMSSE from differently filtered data. React and Tableau
therefore display the same governed metric values and decision semantics.
