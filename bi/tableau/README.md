# RetailPRED v2 Tableau workbook

`retailpred_v2.twb` is the local Tableau Public workbook for the governed
Milestone 5 reporting boundary. `retailpred_v2.hyper` is its small packaged
extract. It contains only aggregate evaluation evidence from the nine
contract-validated CSVs; it does not contain the licensed raw M5 source files.

## Views

1. **01 Model Scorecard** compares the official model metric by dataset and
   model and colors each bar by the frozen decision status.
2. **02 M5 Hierarchy** shows WRMSSE at all 12 official hierarchy levels and
   colors the bars by improvement over Seasonal Naive.
3. **03 Uncertainty Audit** exposes coverage by hierarchy, horizon, volume, and
   intermittency segment, retaining the failed serving decision.
4. **04 MRTS Regimes** compares LightGBM WAPE across pre-COVID, disruption,
   and post-disruption regimes and colors the strict gate result.
5. **05 Data Status** displays the historical cutoff, freshness restriction,
   serving mode, uncertainty status, and evidence reference.
6. **06 MRTS Research Windows** displays window-scoped mean-category WAPE for
   naive, LightGBM, theta, the regime-switch policy, zero-shot Chronos, trained
   NHITS, and the final direct LightGBM + growth challenger. Actual definition is explicit;
   post-hoc Chronos holdout evidence remains labeled descriptive.
7. **07 MRTS Feature Ladder** shows every choose-window feature rung, its
   incremental mean-category WAPE delta, and frozen advancement verdict.
8. **08 MRTS Horizon Accuracy** compares final-vintage exploration WAPE at
   h=1/3/6/12 across naive, LightGBM, theta, Chronos, and NHITS evidence where
   available, exposing the long-horizon degradation of the learned families.
9. **09 MRTS Candidate Comparisons** shows paired origin-block WAPE deltas and
   confidence bounds versus naive and theta for final-vintage and first-release
   actuals.

The workbook reads stored WAPE and WRMSSE values. It does not recalculate them
from filtered rows, so the Tableau, API, and React surfaces retain the same
governed definitions.

## Rebuild

```bash
make bi-extracts AS_OF_DATE=2026-11-03
make setup-tableau
make tableau-workbook
```

The generator validates every CSV header before replacing either workbook
artifact. It creates one Hyper database with nine tables and deterministic
sheet identifiers, so diffs remain reviewable. Open `retailpred_v2.twb` in
Tableau Public 2026.3 or newer; the `.hyper` file must remain beside it.

## Publishing boundary

The tracked workbook is intentionally local. Publishing to a Tableau Public
profile is a separate, explicit release action because it uploads the embedded
evaluation evidence and makes the visualization public. No credentials or
Tableau profile identifiers are stored in this repository.

Published: workbook `retailpred_v2` on Tableau Public with all nine views,
owner release 2026-11-10 (the initial 2026-11-05 release carried six).
The showcase embeds view 09 (MRTS Candidate Comparisons) with workbook
tabs enabled, from
`https://public.tableau.com/views/retailpred_v2/09MRTSCandidateComparisons`.
Republishing after a governed extract rebuild remains a manual release
action so the public surface never changes silently.
