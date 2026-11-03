# M5 hierarchy-aware reconciliation challenger

Date: 2026-11-02  
Status: measured challenger; rolling-origin promotion gate pending

## Result

Validation-weighted upper-hierarchy reconciliation scores **0.768502 WRMSSE** on
`d_1914–d_1941`. This is 5.51% better than the promoted coherent bottom-up
LightGBM score of 0.813301 and 11.64% better than Seasonal Naive at 0.869701.

| Level | Grain | Bottom-up LightGBM | Reconciled | Improvement |
|---:|---|---:|---:|---:|
| 1 | Total | 0.742380 | 0.683681 | 7.91% |
| 2 | State | 0.746982 | 0.694514 | 7.02% |
| 3 | Store | 0.795100 | 0.742715 | 6.59% |
| 4 | Category | 0.764824 | 0.695330 | 9.09% |
| 5 | Department | 0.821928 | 0.758749 | 7.69% |
| 6 | State × category | 0.789547 | 0.724263 | 8.27% |
| 7 | State × department | 0.835649 | 0.772646 | 7.54% |
| 8 | Store × category | 0.826314 | 0.769890 | 6.83% |
| 9 | Store × department | 0.860916 | 0.814137 | 5.43% |
| 10 | Item | 0.875262 | 0.868879 | 0.73% |
| 11 | Item × state | 0.857518 | 0.854620 | 0.34% |
| 12 | Item × store | 0.843189 | 0.842599 | 0.07% |

The reconciliation challenger improves every hierarchy level. Most gains occur
where independent aggregate forecasts contribute new information; lower-level
changes are intentionally smaller.

## Method

1. The promoted global LightGBM produces 30,490 item/store forecasts, which form
   the coherent bottom-up reference.
2. A separate global Tweedie LightGBM forecasts all 154 nodes in levels 1–9.
3. For each level, aggregate and bottom-up predictions are blended using relative
   errors measured only on the prior `d_1886–d_1913` validation window. Aggregate
   weights range from 0.34 at total to approximately 0.54 at store/department.
4. Node error variances are estimated from those validation residuals and shrunk
   halfway toward the corresponding level median.
5. Nonnegative diagonal-variance weighted least squares reconciles levels 1–9
   over the 70 store/department groups.
6. Reconciled store/department totals scale the item/store forecasts within each
   group, after which the complete 42,840-node hierarchy is rebuilt bottom-up.

The independent base has a maximum coherence discrepancy of 2,020.29 units. The
final reconciled hierarchy has a measured maximum discrepancy of exactly 0.0.

This is a scalable, diagonal-variance WLS reconciliation. It does not claim to be
full-covariance MinT. The 154-by-70 upper system captures the business-relevant
aggregate constraints without solving a 42,840-by-30,490 bounded system for each
horizon.

## Leakage boundary

All blend weights and reconciliation variances use the validation window ending
at `d_1913`. The reported comparison is evaluated only on `d_1914–d_1941` with
the same fixed scales and revenue weights as the existing champion. No evaluation
actual is used to select forecasts, weights, or variances.

Reproduce with:

```bash
make reconcile-m5
```

Code revision: `9938fde2caf081753d5ff3fc0b218c4465e3568f`  
Runtime: 28.72 seconds.

## Promotion decision

The one-window result passes numerical, coherence, and per-level gates, but it is
not promoted yet. The next checkpoint repeats the entire aggregate-model,
variance-estimation, and reconciliation process across earlier rolling origins.
