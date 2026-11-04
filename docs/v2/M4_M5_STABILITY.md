# M5 LightGBM rolling-origin stability

Date: 2026-11-02  
Status: passed; LightGBM promoted over Seasonal Naive

## Decision

Global LightGBM improves overall WRMSSE in **all five** 28-day evaluation
windows. The worst improvement is 6.27%, the median is 10.96%, and the mean is
13.77%. This passes the stability gate and promotes LightGBM as RetailPRED's
current M5 point-forecast champion.

| Evaluation origin | Evaluation targets | Seasonal Naive | LightGBM | Improvement |
|---|---|---:|---:|---:|
| `d_1801` | `d_1802–d_1829` | 1.112062 | 0.802033 | 27.88% |
| `d_1829` | `d_1830–d_1857` | 1.021062 | 0.957064 | 6.27% |
| `d_1857` | `d_1858–d_1885` | 0.888939 | 0.791474 | 10.96% |
| `d_1885` | `d_1886–d_1913` | 0.922781 | 0.763554 | 17.26% |
| `d_1913` | `d_1914–d_1941` | 0.869701 | 0.813301 | 6.48% |

Mean WRMSSE falls from 0.962909 to 0.825485. The mean of the five fold-level
relative improvements is 13.77%.

## Hierarchy stability

| Level | Grain | Improved folds | Mean improvement | Worst fold |
|---:|---|---:|---:|---:|
| 1 | Total | 4/5 | 8.72% | -2.54% |
| 2 | State | 5/5 | 10.26% | +0.98% |
| 3 | Store | 4/5 | 9.02% | -2.13% |
| 4 | Category | 4/5 | 9.04% | -0.36% |
| 5 | Department | 4/5 | 11.22% | -0.59% |
| 6 | State × category | 4/5 | 10.78% | -0.84% |
| 7 | State × department | 4/5 | 11.70% | -0.63% |
| 8 | Store × category | 5/5 | 10.74% | +1.22% |
| 9 | Store × department | 5/5 | 11.95% | +3.23% |
| 10 | Item | 5/5 | 18.33% | +14.14% |
| 11 | Item × state | 5/5 | 21.66% | +18.39% |
| 12 | Item × store | 5/5 | 24.24% | +21.58% |

The operational levels improve in every window. Occasional small aggregate-level
regressions motivate a hierarchy-aware challenger, but they do not overturn the
overall promotion decision.

## Leakage and lifecycle controls

Each fold has six earlier training windows, a disjoint validation window for
early stopping, and a subsequent evaluation window. Scales and dollar-sales
weights are recomputed using only data available at that fold's origin.

At early origins, some products have not launched and therefore have no valid
RMSSE scale. They are included only when their pre-origin revenue weight is zero;
any positively weighted node without a valid scale fails the run. This preserves
the scoring boundary without inventing history for pre-launch products.

Run the audit with:

```bash
make rolling-m5
```

Code revision: `6159558dd347856edb79a76ac3b4a3194414715b`  
Runtime: 121.28 seconds for five complete retrains.
