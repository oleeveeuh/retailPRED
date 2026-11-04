# M5 reconciliation rolling-origin stability

Date: 2026-11-02  
Status: strict promotion gate not passed; retain as challenger

## Decision

Hierarchy-aware reconciliation improves four of five 28-day windows and reduces
mean WRMSSE from 0.825485 to 0.787652. Its mean relative improvement is 4.79%
and median improvement is 5.51%, but the earliest fold regresses by 1.82%.

Because the promoted bottom-up LightGBM passed all five of its comparisons, the
same all-fold standard applies here. Reconciliation therefore remains an
evidence-backed challenger rather than the default champion.

| Evaluation origin | Bottom-up | Reconciled | Change |
|---|---:|---:|---:|
| `d_1801` | 0.802033 | 0.816615 | -1.82% |
| `d_1829` | 0.957064 | 0.945754 | +1.18% |
| `d_1857` | 0.791474 | 0.734801 | +7.16% |
| `d_1885` | 0.763554 | 0.672587 | +11.91% |
| `d_1913` | 0.813301 | 0.768502 | +5.51% |

## Stability by hierarchy level

| Level | Grain | Improved folds | Mean change | Worst fold |
|---:|---|---:|---:|---:|
| 1 | Total | 3/5 | +7.19% | -4.32% |
| 2 | State | 4/5 | +5.56% | -4.49% |
| 3 | Store | 4/5 | +6.03% | -3.11% |
| 4 | Category | 5/5 | +8.64% | +0.21% |
| 5 | Department | 4/5 | +7.55% | -2.70% |
| 6 | State × category | 4/5 | +6.67% | -0.82% |
| 7 | State × department | 4/5 | +6.20% | -3.09% |
| 8 | Store × category | 4/5 | +6.13% | -0.58% |
| 9 | Store × department | 4/5 | +5.35% | -1.55% |
| 10 | Item | 3/5 | +0.45% | -1.13% |
| 11 | Item × state | 3/5 | +0.20% | -0.74% |
| 12 | Item × store | 3/5 | +0.08% | -0.41% |

Reconciliation is consistently strongest at aggregate levels, especially
category, while item-level changes remain deliberately small because the method
scales the existing item/store forecast within store/department groups.

## Integrity checks

Every fold independently retrains both LightGBMs and relearns its blend weights
and residual variances from the immediately preceding validation window. Scales
and revenue weights are also recomputed at each evaluation origin. Maximum final
coherence error is 0.0 in all five folds.

Run the audit with:

```bash
make rolling-reconcile-m5
```

Code revision: `0464ed6bd973bd025e9a0ff88e5ac97a08ab5fc9`  
Runtime: 138.62 seconds for five full retrains.

## Interpretation

The result demonstrates real value from hierarchy-aware modeling without hiding
its instability. The bottom-up global LightGBM remains the production candidate.
Reconciliation can be revisited after adding a genuinely later test period or a
predeclared nested deployment gate; the current folds may not be reused to invent
and validate such a gate simultaneously.
