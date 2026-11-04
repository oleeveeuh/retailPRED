# M5 Seasonal Naive baseline

Date: 2026-11-02  
Status: measured, reproducible baseline

## Result

The first real-data M5 benchmark achieves **0.869701 WRMSSE** using a weekly
Seasonal Naive forecast aggregated bottom-up across the complete hierarchy.

| Level | Grain | Series | Level WRMSSE |
|---:|---|---:|---:|
| 1 | Total | 1 | 0.751430 |
| 2 | State | 3 | 0.754396 |
| 3 | Store | 10 | 0.799304 |
| 4 | Category | 3 | 0.762047 |
| 5 | Department | 7 | 0.817086 |
| 6 | State × category | 9 | 0.782983 |
| 7 | State × department | 21 | 0.830436 |
| 8 | Store × category | 30 | 0.839263 |
| 9 | Store × department | 70 | 0.891855 |
| 10 | Item | 3,049 | 1.032248 |
| 11 | Item × state | 9,147 | 1.074959 |
| 12 | Item × store | 30,490 | 1.100402 |

Each level carries one-twelfth of the total score weight. `Level WRMSSE` is the
within-level weighted score; the overall result is the equal average of these 12
values.

The measured result also independently reproduces the public M5 validation
Seasonal Naive score reported as 0.86967 in a community implementation:
[M5 statistical benchmark discussion](https://www.kaggle.com/competitions/m5-forecasting-accuracy/discussion/150078).
The small displayed difference is consistent with rounding. Metric construction
follows the official [M5 Competitors' Guide](https://github.com/Mcompetitions/M5-methods/blob/master/M5-Competitors-Guide.pdf)
and [M5 methods repository](https://github.com/Mcompetitions/M5-methods).

## Frozen evaluation contract

- Training observations: `d_1`–`d_1913`.
- Evaluation observations: `d_1914`–`d_1941`, a 28-day holdout.
- Forecast: repeat `d_1907`–`d_1913` four times at each bottom series.
- RMSSE scales: naive-one squared differences through `d_1913`, starting only
  after each node's first nonzero sale.
- Revenue weights: units multiplied by the contemporaneous weekly price over
  `d_1886`–`d_1913` only.
- Hierarchy weights: equal total weight for each of the 12 levels, then recent
  dollar-sales weighting within a level.
- Reconciliation: exact bottom-up aggregation of all 30,490 product/store
  forecasts into 42,840 nodes.

No evaluation target is used in feature construction, scale estimation, price
weighting, or model fitting.

## Reproduction

```bash
make baseline-m5
```

Code revision: `f5f8cdda5eb931415630c12179d446eb7cb6160a`

The local evidence directory contains the JSON run manifest and a CSV with all
12 level scores. It remains ignored because the manifest records the private
local source path; the measured aggregate results above are committed.

## Interpretation

Errors increase toward the item/store level, where demand is sparse and noisy.
That gradient makes the next challenger meaningful: a global model must improve
fine-grained demand without sacrificing the already stronger aggregate levels.
All later M5 models must reuse this exact holdout, scale, weighting, and hierarchy
contract.
