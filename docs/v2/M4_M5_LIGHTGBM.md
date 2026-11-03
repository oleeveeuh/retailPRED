# M5 global LightGBM challenger

Date: 2026-11-02  
Status: measured challenger; not yet promoted as final champion

## Result

The direct multi-horizon global LightGBM scores **0.818992 WRMSSE**, improving
the frozen weekly Seasonal Naive baseline of 0.869701 by **5.83%**.

| Level | Grain | Seasonal Naive | LightGBM | Relative change |
|---:|---|---:|---:|---:|
| 1 | Total | 0.751430 | 0.747257 | +0.56% |
| 2 | State | 0.754396 | 0.752099 | +0.30% |
| 3 | Store | 0.799304 | 0.798246 | +0.13% |
| 4 | Category | 0.762047 | 0.774368 | -1.62% |
| 5 | Department | 0.817086 | 0.832576 | -1.90% |
| 6 | State × category | 0.782983 | 0.798208 | -1.94% |
| 7 | State × department | 0.830436 | 0.845613 | -1.83% |
| 8 | Store × category | 0.839263 | 0.832980 | +0.75% |
| 9 | Store × department | 0.891855 | 0.867778 | +2.70% |
| 10 | Item | 1.032248 | 0.875874 | +15.15% |
| 11 | Item × state | 1.074959 | 0.858999 | +20.09% |
| 12 | Item × store | 1.100402 | 0.843907 | +23.31% |

Positive change means lower WRMSSE. LightGBM improves eight of 12 levels and
delivers its largest gains at the sparse operational grains. Four middle
aggregate levels regress by 1.6–1.9%, so the model is a strong challenger rather
than an automatic promotion.

## Experimental design

- One pooled Tweedie LightGBM learns all 30,490 item/store series and all 28
  horizons directly.
- Six non-overlapping training origins, `d_1717` through `d_1857`, produce
  5,122,320 training rows.
- `d_1885` is a training-only validation origin with 853,720 rows for early
  stopping. Training stops at iteration 65 of a maximum 500.
- The untouched comparison remains `d_1914`–`d_1941`, scored with exactly the
  baseline's fixed scales, dollar weights, and 42,840-node hierarchy.
- Predictions are nonnegative and reconciled bottom-up, so all reported hierarchy
  forecasts are coherent.

The 29 features include causal demand lags and rolling summaries, static retail
hierarchy identities, target horizon, known calendar/events/SNAP, and the
competition-provided target-week sell price. In a production deployment, future
price must come from the retailer's approved price plan or an explicit scenario;
it cannot be silently replaced by realized future data.

## Explainability and runtime

The top gain features are 28-day, 56-day, and seven-day rolling means, followed
by item identity and target weekday. This is consistent with a global model
sharing seasonal demand structure while retaining item-level differences.

The complete local run took 26.38 seconds on Apple arm64 with Python 3.11.17 and
LightGBM 4.5.0; model fitting took 17.42 seconds. Source file hashes match the M5
source validation report.

Code revision: `3d07c852753804f3dfbacd41266c01014b5619bb`

Reproduce with:

```bash
make benchmark-m5
```

## Promotion decision

The overall result passes a simple accuracy gate, but the model is not yet the
final champion because this is one 28-day evaluation window and four hierarchy
levels regress. The next gate is rolling-origin stability by level, followed by
an independently modeled aggregate forecast and weighted reconciliation. A
weighted projection of an already coherent bottom-up forecast would be identical
and would not constitute a meaningful reconciliation comparison.
