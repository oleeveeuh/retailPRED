# M5 global LightGBM challenger

Date: 2026-11-02  
Status: promoted as the current M5 point-forecast champion after rolling-origin audit

## Result

The direct multi-horizon global LightGBM scores **0.813301 WRMSSE**, improving
the frozen weekly Seasonal Naive baseline of 0.869701 by **6.48%**.

| Level | Grain | Seasonal Naive | LightGBM | Relative change |
|---:|---|---:|---:|---:|
| 1 | Total | 0.751430 | 0.742380 | +1.20% |
| 2 | State | 0.754396 | 0.746982 | +0.98% |
| 3 | Store | 0.799304 | 0.795100 | +0.53% |
| 4 | Category | 0.762047 | 0.764824 | -0.36% |
| 5 | Department | 0.817086 | 0.821928 | -0.59% |
| 6 | State × category | 0.782983 | 0.789547 | -0.84% |
| 7 | State × department | 0.830436 | 0.835649 | -0.63% |
| 8 | Store × category | 0.839263 | 0.826314 | +1.54% |
| 9 | Store × department | 0.891855 | 0.860916 | +3.47% |
| 10 | Item | 1.032248 | 0.875262 | +15.21% |
| 11 | Item × state | 1.074959 | 0.857518 | +20.23% |
| 12 | Item × store | 1.100402 | 0.843189 | +23.37% |

Positive change means lower WRMSSE. LightGBM improves eight of 12 levels and
delivers its largest gains at the sparse operational grains. Four middle
aggregate levels regress by less than 0.9% in this holdout. The subsequent
five-fold audit improves overall WRMSSE in every fold, supporting promotion as
the current M5 point champion.

## Experimental design

- One pooled Tweedie LightGBM learns all 30,490 item/store series and all 28
  horizons directly.
- Six non-overlapping training origins, `d_1717` through `d_1857`, produce
  5,122,320 training rows.
- `d_1885` is a training-only validation origin with 853,720 rows for early
stopping. Training stops at iteration 69 of a maximum 500.
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

The complete local run took 26.41 seconds on Apple arm64 with Python 3.11.17 and
LightGBM 4.5.0; model fitting took 17.89 seconds. Source file hashes match the M5
source validation report.

Code revision: `18617afa1ad91ade00c9bb4040dcd640984d3640`

Reproduce with:

```bash
make benchmark-m5
```

## Promotion decision

The five-fold rolling-origin audit improves overall WRMSSE in every window, with
a 6.27% worst-case and 13.77% mean improvement. LightGBM is therefore promoted
over Seasonal Naive as the current M5 point champion. See
`docs/v2/M4_M5_STABILITY.md`.

The next challenger is an independently modeled aggregate forecast followed by
weighted reconciliation. A weighted projection of an already coherent bottom-up
forecast would be identical and would not constitute a meaningful comparison.
