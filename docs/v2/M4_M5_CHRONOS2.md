# M5 Chronos-2 official benchmark

Date: 2026-11-03

## Decision

Do not promote zero-shot Chronos-2 and do not run the rolling stability suite.
It passes the bounded high-revenue subset gate but fails the complete official
M5 holdout by a wide margin. No hyperparameters or calibration factors will be
tuned against this final window.

| Model | Official WRMSSE | Relative to Seasonal Naive |
|---|---:|---:|
| Global LightGBM | 0.813301 | +6.48% |
| Seasonal Naive | 0.869701 | reference |
| Chronos-2 zero-shot | 1.937468 | -122.77% |

Chronos-2 is 138.22% worse than the incumbent global LightGBM. Its earlier
600-series score was a valid feasibility result, but the revenue-stratified
selection was not representative of the complete 30,490-series hierarchy.

## Frozen benchmark

- Model: `amazon/chronos-2` through `chronos-forecasting==2.3.2`.
- Zero fine-tuning, covariates, hyperparameter search, or holdout adaptation.
- Context: all 1,913 observations through `d_1913`.
- Horizon: `d_1914`–`d_1941`.
- Point output: package `predictions`, clipped at zero.
- Execution: CPU, batch size 32, cross-learning off.
- Scoring: the existing official 42,840-node, 12-level WRMSSE contract with
  scales and revenue weights ending at `d_1913`.

Source hashes match the incumbent LightGBM benchmark exactly.

## Hierarchy result

The overall failure is not uniform. Chronos-2 improves Seasonal Naive at the
two most disaggregated levels:

| Level | Chronos-2 | Seasonal Naive | Relative improvement |
|---|---:|---:|---:|
| Item/state | 0.941108 | 1.074959 | +12.45% |
| Item/store | 0.868109 | 1.100402 | +21.11% |

It is worse at every other level. The most aggregated examples are:

| Level | Chronos-2 | Seasonal Naive | Relative improvement |
|---|---:|---:|---:|
| Total | 2.476406 | 0.751430 | -229.56% |
| Category | 2.485081 | 0.762047 | -226.11% |
| Department | 2.482319 | 0.817086 | -203.80% |

The complete forecast contains 834,386 units against 1,231,764 actual units,
an aggregate ratio of 0.677. Daily ratios range from 0.581 to 0.794. Negative
raw predictions affect only 0.83% of cells and clipping adds about 21 units, so
the aggregate underforecast is not a clipping artifact.

This is a useful distinction: bottom-up forecasts are algebraically coherent,
but coherence does not guarantee accurate aggregate demand. Small systematic
biases across many sparse item/store series accumulate at high levels.

## Runtime

- Inference: 647.22 seconds for all 30,490 bottom series.
- End-to-end benchmark: 656.95 seconds.
- Peak isolated inference memory: 1,589 MB.
- Peak parent process memory: 1,656 MB.

The measured runtime is practical and close to the subset extrapolation. The
model is rejected for accuracy, not operational cost.

## Portfolio conclusion

Chronos-2 remains a documented foundation-model experiment, not an M5
production challenger. The result adds depth to the project because it shows a
predeclared funnel, complete hierarchy evaluation, and an evidence-based stop
decision rather than selective reporting. Global LightGBM remains the M5 point
champion.

Possible future research may use training-only aggregate calibration or a
hybrid model, but it must start with new rolling-origin folds and may not use
this final holdout for tuning.

## Reproduction

```bash
make setup-foundation
make feasibility-chronos-m5
make benchmark-chronos-m5
```
