# M5 probabilistic evaluation

Date: 2026-11-03

Status: measured; interval policy remains a research output

## Decision

Do not promote the first M5 uncertainty policy. It achieves nearly exact
overall 80% coverage and acceptable coverage at every hierarchy level, but it
misses the predeclared minimum for high-volume bottom series and varies too
sharply by forecast horizon. The point LightGBM champion is unchanged.

## Leakage-safe design

- Center intervals on the coherent global LightGBM point forecasts.
- Use four complete, earlier rolling windows ending at origins `d_1801`,
  `d_1829`, `d_1857`, and `d_1885` for calibration.
- Reserve `d_1914`–`d_1941` exclusively for evaluation.
- Normalize absolute calibration errors by each node's training-only RMSSE
  scale.
- Estimate separate finite-sample conformal adjustments for every hierarchy
  level and forecast horizon.
- Derive P10/P50/P90 as the nonnegative lower bound, coherent point forecast,
  and upper bound. Bounds are node-specific uncertainty statements; quantile
  additivity is not claimed.

This avoids the invalid shortcut of summing bottom-series quantiles, because a
quantile of a sum is generally not the sum of component quantiles.

## Predeclared gate

- Overall weighted coverage: 77–83%.
- Coverage at each hierarchy level: 70–90%.
- Coverage for every bottom demand segment: at least 70%.

The policy passes the first two conditions and fails the third. Thresholds are
not changed after observing the result.

## Overall result

| Metric | Result |
|---|---:|
| Nominal central coverage | 80.00% |
| Official-weighted coverage | 80.38% |
| Mean RMSSE-scaled width | 2.1053 |
| Scale-normalized WIS | 0.4194 |
| Center point WRMSSE | 0.813301 |

Reproducing the champion WRMSSE exactly verifies that interval evaluation did
not alter the point model.

## Hierarchy coverage

All levels satisfy the declared range. Coverage spans 71.43% at the single
total node to 88.10% at category. The largest operational levels are:

| Level | Coverage | Scaled width | Normalized WIS |
|---|---:|---:|---:|
| Item | 80.23% | 2.0623 | 0.4344 |
| Item/state | 79.81% | 1.6997 | 0.4003 |
| Item/store | 78.90% | 1.2740 | 0.3701 |

## Demand-segment result

| Segment | Series | Coverage |
|---|---:|---:|
| High recent volume | 5,043 | **69.38%** |
| Medium recent volume | 11,808 | 75.78% |
| Low recent volume | 10,491 | 84.47% |
| Very low recent volume | 3,148 | 87.24% |
| Regular demand | 11,549 | 70.78% |
| Mixed demand | 15,168 | 83.13% |
| Intermittent demand | 3,773 | 86.73% |

The high-volume result fails the 70% floor. Intervals are conservative for
sparse demand and too narrow for the most commercially important series.

## Horizon stability

Coverage is not stable despite the strong aggregate result. It ranges from
93.87% at horizon 2 to 38.95% at horizon 21. Horizon 21 also has the worst
normalized WIS at 0.9871. This indicates distribution shift or calendar/event
effects not captured by four origin-local residual samples; averaging across
28 horizons would hide the failure.

## Runtime and artifacts

The five-fold evaluation finished in 116.73 seconds on Apple arm64 with
LightGBM 4.5.0. Ignored evidence includes calibration adjustments, fold
metadata, interval forecasts, and breakdowns by level, horizon, volume, and
intermittency. Source hashes and code revision are recorded in the summary.

## Next research step

Do not tune on this final window. A future interval challenger should be
declared on new rolling origins and may test more calibration folds,
calendar-conditioned residual groups, or distributional count models. Until
then, the serving layer must label M5 intervals unavailable rather than imply
calibrated uncertainty.

## Reproduction

```bash
make probabilistic-m5
```
