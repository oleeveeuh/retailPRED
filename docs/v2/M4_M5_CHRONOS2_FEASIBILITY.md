# M5 Chronos-2 feasibility gate

Date: 2026-11-03

## Decision

Chronos-2 passes the predeclared bounded gate and advances to the complete M5
benchmark. This is not a promotion decision: the feasibility score covers a
fixed 600-series subset and is not the official 12-level WRMSSE.

| Model | Revenue-weighted bottom RMSSE |
|---|---:|
| Chronos-2 zero-shot | 0.928768 |
| Global LightGBM | 0.958569 |
| Seasonal Naive | 1.151850 |

Chronos-2 improves on subset-trained LightGBM by 3.11% and Seasonal Naive by
19.37%. It therefore clears the declared requirements of at least 5% improvement
over Seasonal Naive and any positive improvement over the incumbent.

## Frozen design

- Select the top 20 recent-revenue series in each of the 30 store/category
  strata: 600 of 30,490 bottom series.
- Use the same selection, history through `d_1913`, and evaluation horizon
  `d_1914`–`d_1941` as the NHITS gate.
- Evaluate the package's `predictions` point output after clipping negative
  demand to zero.
- Run `amazon/chronos-2` with `chronos-forecasting==2.3.2`, PyTorch 2.15.0,
  1,913 days of context, batch size 32, CPU execution, and cross-learning off.
- Perform zero fine-tuning, hyperparameter search, or evaluation-window
  adaptation.

The subset metric is revenue-weighted bottom-level RMSSE. It is intentionally
useful for a bounded comparison but must not be described as official M5
WRMSSE because it omits hierarchy levels 1–11.

## Resource result

- Model loading: 19.19 seconds on the first cached download run.
- Inference: 13.31 seconds for 600 series.
- Linear 30,490-series inference estimate: 0.188 hours (about 11.3 minutes).
- Peak isolated process memory: 1,373.5 MB.

The measured runtime clears the declared two-hour full-scale limit. Model files
are cached under ignored `data/v2/`; source data, predictions, and model weights
are not committed.

## Interpretation

This result supports the distinct portfolio hypothesis behind the model suite:
a pretrained zero-shot model can add accuracy without task-specific fitting.
The gain is modest relative to LightGBM on the selected subset, so the full
official hierarchy benchmark is necessary before any promotion. If the full
holdout passes, rolling-origin stability remains required because one final
window cannot establish robustness.

## Reproduction

```bash
make setup-foundation
make feasibility-chronos-m5
```

The generated summary records source hashes, code revision, package versions,
selection details, scores, runtime, and gate outcomes.
