# M5 NHITS feasibility gate

Date: 2026-11-02  
Status: bounded gate completed; full-scale benchmark not recommended

## Decision

NHITS demonstrates value over a simple baseline but does not beat the incumbent
model on the predeclared subset:

| Model | Revenue-weighted subset RMSSE |
|---|---:|
| Global LightGBM | **0.958569** |
| NHITS | 1.059983 |
| Weekly Seasonal Naive | 1.151850 |

NHITS improves Seasonal Naive by 7.98% but is 10.58% worse than the
subset-trained global LightGBM. The gate therefore stops before a full 30,490-
series run. The result does not justify replacing or fully benchmarking the
promoted LightGBM champion.

## Declared subset

The experiment selects the 20 highest-revenue item/store series from each of the
30 store/category strata, producing 600 series with complete business-grain
coverage. Revenue is measured only over `d_1886–d_1913`; evaluation uses
`d_1914–d_1941`.

The score is revenue-weighted bottom-level RMSSE normalized within this subset.
It is explicitly **not** the official 12-level M5 WRMSSE and is used only for the
feasibility decision.

## NHITS configuration

- NeuralForecast 3.3.0 official NHITS implementation.
- Direct 28-day forecast with 112-day input context.
- 730 days of available training history per series.
- Three hierarchical interpolation stacks with 64-unit hidden layers.
- MSE loss, robust per-series scaling, deterministic seed 42.
- 100 optimization steps on CPU.
- Prior 28 days reserved for validation inside NeuralForecast.

The official implementation describes NHITS as a direct, univariate global model
with historical, future, static, and categorical exogenous support. This bounded
gate intentionally tests the autoregressive core before paying the complexity
cost of feature parity with LightGBM.

## Engineering evidence

- Neural training and inference: 0.60 seconds after process startup.
- Complete data preparation, both models, scoring, and artifact generation: 7.91
  seconds.
- Peak process memory: 1,314 MB.
- A separate `.venv-neural` prevents PyTorch/Ray dependencies from destabilizing
  the core and MLflow environments.
- NHITS runs in a clean child process because LightGBM and PyTorch training in the
  same macOS process produced an OpenMP runtime conflict.
- The machine's PyTorch build reported CPU only; no MPS or GPU acceleration is
  claimed.

The linear training-time extrapolation was small, but it is not used to approve
full scale because accuracy fails first and memory/data-loader behavior is not
guaranteed to scale linearly.

## Reproduction

```bash
make setup-neural
make feasibility-nhits-m5
```

Code revision: `fe69aa0e37a7beed8eff8155cc82915104f0f251`

## Model-suite implication

NHITS remains a documented neural reference rather than a promoted M5 model. No
hyperparameters will be tuned against this evaluation subset. A future NHITS
attempt would require a new untouched window and a predeclared exogenous-feature
design. The next distinct challenger is the zero-shot foundation-model gate,
which tests a different hypothesis rather than tuning this failed one.
