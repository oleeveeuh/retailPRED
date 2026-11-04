# Milestone 3 Probabilistic Development Benchmark

Status: measured; native and conformal intervals remain research challengers

## Evaluation contract

- Source and data hash: the same Census 2026 revision workbook used by the point benchmarks (`e1243966e8e40ea1b99ddb7158f2224ba91147bc40b7a47c2ab7b398077d8796`).
- Code revision: `7649fc49857bdf3afc9866ea64e1516dbe3d9749`.
- Native forecasts: independently trained LightGBM P10, P50, and P90 objectives with monotone rearrangement to prevent quantile crossing.
- Calibration: multiplicative conformalized quantile regression, grouped by series and horizon so dollar scale and volatility are not pooled incorrectly.
- Calibration partition: 2,398 forecast rows whose targets end by December 2021.
- Evaluation partition: 1,826 forecast rows whose targets begin in January 2022. No realized target appears in both partitions.
- Nominal central interval coverage: 80%.
- Evaluation: coverage, relative width, pinball loss, and normalized weighted interval score (WIS) are considered together.
- Runtime: 81.30 seconds on Python 3.11.17, LightGBM 4.5.0, Apple arm64.

## Overall result

| Method | 80% coverage | Mean relative width | Normalized WIS |
|---|---:|---:|---:|
| Native P10–P90 | **86.0%** | **19.3%** | **0.0192** |
| Conformalized P10–P90 | 92.4% | 27.8% | 0.0225 |

The native interval is already conservative overall. Conformalization increases coverage by 6.4 percentage points, but moves farther from the 80% target, widens intervals materially, and worsens WIS by 17.0%. It is therefore rejected as the default.

## Results by horizon

| Horizon | Native coverage | Conformal coverage | Native relative width | Conformal relative width | Native normalized WIS | Conformal normalized WIS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 month | 67.6% | 86.2% | 4.4% | 6.4% | 0.0079 | **0.0078** |
| 3 months | 85.3% | 92.3% | 10.7% | 13.8% | **0.0122** | 0.0132 |
| 6 months | 91.6% | 95.9% | 20.5% | 31.1% | **0.0201** | 0.0243 |
| 12 months | 96.0% | 94.1% | 36.9% | 52.8% | **0.0329** | 0.0398 |

The problem is horizon-dependent: native one-month intervals under-cover, while six- and twelve-month intervals substantially over-cover. Conformalization helps the one-month interval but does not provide a globally acceptable solution. Selecting a different method per horizon after inspecting these evaluation results would itself be evaluation leakage, so no interval configuration is promoted from this run.

## Decision

- Keep native LightGBM quantiles as a measured research output.
- Do not expose either interval method as production-calibrated uncertainty.
- Preserve the point LightGBM challenger result separately; this calibration result does not change its point metrics.
- Next evaluate rolling, origin-local calibration and direct multi-horizon models using a new, predeclared evaluation protocol.
- Continue reporting coverage and width together; coverage alone would reward unusably broad intervals.

This result is development evidence on revised history, not a distribution-free guarantee for future releases. COVID-era calibration errors and post-2021 evaluation behavior are not exchangeable, which is a plausible reason fixed split conformal calibration over-corrects later intervals.

## Reproduce

```bash
make probabilistic-mrts \
  WORKBOOK=/absolute/path/to/benchsales26.xlsx \
  VINTAGE_TIME=2026-11-02T00:00:00+00:00
```

The fold-level evaluation file and machine-readable summary are written beneath ignored `data/v2/`.
