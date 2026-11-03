# Milestone 3 Global LightGBM Development Benchmark

Status: measured challenger; numerical promotion gate passed, final champion designation pending untouched evidence

## Evaluation contract

- Source: the same U.S. Census Bureau 2026 annual-revision workbook used for the Milestone 1 baseline.
- Source URL: <https://www.census.gov/retail/mrts/www/benchmark/2026/excel/benchsales26.xlsx>
- File SHA-256: `e1243966e8e40ea1b99ddb7158f2224ba91147bc40b7a47c2ab7b398077d8796`.
- Code revision: `97d3a132f7043cd44e557ec811663c781c1bba46`.
- Data: 11 seasonally adjusted monthly series, January 2010 through December 2025.
- Folds: expanding windows beginning at 60 months; monthly origins; horizons 1, 3, 6, and 12 months.
- Rows per model: 5,324. The panel evaluator verifies identical series, origins, targets, and horizons.
- Model: global LightGBM v2.0.0, 250 trees, deterministic seed 42 and single-threaded fitting.
- Target: `log(value / lag_12)`, making growth comparable across categories with very different dollar scales.
- Inputs: causal normalized lags, shifted rolling summaries, prior change, calendar seasonality, and static series identity. Recursive horizons use earlier predictions, never later actuals.
- Runtime: 26.24 seconds for all LightGBM folds on Python 3.11.17, LightGBM 4.5.0, Apple arm64.

## Overall result

| Model | Fold rows | Aggregate WAPE | Mean category WAPE | MAE (USD millions) | Bias (USD millions) |
|---|---:|---:|---:|---:|---:|
| Seasonal Naive | 5,324 | 5.39% | 6.35% | 4,403.8 | -3,235.9 |
| Global LightGBM | 5,324 | **3.15%** | **4.49%** | **2,575.1** | **-512.3** |

LightGBM reduces aggregate WAPE by 41.5% and the equally weighted mean category WAPE by 29.2%. Aggregate WAPE is influenced by the largest series, so category-level results remain the primary diagnostic.

## Results by category

| Series | Seasonal Naive WAPE | LightGBM WAPE | Relative improvement |
|---|---:|---:|---:|
| Retail total | 4.81% | **2.41%** | 49.8% |
| Motor vehicle and parts | 5.87% | **4.55%** | 22.4% |
| Furniture and home furnishings | 6.56% | **5.68%** | 13.3% |
| Electronics and appliances | 7.41% | **6.08%** | 18.0% |
| Building material and garden | 5.64% | **3.30%** | 41.5% |
| Food and beverage | 4.08% | **1.66%** | 59.4% |
| Health and personal care | 4.74% | **2.13%** | 55.1% |
| Gasoline stations | 12.62% | **8.69%** | 31.2% |
| Clothing and accessories | 7.94% | **7.00%** | 11.8% |
| Sporting goods and hobby | 5.91% | **5.73%** | 3.0% |
| General merchandise | 4.23% | **2.18%** | 48.5% |

The challenger improves all 11 categories, although sporting goods is effectively close and gasoline remains the hardest category.

## Results by horizon

| Horizon | Seasonal Naive WAPE | LightGBM WAPE | Relative improvement |
|---:|---:|---:|---:|
| 1 month | 5.45% | **1.79%** | 67.2% |
| 3 months | 5.42% | **2.64%** | 51.2% |
| 6 months | 5.39% | **3.50%** | 35.1% |
| 12 months | 5.32% | **4.63%** | 13.1% |

The advantage narrows with recursive distance, which is expected because predicted values feed later lag features. Seasonal Naive remains competitive at 12 months and stays in every result table.

## Promotion interpretation and limitations

The declared numerical policy requires at least 2% relative WAPE improvement and a runtime below 900 seconds. LightGBM passes both. It is nevertheless labeled a measured challenger rather than the final champion because the revised-history development backtest informed the scale-free target design. A final designation requires point-in-time vintages and an evaluation period or dataset not used for model design.

Other limitations:

- This is revised Census history, not a real-time-vintage backtest.
- The 11 national aggregates are a small panel and do not establish store/SKU performance.
- No economic covariates are included yet; unreleased observations are explicitly excluded by the covariate contract.
- Quantile and conformal calibration evidence is reported separately in `M3_PROBABILISTIC.md`; neither interval method is promoted for serving.
- MLflow logging has been validated end to end against a local SQLite backend; the verified run manifest is recorded in `MLFLOW_RUN.md`. A durable remote run still requires a configured workspace.

## Reproduce

```bash
make benchmark-mrts \
  WORKBOOK=/absolute/path/to/benchsales26.xlsx \
  VINTAGE_TIME=2026-11-02T00:00:00+00:00
```

Fold rows and the machine-readable summary are written under ignored `data/v2/`. Add `--tracking-uri` when running the script directly to record the data hash, Git revision, model parameters, metrics, runtime metadata, folds, and evidence artifacts in MLflow. MLflow's tracking API records parameters, code versions, metrics, datasets, and artifacts ([official documentation](https://mlflow.org/docs/latest/tracking/)).
