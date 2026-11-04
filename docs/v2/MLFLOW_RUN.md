# MLflow Tracking Validation

Status: completed against an ignored local SQLite backend

The complete MRTS LightGBM comparison was recorded through MLflow 3.16.1 after the untracked unit boundary had passed. This validates the same tracking calls intended for a Databricks or hosted MLflow backend without publishing a local database or generated fold rows.

## Verified run manifest

| Field | Value |
|---|---|
| Run status | `FINISHED` |
| Run ID | `2374564ed5aa419abd1dcd6bda928c13` |
| Experiment ID | `1` (`retailpred-mrts`) |
| Code revision | `5410fc6abdd971dfbba66772bf1c38c227b015bc` |
| Model | `global_lightgbm` v2.0.0 |
| Census data SHA-256 | `e1243966e8e40ea1b99ddb7158f2224ba91147bc40b7a47c2ab7b398077d8796` |
| MLflow dataset digest | `8b44c7bf` |
| Dataset context | `evaluation` |
| Fold rows | 5,324 |
| WAPE | 3.1547% |
| MAE | 2,575.10 USD millions |
| Bias | -512.25 USD millions |

The stored run was read back through `MlflowClient`, not inferred from console output. Verification confirmed the immutable data/code tags, metrics, evaluation dataset input, and these artifacts:

- `evidence/mrts_model_folds.csv`
- `evidence/mrts_model_summary.json`
- `metadata/experiment_runtime.json`

## Reproduce locally

Use a dedicated environment because MLflow 3.16.1 currently requires `pyarrow<26`, while RetailPRED's core verification lock tests PyArrow 26.

```bash
make setup-experiment
make track-mrts \
  WORKBOOK=/absolute/path/to/benchsales26.xlsx \
  VINTAGE_TIME=2026-11-02T00:00:00+00:00 \
  TRACKING_URI=sqlite:////absolute/path/to/retailpred.mlflow.db
```

The local database, artifact store, source workbook, and fold rows are ignored. Therefore this run ID is evidence that the integration executed, not a public link. A durable externally inspectable run still requires the owner's Databricks or hosted MLflow workspace.
