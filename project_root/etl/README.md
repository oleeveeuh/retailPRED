# ETL status

The files in this directory implement the preserved v1/undergrad workflow. They remain readable so the historical project can be audited, but they are not the default v2 data path.

For v2:

- fetch native-monthly Census observations with `python -m scripts.fetch_mrts_v2`;
- validate and version them with `forecasting.data.monthly_dataset`;
- construct causal features with `forecasting.features.monthly`; and
- evaluate them with `forecasting.evaluation.backtest`.

`build_multi_resolution_dataset.py` requires `--allow-legacy-v1` because it interpolates monthly targets to daily frequency. `fetch_mrts.py` no longer calls its historical random-data generator when Census requests fail.

The tracked `project_root/data_multi_resolution/` files are frozen v1 evidence, not v2 training data.
