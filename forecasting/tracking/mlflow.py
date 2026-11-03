"""MLflow logging boundary kept optional for the core test environment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from forecasting.evaluation.metrics import bias, mae, wape


def log_backtest_run(
    folds: pd.DataFrame,
    *,
    experiment_name: str,
    run_name: str,
    tracking_uri: str,
    data_version: str,
    code_revision: str,
    model_metadata: dict[str, Any],
    runtime_metadata: dict[str, Any],
    artifacts: tuple[Path, ...] = (),
) -> str:
    """Log immutable inputs, parameters, metrics, runtime, and artifacts."""
    try:
        import mlflow
    except ImportError as error:
        raise RuntimeError(
            "MLflow tracking requires: pip install -r requirements-experiment.txt"
        ) from error

    required = {"actual", "prediction"}
    if missing := required.difference(folds.columns):
        raise ValueError(f"fold data is missing columns: {sorted(missing)}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name, log_system_metrics=True) as run:
        mlflow.set_tags(
            {
                "data_version": data_version,
                "code_revision": code_revision,
                "model_name": model_metadata["name"],
                "model_version": model_metadata["version"],
            }
        )
        parameters = model_metadata.get("parameters", {})
        mlflow.log_params({key: str(value) for key, value in parameters.items()})
        mlflow.log_metrics(
            {
                "wape": wape(folds["actual"], folds["prediction"]),
                "mae": mae(folds["actual"], folds["prediction"]),
                "bias": bias(folds["actual"], folds["prediction"]),
                "fold_rows": float(len(folds)),
            }
        )
        dataset = mlflow.data.from_pandas(
            folds,
            source=f"retailpred://{data_version}",
            name="rolling_origin_folds",
        )
        mlflow.log_input(dataset, context="evaluation")
        metadata_path = Path("data/v2/experiment_runtime.json")
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(runtime_metadata, indent=2, sort_keys=True) + "\n"
        )
        mlflow.log_artifact(str(metadata_path), artifact_path="metadata")
        for artifact in artifacts:
            mlflow.log_artifact(str(artifact), artifact_path="evidence")
        return run.info.run_id
