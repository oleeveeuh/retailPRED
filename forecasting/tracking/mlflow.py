"""MLflow logging boundary kept optional for the core test environment."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from forecasting.evaluation.metrics import bias, mae, wape


def prepare_tracking(*, tracking_uri: str, experiment_name: str) -> None:
    """Fail fast on missing dependencies or an invalid tracking backend."""
    try:
        import mlflow
    except ImportError as error:
        raise RuntimeError(
            "MLflow tracking requires: pip install -r requirements-experiment.txt"
        ) from error
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


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
) -> dict[str, str]:
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
    prepare_tracking(tracking_uri=tracking_uri, experiment_name=experiment_name)
    # Runtime/hardware metadata is logged explicitly below. MLflow's optional
    # system monitor requires psutil and is deliberately not a core dependency.
    with mlflow.start_run(run_name=run_name, log_system_metrics=False) as run:
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
            name="rolling_origin_folds",
        )
        mlflow.log_input(dataset, context="evaluation")
        with tempfile.TemporaryDirectory(prefix="retailpred-mlflow-") as directory:
            metadata_path = Path(directory) / "experiment_runtime.json"
            metadata_path.write_text(
                json.dumps(runtime_metadata, indent=2, sort_keys=True) + "\n"
            )
            mlflow.log_artifact(str(metadata_path), artifact_path="metadata")
        for artifact in artifacts:
            mlflow.log_artifact(str(artifact), artifact_path="evidence")
        return {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "artifact_uri": run.info.artifact_uri,
            "tracking_uri": tracking_uri,
        }


def log_artifact_to_run(
    *, run_id: str, tracking_uri: str, artifact: Path, artifact_path: str
) -> None:
    """Attach an artifact created after the fluent run context has closed."""
    try:
        import mlflow
    except ImportError as error:
        raise RuntimeError(
            "MLflow tracking requires: pip install -r requirements-experiment.txt"
        ) from error
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.MlflowClient().log_artifact(
        run_id, str(artifact), artifact_path=artifact_path
    )
