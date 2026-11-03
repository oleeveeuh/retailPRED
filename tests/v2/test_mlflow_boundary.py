import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from forecasting.tracking.mlflow import log_artifact_to_run, log_backtest_run


class _Run:
    info = SimpleNamespace(
        run_id="run-123",
        experiment_id="experiment-7",
        artifact_uri="artifact://run-123",
    )

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


def test_mlflow_boundary_returns_portable_run_manifest(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, object]] = []
    client = SimpleNamespace(
        log_artifact=lambda run_id, artifact, artifact_path: calls.append(
            ("client_artifact", (run_id, artifact, artifact_path))
        )
    )
    fake = SimpleNamespace(
        data=SimpleNamespace(
            from_pandas=lambda *_args, **_kwargs: "dataset"
        ),
        set_tracking_uri=lambda value: calls.append(("tracking_uri", value)),
        set_experiment=lambda value: calls.append(("experiment", value)),
        start_run=lambda **_kwargs: _Run(),
        set_tags=lambda value: calls.append(("tags", value)),
        log_params=lambda value: calls.append(("params", value)),
        log_metrics=lambda value: calls.append(("metrics", value)),
        log_input=lambda *_args, **_kwargs: None,
        log_artifact=lambda *_args, **_kwargs: None,
        MlflowClient=lambda: client,
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    folds = pd.DataFrame({"actual": [10.0, 20.0], "prediction": [11.0, 19.0]})
    runtime = tmp_path / "runtime.json"
    manifest = log_backtest_run(
        folds,
        experiment_name="retailpred-test",
        run_name="test-run",
        tracking_uri="sqlite:///test.mlflow.db",
        data_version="sha256:test",
        code_revision="abc123",
        model_metadata={"name": "test", "version": "1", "parameters": {}},
        runtime_metadata={"python": "3.11"},
    )
    assert manifest == {
        "run_id": "run-123",
        "experiment_id": "experiment-7",
        "artifact_uri": "artifact://run-123",
        "tracking_uri": "sqlite:///test.mlflow.db",
    }
    evidence = tmp_path / "summary.json"
    evidence.write_text("{}")
    log_artifact_to_run(
        run_id=manifest["run_id"],
        tracking_uri=manifest["tracking_uri"],
        artifact=evidence,
        artifact_path="evidence",
    )
    assert calls[-1][0] == "client_artifact"
