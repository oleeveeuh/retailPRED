"""Internal isolated-process Chronos-2 predictor for the M5 feasibility gate."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import time
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import torch
from chronos import BaseChronosPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--model-id", default="amazon/chronos-2")
    parser.add_argument("--prediction-length", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def _peak_memory_mb() -> float:
    usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return usage / (1024 * 1024) if platform.system() == "Darwin" else usage / 1024


def main() -> None:
    args = parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    history = pd.read_parquet(args.history)
    load_started = time.perf_counter()
    pipeline = BaseChronosPipeline.from_pretrained(
        args.model_id,
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    load_seconds = time.perf_counter() - load_started
    inference_started = time.perf_counter()
    forecast = pipeline.predict_df(
        history,
        id_column="series_id",
        timestamp_column="timestamp",
        target="target",
        prediction_length=args.prediction_length,
        quantile_levels=[0.1, 0.5, 0.9],
        batch_size=args.batch_size,
        context_length=len(history) // history["series_id"].nunique(),
        cross_learning=False,
        freq="D",
    )
    inference_seconds = time.perf_counter() - inference_started
    forecast.to_parquet(args.predictions, index=False)
    args.metadata.write_text(
        json.dumps(
            {
                "batch_size": args.batch_size,
                "chronos_forecasting": version("chronos-forecasting"),
                "cross_learning": False,
                "inference_seconds": inference_seconds,
                "load_seconds": load_seconds,
                "model_id": args.model_id,
                "peak_process_memory_mb": _peak_memory_mb(),
                "torch": torch.__version__,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
