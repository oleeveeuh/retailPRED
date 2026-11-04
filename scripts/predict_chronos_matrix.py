"""Run isolated Chronos-2 inference for a series-by-time NumPy matrix."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from chronos import BaseChronosPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("context", type=Path)
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
    context = np.load(args.context, mmap_mode="c")
    if context.ndim != 2 or not np.isfinite(context).all():
        raise ValueError("context must be a finite series-by-time matrix")
    load_started = time.perf_counter()
    pipeline = BaseChronosPipeline.from_pretrained(
        args.model_id,
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    load_seconds = time.perf_counter() - load_started
    inference_started = time.perf_counter()
    _, point = pipeline.predict_quantiles(
        inputs=context[:, np.newaxis, :],
        prediction_length=args.prediction_length,
        quantile_levels=[0.1, 0.5, 0.9],
        batch_size=args.batch_size,
        context_length=context.shape[1],
        cross_learning=False,
    )
    prediction = torch.stack(point).numpy()[:, 0, :]
    inference_seconds = time.perf_counter() - inference_started
    if prediction.shape != (context.shape[0], args.prediction_length):
        raise RuntimeError("Chronos-2 returned an unexpected prediction shape")
    np.save(args.predictions, prediction.astype(np.float32, copy=False))
    args.metadata.write_text(
        json.dumps(
            {
                "batch_size": args.batch_size,
                "chronos_forecasting": version("chronos-forecasting"),
                "context_length": int(context.shape[1]),
                "cross_learning": False,
                "inference_seconds": inference_seconds,
                "load_seconds": load_seconds,
                "model_id": args.model_id,
                "peak_process_memory_mb": _peak_memory_mb(),
                "series": int(context.shape[0]),
                "torch": torch.__version__,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
