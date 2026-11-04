"""Internal isolated-process NHITS trainer used by the M5 feasibility gate."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import pandas as pd
import torch
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MSE
from neuralforecast.models import NHITS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--input-size", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    frame = pd.read_parquet(args.train)
    model = NHITS(
        h=28,
        input_size=args.input_size,
        loss=MSE(),
        mlp_units=3 * [[64, 64]],
        max_steps=args.max_steps,
        learning_rate=1e-3,
        val_check_steps=min(20, args.max_steps),
        early_stop_patience_steps=3,
        batch_size=64,
        valid_batch_size=64,
        windows_batch_size=128,
        inference_windows_batch_size=128,
        scaler_type="robust",
        random_seed=42,
        alias="nhits",
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        dataloader_kwargs={"num_workers": 0},
    )
    neural = NeuralForecast(models=[model], freq="D")
    started = time.perf_counter()
    neural.fit(df=frame, val_size=28)
    forecast = neural.predict()
    elapsed = time.perf_counter() - started
    forecast.to_parquet(args.predictions, index=False)
    args.metadata.write_text(
        json.dumps(
            {
                "elapsed_seconds": elapsed,
                "peak_process_memory_mb": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss
                / (1024 * 1024),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
