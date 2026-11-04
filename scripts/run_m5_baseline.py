"""Run the fixed 28-day M5 Seasonal Naive hierarchy benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting.evaluation.m5_backtest import (
    hierarchy_rmsse_scales,
    official_m5_weights,
    recent_bottom_revenue,
    score_hierarchical_forecast,
    seasonal_naive_bottom_forecast,
)
from forecasting.reconciliation.m5 import M5Hierarchy
from retailpred.platform.m5_pipeline import M5_ID_COLUMNS, file_sha256


TRAIN_END_DAY = 1_913
HORIZON = 28
WEIGHT_WINDOW = 28
SEASONAL_PERIOD = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_directory", type=Path, nargs="?", default=Path("data/v2/source/m5")
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("data/v2/evidence/m5_seasonal_naive"),
    )
    return parser.parse_args()


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    args = parse_args()
    source = args.source_directory.resolve()
    sales_path = source / "sales_train_evaluation.csv"
    calendar_path = source / "calendar.csv"
    prices_path = source / "sell_prices.csv"
    train_days = [f"d_{day}" for day in range(1, TRAIN_END_DAY + 1)]
    evaluation_days = [
        f"d_{day}" for day in range(TRAIN_END_DAY + 1, TRAIN_END_DAY + HORIZON + 1)
    ]
    day_columns = train_days + evaluation_days
    dtypes: dict[str, object] = {column: "float32" for column in day_columns}
    dtypes.update({column: "string" for column in M5_ID_COLUMNS})
    sales = pd.read_csv(
        sales_path,
        usecols=list(M5_ID_COLUMNS) + day_columns,
        dtype=dtypes,
    )
    bottom = sales.loc[:, M5_ID_COLUMNS].rename(columns={"id": "series_id"})
    hierarchy = M5Hierarchy.from_bottom(bottom)
    hierarchy.validate_official_shape()
    ordered_sales = sales.set_index("id").reindex(hierarchy.bottom["series_id"])
    history = ordered_sales.loc[:, train_days].to_numpy(dtype=np.float64, copy=False)
    bottom_actual = ordered_sales.loc[:, evaluation_days].to_numpy(
        dtype=np.float64, copy=False
    )

    scales = hierarchy_rmsse_scales(hierarchy, history)
    recent_days = train_days[-WEIGHT_WINDOW:]
    revenue = recent_bottom_revenue(
        bottom=hierarchy.bottom,
        recent_units=history[:, -WEIGHT_WINDOW:],
        recent_days=recent_days,
        calendar_path=calendar_path,
        prices_path=prices_path,
    )
    weights = official_m5_weights(hierarchy, revenue)
    bottom_prediction = seasonal_naive_bottom_forecast(
        history, horizon=HORIZON, seasonal_period=SEASONAL_PERIOD
    )
    actual = hierarchy.summing_matrix @ bottom_actual
    prediction = hierarchy.summing_matrix @ bottom_prediction
    result = score_hierarchical_forecast(
        hierarchy,
        actual=actual,
        prediction=prediction,
        scales=scales,
        weights=weights,
    )

    args.output_directory.mkdir(parents=True, exist_ok=True)
    level_path = args.output_directory / "m5_seasonal_naive_by_level.csv"
    summary_path = args.output_directory / "m5_seasonal_naive_summary.json"
    result.by_level.to_csv(level_path, index=False)
    summary = {
        "benchmark": "weekly_seasonal_naive_bottom_up",
        "code_revision": _git_revision(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_days": [evaluation_days[0], evaluation_days[-1]],
        "hierarchy_nodes": len(hierarchy.nodes),
        "horizon": HORIZON,
        "overall_wrmsse": result.overall_wrmsse,
        "seasonal_period": SEASONAL_PERIOD,
        "source_sha256": {
            path.name: file_sha256(path)
            for path in (calendar_path, sales_path, prices_path)
        },
        "train_days": [train_days[0], train_days[-1]],
        "weight_days": [recent_days[0], recent_days[-1]],
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(result.by_level.to_string(index=False))
    print(f"Wrote {summary_path} and {level_path}")


if __name__ == "__main__":
    main()
