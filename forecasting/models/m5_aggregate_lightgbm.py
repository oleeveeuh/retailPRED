"""Independent direct forecasts for M5 hierarchy levels 1–9."""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


CATEGORICAL_FEATURES = (
    "node_code",
    "level_code",
    "event_type_1_code",
    "event_type_2_code",
)
FEATURE_COLUMNS = (
    "node_code",
    "level_code",
    "horizon",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "lag_56",
    "lag_364",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_mean_56",
    "rolling_std_28",
    "target_wday",
    "target_month",
    "target_year",
    "event_1",
    "event_2",
    "event_type_1_code",
    "event_type_2_code",
)


class M5AggregateFeatureBuilder:
    def __init__(self, *, nodes: pd.DataFrame, calendar: pd.DataFrame) -> None:
        if missing := {"node_id", "level"}.difference(nodes.columns):
            raise ValueError(f"aggregate nodes are missing columns: {sorted(missing)}")
        required_calendar = {
            "wday",
            "month",
            "year",
            "event_type_1",
            "event_type_2",
        }
        if missing := required_calendar.difference(calendar.columns):
            raise ValueError(f"calendar is missing columns: {sorted(missing)}")
        self.nodes = nodes.reset_index(drop=True).copy()
        self.calendar = calendar.reset_index(drop=True).copy()
        event_values = pd.concat(
            [self.calendar["event_type_1"], self.calendar["event_type_2"]],
            ignore_index=True,
        ).fillna("none")
        categories = sorted(event_values.astype(str).unique())
        self.event_codes = {
            column: pd.Categorical(
                self.calendar[column].fillna("none").astype(str),
                categories=categories,
            ).codes.astype(np.int8)
            for column in ("event_type_1", "event_type_2")
        }

    def build(
        self,
        history: np.ndarray,
        *,
        origin_index: int,
        horizons: tuple[int, ...],
        include_target: bool,
    ) -> tuple[pd.DataFrame, np.ndarray | None]:
        values = np.asarray(history, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] != len(self.nodes):
            raise ValueError("aggregate history must have one row per node")
        if origin_index < 363:
            raise ValueError("aggregate features require 364 observed days")
        if not horizons or min(horizons) < 1:
            raise ValueError("horizons must be positive")
        if origin_index + max(horizons) >= len(self.calendar):
            raise ValueError("calendar does not cover aggregate horizons")
        if include_target and origin_index + max(horizons) >= values.shape[1]:
            raise ValueError("aggregate history does not contain targets")
        lag_values = {
            lag: values[:, origin_index + 1 - lag]
            for lag in (1, 7, 14, 28, 56, 364)
        }
        recent_7 = values[:, origin_index - 6 : origin_index + 1]
        recent_28 = values[:, origin_index - 27 : origin_index + 1]
        recent_56 = values[:, origin_index - 55 : origin_index + 1]
        frames: list[pd.DataFrame] = []
        targets: list[np.ndarray] = []
        for horizon in horizons:
            target_index = origin_index + horizon
            row = self.calendar.iloc[target_index]
            frame = pd.DataFrame(
                {
                    "node_code": np.arange(len(self.nodes), dtype=np.int16),
                    "level_code": self.nodes["level"].to_numpy(dtype=np.int8) - 1,
                    "horizon": np.full(len(self.nodes), horizon, dtype=np.int8),
                    "lag_1": lag_values[1],
                    "lag_7": lag_values[7],
                    "lag_14": lag_values[14],
                    "lag_28": lag_values[28],
                    "lag_56": lag_values[56],
                    "lag_364": lag_values[364],
                    "rolling_mean_7": recent_7.mean(axis=1),
                    "rolling_mean_28": recent_28.mean(axis=1),
                    "rolling_mean_56": recent_56.mean(axis=1),
                    "rolling_std_28": recent_28.std(axis=1),
                    "target_wday": np.full(len(self.nodes), int(row["wday"]), dtype=np.int8),
                    "target_month": np.full(len(self.nodes), int(row["month"]), dtype=np.int8),
                    "target_year": np.full(len(self.nodes), int(row["year"]), dtype=np.int16),
                    "event_1": np.full(
                        len(self.nodes), int(pd.notna(row["event_type_1"])), dtype=np.int8
                    ),
                    "event_2": np.full(
                        len(self.nodes), int(pd.notna(row["event_type_2"])), dtype=np.int8
                    ),
                    "event_type_1_code": np.full(
                        len(self.nodes), self.event_codes["event_type_1"][target_index], dtype=np.int8
                    ),
                    "event_type_2_code": np.full(
                        len(self.nodes), self.event_codes["event_type_2"][target_index], dtype=np.int8
                    ),
                },
                columns=FEATURE_COLUMNS,
            )
            frames.append(frame)
            if include_target:
                targets.append(values[:, target_index])
        return (
            pd.concat(frames, ignore_index=True),
            np.concatenate(targets).astype(np.float32) if include_target else None,
        )


class M5AggregateLightGBM:
    def __init__(self, *, n_estimators: int = 500, random_state: int = 42) -> None:
        self.params: dict[str, Any] = {
            "objective": "tweedie",
            "tweedie_variance_power": 1.1,
            "n_estimators": n_estimators,
            "learning_rate": 0.04,
            "num_leaves": 31,
            "min_child_samples": 20,
            "max_bin": 63,
            "subsample": 0.9,
            "subsample_freq": 1,
            "colsample_bytree": 0.9,
            "reg_lambda": 0.1,
            "random_state": random_state,
            "bagging_seed": random_state,
            "feature_fraction_seed": random_state,
            "data_random_seed": random_state,
            "deterministic": True,
            "force_col_wise": True,
            "n_jobs": -1,
            "verbosity": -1,
        }
        self.model: LGBMRegressor | None = None

    def fit(
        self,
        x_train: pd.DataFrame,
        y_train: np.ndarray,
        *,
        x_validation: pd.DataFrame,
        y_validation: np.ndarray,
        early_stopping_rounds: int = 50,
    ) -> "M5AggregateLightGBM":
        if tuple(x_train.columns) != FEATURE_COLUMNS:
            raise ValueError("aggregate feature contract mismatch")
        train_weight = 1.0 / (x_train["rolling_mean_28"].to_numpy() + 1.0)
        train_weight /= train_weight.mean()
        validation_weight = 1.0 / (
            x_validation["rolling_mean_28"].to_numpy() + 1.0
        )
        validation_weight /= validation_weight.mean()
        self.model = LGBMRegressor(**self.params).fit(
            x_train,
            y_train,
            sample_weight=train_weight,
            categorical_feature=list(CATEGORICAL_FEATURES),
            eval_set=[(x_validation, y_validation)],
            eval_sample_weight=[validation_weight],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("fit must be called before predict")
        return np.clip(self.model.predict(features), 0.0, None)
