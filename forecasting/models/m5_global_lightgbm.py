"""Direct multi-horizon global LightGBM for M5 item/store demand."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


STATIC_COLUMNS = ("item_id", "dept_id", "cat_id", "store_id", "state_id")
CATEGORICAL_FEATURES = (
    "item_code",
    "dept_code",
    "cat_code",
    "store_code",
    "state_code",
    "event_type_1_code",
    "event_type_2_code",
)
FEATURE_COLUMNS = (
    *CATEGORICAL_FEATURES[:5],
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
    "nonzero_rate_28",
    "nonzero_rate_56",
    "days_since_sale",
    "target_wday",
    "target_month",
    "target_year",
    "target_snap",
    "target_price",
    "price_change",
    "event_1",
    "event_2",
    "event_type_1_code",
    "event_type_2_code",
)


class M5DirectFeatureBuilder:
    """Build origin-safe historical and known-future features in horizon blocks."""

    def __init__(
        self,
        *,
        bottom: pd.DataFrame,
        calendar: pd.DataFrame,
        prices_by_week: dict[int, np.ndarray],
    ) -> None:
        required_bottom = {"series_id", *STATIC_COLUMNS}
        if missing := required_bottom.difference(bottom.columns):
            raise ValueError(f"bottom metadata is missing columns: {sorted(missing)}")
        required_calendar = {
            "wm_yr_wk",
            "wday",
            "month",
            "year",
            "event_type_1",
            "event_type_2",
            "snap_CA",
            "snap_TX",
            "snap_WI",
        }
        if missing := required_calendar.difference(calendar.columns):
            raise ValueError(f"calendar is missing columns: {sorted(missing)}")
        self.bottom = bottom.reset_index(drop=True).copy()
        self.calendar = calendar.reset_index(drop=True).copy()
        self.prices_by_week = prices_by_week
        for week, values in prices_by_week.items():
            if np.asarray(values).shape != (len(bottom),):
                raise ValueError(f"price vector for week {week} has invalid shape")
        self.static_codes = {
            column: pd.Categorical(self.bottom[column]).codes.astype(np.int32)
            for column in STATIC_COLUMNS
        }
        event_values = pd.concat(
            [self.calendar["event_type_1"], self.calendar["event_type_2"]],
            ignore_index=True,
        ).fillna("none")
        event_categories = sorted(event_values.astype(str).unique())
        self.event_codes = {
            column: pd.Categorical(
                self.calendar[column].fillna("none").astype(str),
                categories=event_categories,
            ).codes.astype(np.int32)
            for column in ("event_type_1", "event_type_2")
        }
        state = self.bottom["state_id"].astype(str).to_numpy()
        self.snap_columns = np.array([f"snap_{value}" for value in state])

    def _price(self, day_index: int) -> np.ndarray:
        week = int(self.calendar.iloc[day_index]["wm_yr_wk"])
        values = self.prices_by_week.get(week)
        if values is None:
            raise ValueError(f"missing price vector for calendar week {week}")
        return np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0)

    def build(
        self,
        history: np.ndarray,
        *,
        origin_index: int,
        horizons: tuple[int, ...],
        include_target: bool,
    ) -> tuple[pd.DataFrame, np.ndarray | None]:
        values = np.asarray(history, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] != len(self.bottom):
            raise ValueError("history must have one row per bottom series")
        if origin_index < 363:
            raise ValueError("M5 features require at least 364 observed days")
        if not horizons or min(horizons) < 1:
            raise ValueError("horizons must contain positive integers")
        if origin_index + max(horizons) >= len(self.calendar):
            raise ValueError("calendar does not cover requested horizons")
        if include_target and origin_index + max(horizons) >= values.shape[1]:
            raise ValueError("history does not contain requested training targets")

        lag_values = {
            lag: values[:, origin_index + 1 - lag]
            for lag in (1, 7, 14, 28, 56, 364)
        }
        recent_7 = values[:, origin_index - 6 : origin_index + 1]
        recent_28 = values[:, origin_index - 27 : origin_index + 1]
        recent_56 = values[:, origin_index - 55 : origin_index + 1]
        reversed_sales = values[:, : origin_index + 1][:, ::-1] > 0
        has_sale = reversed_sales.any(axis=1)
        days_since_sale = np.where(has_sale, reversed_sales.argmax(axis=1), origin_index + 1)
        origin_price = self._price(origin_index)
        frames: list[pd.DataFrame] = []
        targets: list[np.ndarray] = []
        for horizon in horizons:
            target_index = origin_index + horizon
            calendar_row = self.calendar.iloc[target_index]
            target_price = self._price(target_index)
            snap_matrix = self.calendar.loc[target_index, self.snap_columns].to_numpy()
            frame = pd.DataFrame(
                {
                    "item_code": self.static_codes["item_id"],
                    "dept_code": self.static_codes["dept_id"],
                    "cat_code": self.static_codes["cat_id"],
                    "store_code": self.static_codes["store_id"],
                    "state_code": self.static_codes["state_id"],
                    "horizon": np.full(len(self.bottom), horizon, dtype=np.int16),
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
                    "nonzero_rate_28": (recent_28 > 0).mean(axis=1),
                    "nonzero_rate_56": (recent_56 > 0).mean(axis=1),
                    "days_since_sale": days_since_sale.astype(np.int16),
                    "target_wday": np.full(
                        len(self.bottom), int(calendar_row["wday"]), dtype=np.int8
                    ),
                    "target_month": np.full(
                        len(self.bottom), int(calendar_row["month"]), dtype=np.int8
                    ),
                    "target_year": np.full(
                        len(self.bottom), int(calendar_row["year"]), dtype=np.int16
                    ),
                    "target_snap": snap_matrix.astype(np.int8),
                    "target_price": target_price,
                    "price_change": np.divide(
                        target_price,
                        origin_price,
                        out=np.ones_like(target_price),
                        where=origin_price > 0,
                    ),
                    "event_1": np.full(
                        len(self.bottom), int(pd.notna(calendar_row["event_type_1"])), dtype=np.int8
                    ),
                    "event_2": np.full(
                        len(self.bottom), int(pd.notna(calendar_row["event_type_2"])), dtype=np.int8
                    ),
                    "event_type_1_code": np.full(
                        len(self.bottom), self.event_codes["event_type_1"][target_index], dtype=np.int8
                    ),
                    "event_type_2_code": np.full(
                        len(self.bottom), self.event_codes["event_type_2"][target_index], dtype=np.int8
                    ),
                },
                columns=FEATURE_COLUMNS,
            )
            frames.append(frame)
            if include_target:
                targets.append(values[:, target_index])
        x = pd.concat(frames, ignore_index=True)
        y = np.concatenate(targets).astype(np.float32) if include_target else None
        return x, y


def load_m5_price_vectors(
    *,
    bottom: pd.DataFrame,
    prices_path: str | Path,
    weeks: set[int],
    chunk_size: int = 500_000,
) -> dict[int, np.ndarray]:
    """Load only required weekly prices and align them to hierarchy bottom rows."""
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        prices_path,
        usecols=["store_id", "item_id", "wm_yr_wk", "sell_price"],
        dtype={"store_id": "string", "item_id": "string", "sell_price": "float32"},
        chunksize=chunk_size,
    ):
        selected = chunk.loc[chunk["wm_yr_wk"].isin(weeks)]
        if not selected.empty:
            parts.append(selected)
    if not parts:
        raise ValueError("no prices found for requested M5 weeks")
    prices = pd.concat(parts, ignore_index=True)
    if prices.duplicated(["store_id", "item_id", "wm_yr_wk"]).any():
        raise ValueError("M5 prices contain duplicate keys")
    indexed = prices.set_index(["store_id", "item_id", "wm_yr_wk"])["sell_price"]
    pair_index = pd.MultiIndex.from_frame(bottom[["store_id", "item_id"]])
    result: dict[int, np.ndarray] = {}
    for week in sorted(weeks):
        lookup = pd.MultiIndex.from_arrays(
            [
                pair_index.get_level_values(0),
                pair_index.get_level_values(1),
                np.full(len(bottom), week),
            ],
            names=["store_id", "item_id", "wm_yr_wk"],
        )
        values = indexed.reindex(lookup).to_numpy(dtype=np.float32)
        if (values[np.isfinite(values)] <= 0).any():
            raise ValueError("M5 prices must be positive")
        result[week] = values
    return result


class M5GlobalLightGBM:
    """Deterministic Tweedie model shared across series and forecast horizons."""

    name = "m5_global_lightgbm"
    version = "1.0.0"

    def __init__(
        self,
        *,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        min_child_samples: int = 500,
        random_state: int = 42,
    ) -> None:
        self.params: dict[str, Any] = {
            "objective": "tweedie",
            "tweedie_variance_power": 1.1,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "max_bin": 63,
            "subsample": 0.8,
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
        x_validation: pd.DataFrame | None = None,
        y_validation: np.ndarray | None = None,
        early_stopping_rounds: int = 50,
    ) -> "M5GlobalLightGBM":
        if tuple(x_train.columns) != FEATURE_COLUMNS:
            raise ValueError("M5 LightGBM feature contract mismatch")
        options: dict[str, Any] = {"categorical_feature": list(CATEGORICAL_FEATURES)}
        if x_validation is not None or y_validation is not None:
            if x_validation is None or y_validation is None:
                raise ValueError("validation features and targets must be supplied together")
            options.update(
                {
                    "eval_set": [(x_validation, y_validation)],
                    "eval_metric": "rmse",
                    "callbacks": [lgb.early_stopping(early_stopping_rounds, verbose=False)],
                }
            )
        self.model = LGBMRegressor(**self.params).fit(x_train, y_train, **options)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("fit must be called before predict")
        if tuple(features.columns) != FEATURE_COLUMNS:
            raise ValueError("M5 LightGBM feature contract mismatch")
        return np.clip(self.model.predict(features), 0.0, None)

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "parameters": dict(self.params),
            "features": list(FEATURE_COLUMNS),
            "categorical_features": list(CATEGORICAL_FEATURES),
            "strategy": "direct pooled horizon",
        }

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("fit must be called before save")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            pickle.dump(self, stream, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "M5GlobalLightGBM":
        with path.open("rb") as stream:
            model = pickle.load(stream)  # noqa: S301 - trusted local artifact only
        if not isinstance(model, cls):
            raise TypeError("artifact is not an M5GlobalLightGBM model")
        return model
