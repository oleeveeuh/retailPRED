"""Global LightGBM forecaster trained across a balanced monthly panel."""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from forecasting.features.monthly import build_causal_features, validate_monthly_panel


FEATURE_COLUMNS = (
    "series_code",
    "lag_1_ratio",
    "lag_2_ratio",
    "lag_3_ratio",
    "lag_6_ratio",
    "rolling_mean_3_ratio",
    "rolling_std_3_ratio",
    "rolling_mean_6_ratio",
    "rolling_std_6_ratio",
    "rolling_mean_12_ratio",
    "rolling_std_12_ratio",
    "diff_1_ratio",
    "pct_change_1",
    "month_sin",
    "month_cos",
)


class GlobalLightGBM:
    """Recursive global model with deterministic training and causal features."""

    name = "global_lightgbm"
    version = "2.0.0"

    def __init__(
        self,
        *,
        n_estimators: int = 250,
        learning_rate: float = 0.03,
        num_leaves: int = 15,
        min_child_samples: int = 10,
        random_state: int = 42,
        quantiles: tuple[float, ...] = (),
    ) -> None:
        if n_estimators < 1 or learning_rate <= 0 or num_leaves < 2:
            raise ValueError("invalid LightGBM training parameters")
        if min_child_samples < 1:
            raise ValueError("min_child_samples must be positive")
        if any(not 0 < quantile < 1 for quantile in quantiles):
            raise ValueError("quantiles must be strictly between zero and one")
        if len(set(quantiles)) != len(quantiles):
            raise ValueError("quantiles must be unique")
        self.params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "random_state": random_state,
            "n_jobs": 1,
            "verbosity": -1,
            "deterministic": True,
            "force_col_wise": True,
        }
        self.quantiles = tuple(sorted(quantiles))
        self._point_model: LGBMRegressor | None = None
        self._quantile_models: dict[float, LGBMRegressor] = {}
        self._history: pd.DataFrame | None = None
        self._series_codes: dict[str, int] = {}

    def _new_model(self, *, quantile: float | None = None) -> LGBMRegressor:
        objective: dict[str, Any] = {"objective": "regression_l1"}
        if quantile is not None:
            objective = {"objective": "quantile", "alpha": quantile}
        return LGBMRegressor(**self.params, **objective)

    def _features(self, panel: pd.DataFrame) -> pd.DataFrame:
        featured = build_causal_features(panel)
        unknown = set(featured["series_id"].astype(str)) - set(self._series_codes)
        if unknown:
            raise ValueError(f"panel contains unknown series: {sorted(unknown)}")
        featured["series_code"] = (
            featured["series_id"].astype(str).map(self._series_codes).astype("int32")
        )
        radians = 2 * math.pi * featured["month"] / 12
        featured["month_sin"] = np.sin(radians)
        featured["month_cos"] = np.cos(radians)
        scale = featured["lag_12"]
        for column in ("lag_1", "lag_2", "lag_3", "lag_6"):
            featured[f"{column}_ratio"] = featured[column] / scale
        for column in (
            "rolling_mean_3",
            "rolling_std_3",
            "rolling_mean_6",
            "rolling_std_6",
            "rolling_mean_12",
            "rolling_std_12",
            "diff_1",
        ):
            featured[f"{column}_ratio"] = featured[column] / scale
        return featured

    @staticmethod
    def _validate_balanced(panel: pd.DataFrame) -> None:
        calendars = {
            series_id: tuple(group["observation_time"])
            for series_id, group in panel.groupby("series_id", sort=False)
        }
        if not calendars:
            raise ValueError("history cannot be empty")
        reference = next(iter(calendars.values()))
        if any(calendar != reference for calendar in calendars.values()):
            raise ValueError("global LightGBM requires a balanced monthly panel")

    def fit(self, history: pd.DataFrame) -> "GlobalLightGBM":
        panel = validate_monthly_panel(history)
        self._validate_balanced(panel)
        numeric_values = pd.to_numeric(panel["value"], errors="coerce")
        if numeric_values.isna().any() or not np.isfinite(numeric_values).all():
            raise ValueError("history values must be finite numbers")
        if (numeric_values <= 0).any():
            raise ValueError("seasonal log-ratio training requires positive history values")
        panel["value"] = numeric_values.astype(float)
        self._series_codes = {
            series_id: index
            for index, series_id in enumerate(sorted(panel["series_id"].astype(str).unique()))
        }
        featured = self._features(panel).dropna(subset=[*FEATURE_COLUMNS, "value"])
        featured = featured[np.isfinite(featured.loc[:, FEATURE_COLUMNS]).all(axis=1)]
        if featured.empty:
            raise ValueError("at least 13 complete months are required to train LightGBM")
        x_train = featured.loc[:, FEATURE_COLUMNS]
        y_train = np.log(featured["value"].astype(float) / featured["lag_12"])
        fit_options = {"categorical_feature": ["series_code"]}
        self._point_model = self._new_model().fit(x_train, y_train, **fit_options)
        self._quantile_models = {
            quantile: self._new_model(quantile=quantile).fit(
                x_train, y_train, **fit_options
            )
            for quantile in self.quantiles
        }
        self._history = panel[["series_id", "observation_time", "value"]].copy()
        return self

    def _require_fit(self) -> tuple[pd.DataFrame, LGBMRegressor]:
        if self._history is None or self._point_model is None:
            raise RuntimeError("fit must be called before predict")
        return self._history.copy(), self._point_model

    def _recursive_predictions(
        self, horizon: int, models: dict[str, LGBMRegressor]
    ) -> pd.DataFrame:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        history, _ = self._require_fit()
        records: list[dict[str, object]] = []
        for step in range(1, horizon + 1):
            next_rows = pd.DataFrame(
                {
                    "series_id": sorted(self._series_codes),
                    "observation_time": [
                        history.loc[history["series_id"] == series_id, "observation_time"].max()
                        + pd.offsets.MonthBegin(1)
                        for series_id in sorted(self._series_codes)
                    ],
                    "value": np.nan,
                }
            )
            extended = pd.concat([history, next_rows], ignore_index=True)
            design = self._features(extended)
            current = design[design["value"].isna()].copy()
            if len(current) != len(self._series_codes):
                raise RuntimeError("recursive feature construction produced an invalid panel")
            current = current.sort_values("series_id")
            log_ratios = models["point"].predict(current.loc[:, FEATURE_COLUMNS])
            point_values = current["lag_12"].to_numpy(dtype=float) * np.exp(log_ratios)
            for row_index, (_, row) in enumerate(current.iterrows()):
                series_id = str(row["series_id"])
                target_time = pd.Timestamp(row["observation_time"])
                prediction = float(point_values[row_index])
                records.append(
                    {
                        "series_id": series_id,
                        "target_time": target_time,
                        "horizon": step,
                        "prediction": prediction,
                    }
                )
                history.loc[len(history)] = [series_id, target_time, prediction]
        return pd.DataFrame.from_records(records)

    def predict(self, horizon: int) -> pd.DataFrame:
        _, point_model = self._require_fit()
        return self._recursive_predictions(horizon, {"point": point_model})

    def predict_quantiles(
        self, horizon: int, quantiles: tuple[float, ...]
    ) -> pd.DataFrame:
        if not quantiles:
            raise ValueError("at least one quantile is required")
        missing = set(quantiles) - set(self._quantile_models)
        if missing:
            raise ValueError(f"quantile models were not fitted for: {sorted(missing)}")
        frames: list[pd.DataFrame] = []
        for quantile in quantiles:
            predicted = self._recursive_predictions(
                horizon, {"point": self._quantile_models[quantile]}
            )
            predicted["quantile"] = quantile
            frames.append(predicted)
        return pd.concat(frames, ignore_index=True)

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "parameters": dict(self.params),
            "features": list(FEATURE_COLUMNS),
            "target_transformation": "log(value / lag_12)",
            "quantiles": list(self.quantiles),
            "series_ids": sorted(self._series_codes),
        }

    def save(self, path: Path) -> None:
        self._require_fit()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            pickle.dump(self, stream, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "GlobalLightGBM":
        with path.open("rb") as stream:
            model = pickle.load(stream)  # noqa: S301 - only load trusted local artifacts
        if not isinstance(model, cls):
            raise TypeError("artifact is not a GlobalLightGBM model")
        return model
