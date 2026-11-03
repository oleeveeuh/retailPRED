"""Sparse 12-level M5 hierarchy and coherent forecast reconciliation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import lsq_linear


@dataclass(frozen=True)
class M5Level:
    level: int
    name: str
    group_columns: tuple[str, ...]
    official_series_count: int


M5_LEVELS = (
    M5Level(1, "total", (), 1),
    M5Level(2, "state", ("state_id",), 3),
    M5Level(3, "store", ("store_id",), 10),
    M5Level(4, "category", ("cat_id",), 3),
    M5Level(5, "department", ("dept_id",), 7),
    M5Level(6, "state_category", ("state_id", "cat_id"), 9),
    M5Level(7, "state_department", ("state_id", "dept_id"), 21),
    M5Level(8, "store_category", ("store_id", "cat_id"), 30),
    M5Level(9, "store_department", ("store_id", "dept_id"), 70),
    M5Level(10, "item", ("item_id",), 3_049),
    M5Level(11, "item_state", ("item_id", "state_id"), 9_147),
    M5Level(12, "item_store", ("series_id",), 30_490),
)

BOTTOM_COLUMNS = (
    "series_id",
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
)


def _node_id(level: M5Level, values: tuple[object, ...]) -> str:
    if not level.group_columns:
        return "L01:total"
    labels = "|".join(
        f"{column}={value}" for column, value in zip(level.group_columns, values, strict=True)
    )
    return f"L{level.level:02d}:{labels}"


@dataclass(frozen=True)
class M5Hierarchy:
    """Metadata and sparse summing matrix from product/store bottom series."""

    bottom: pd.DataFrame
    nodes: pd.DataFrame
    summing_matrix: sparse.csr_matrix

    @classmethod
    def from_bottom(cls, bottom: pd.DataFrame) -> "M5Hierarchy":
        if missing := set(BOTTOM_COLUMNS).difference(bottom.columns):
            raise ValueError(f"bottom metadata is missing columns: {sorted(missing)}")
        clean = bottom.loc[:, BOTTOM_COLUMNS].copy()
        if clean.isna().any().any():
            raise ValueError("bottom metadata cannot contain nulls")
        if clean["series_id"].duplicated().any():
            raise ValueError("series_id must be unique at the bottom level")
        clean = clean.astype(str).sort_values("series_id", ignore_index=True)
        row_indices: list[int] = []
        column_indices: list[int] = []
        node_records: list[dict[str, object]] = []
        row_index = 0
        for level in M5_LEVELS:
            if not level.group_columns:
                groups = [((), clean.index.to_numpy())]
            else:
                grouper: str | list[str]
                grouper = (
                    level.group_columns[0]
                    if len(level.group_columns) == 1
                    else list(level.group_columns)
                )
                groups = [
                    (
                        key if isinstance(key, tuple) else (key,),
                        group.index.to_numpy(),
                    )
                    for key, group in clean.groupby(grouper, sort=True, observed=True)
                ]
            for values, members in groups:
                node_records.append(
                    {
                        "node_id": _node_id(level, values),
                        "level": level.level,
                        "level_name": level.name,
                        "member_count": len(members),
                    }
                )
                row_indices.extend([row_index] * len(members))
                column_indices.extend(members.tolist())
                row_index += 1
        matrix = sparse.coo_matrix(
            (
                np.ones(len(row_indices), dtype=np.float64),
                (row_indices, column_indices),
            ),
            shape=(len(node_records), len(clean)),
        ).tocsr()
        nodes = pd.DataFrame.from_records(node_records)
        if nodes["node_id"].duplicated().any():
            raise RuntimeError("hierarchy construction produced duplicate node IDs")
        return cls(bottom=clean, nodes=nodes, summing_matrix=matrix)

    def validate_official_shape(self) -> None:
        if len(self.bottom) != M5_LEVELS[-1].official_series_count:
            raise ValueError(
                f"official M5 bottom level requires {M5_LEVELS[-1].official_series_count:,} series"
            )
        actual = self.nodes.groupby("level").size().to_dict()
        expected = {level.level: level.official_series_count for level in M5_LEVELS}
        if actual != expected:
            raise ValueError(f"M5 hierarchy counts differ; expected={expected}, actual={actual}")

    def aggregate_bottom(self, forecasts: pd.DataFrame) -> pd.DataFrame:
        required = {"series_id", "target_time", "prediction"}
        if missing := required.difference(forecasts.columns):
            raise ValueError(f"bottom forecasts are missing columns: {sorted(missing)}")
        if forecasts[["series_id", "target_time"]].duplicated().any():
            raise ValueError("bottom forecasts contain duplicate series/time rows")
        expected = set(self.bottom["series_id"])
        records: list[pd.DataFrame] = []
        for target_time, group in forecasts.groupby("target_time", sort=True):
            indexed = group.set_index("series_id")["prediction"]
            if set(indexed.index) != expected:
                raise ValueError("every target time must contain the complete bottom level")
            bottom_values = indexed.reindex(self.bottom["series_id"]).to_numpy(dtype=float)
            if not np.isfinite(bottom_values).all():
                raise ValueError("forecast values must be finite")
            aggregated = self.summing_matrix @ bottom_values
            result = self.nodes[["node_id", "level", "level_name"]].copy()
            result["target_time"] = pd.Timestamp(target_time)
            result["prediction"] = aggregated
            records.append(result)
        if not records:
            raise ValueError("bottom forecasts cannot be empty")
        return pd.concat(records, ignore_index=True)

    def coherence_error(self, forecasts: pd.DataFrame) -> float:
        required = {"node_id", "target_time", "prediction"}
        if missing := required.difference(forecasts.columns):
            raise ValueError(f"hierarchical forecasts are missing columns: {sorted(missing)}")
        bottom_ids = set(
            self.nodes.loc[self.nodes["level"] == 12, "node_id"]
        )
        bottom_node_to_series = dict(
            zip(
                self.nodes.loc[self.nodes["level"] == 12, "node_id"],
                self.bottom["series_id"],
                strict=True,
            )
        )
        errors: list[float] = []
        for target_time, group in forecasts.groupby("target_time", sort=False):
            indexed = group.set_index("node_id")["prediction"]
            if set(indexed.index) != set(self.nodes["node_id"]):
                raise ValueError("hierarchical forecast must contain every node")
            bottom = pd.DataFrame(
                {
                    "series_id": [bottom_node_to_series[node] for node in bottom_ids],
                    "target_time": target_time,
                    "prediction": [indexed.loc[node] for node in bottom_ids],
                }
            )
            coherent = self.aggregate_bottom(bottom).set_index("node_id")["prediction"]
            errors.append(float(np.max(np.abs(indexed - coherent.reindex(indexed.index)))))
        return max(errors, default=0.0)


class WeightedLeastSquaresReconciler:
    """Diagonal-variance MinT-style projection solved in sparse form."""

    def __init__(self, hierarchy: M5Hierarchy, *, nonnegative: bool = True) -> None:
        self.hierarchy = hierarchy
        self.nonnegative = nonnegative

    def reconcile(
        self, base_forecasts: pd.DataFrame, variances: pd.Series
    ) -> pd.DataFrame:
        required = {"node_id", "target_time", "prediction"}
        if missing := required.difference(base_forecasts.columns):
            raise ValueError(f"base forecasts are missing columns: {sorted(missing)}")
        variance = variances.reindex(self.hierarchy.nodes["node_id"])
        if variance.isna().any() or (variance <= 0).any():
            raise ValueError("every hierarchy node requires a positive variance")
        inverse_scale = 1.0 / np.sqrt(variance.to_numpy(dtype=float))
        weighted_s = sparse.diags(inverse_scale) @ self.hierarchy.summing_matrix
        records: list[pd.DataFrame] = []
        expected_nodes = set(self.hierarchy.nodes["node_id"])
        for target_time, group in base_forecasts.groupby("target_time", sort=True):
            indexed = group.set_index("node_id")["prediction"]
            if set(indexed.index) != expected_nodes:
                raise ValueError("every target time must contain every hierarchy node")
            y_hat = indexed.reindex(self.hierarchy.nodes["node_id"]).to_numpy(dtype=float)
            weighted_y = inverse_scale * y_hat
            bounds = (0.0, np.inf) if self.nonnegative else (-np.inf, np.inf)
            solution = lsq_linear(weighted_s, weighted_y, bounds=bounds, lsmr_tol="auto")
            if not solution.success:
                raise RuntimeError(f"hierarchy reconciliation failed: {solution.message}")
            coherent = self.hierarchy.summing_matrix @ solution.x
            result = self.hierarchy.nodes[["node_id", "level", "level_name"]].copy()
            result["target_time"] = pd.Timestamp(target_time)
            result["prediction"] = coherent
            records.append(result)
        if not records:
            raise ValueError("base forecasts cannot be empty")
        return pd.concat(records, ignore_index=True)


def m5_revenue_weights(
    hierarchy: M5Hierarchy, recent_bottom_revenue: pd.Series
) -> pd.Series:
    """Give each hierarchy level equal weight, then allocate by dollar sales."""
    revenue = recent_bottom_revenue.reindex(hierarchy.bottom["series_id"])
    if revenue.isna().any() or (revenue < 0).any():
        raise ValueError("every bottom series requires non-negative recent revenue")
    node_revenue = hierarchy.summing_matrix @ revenue.to_numpy(dtype=float)
    weights = np.zeros(len(hierarchy.nodes), dtype=float)
    for level in M5_LEVELS:
        mask = hierarchy.nodes["level"].to_numpy() == level.level
        total = float(node_revenue[mask].sum())
        if total <= 0:
            raise ValueError(f"level {level.level} has zero total revenue")
        weights[mask] = node_revenue[mask] / total / len(M5_LEVELS)
    return pd.Series(weights, index=hierarchy.nodes["node_id"], name="weight")


def weighted_rmsse(
    evaluation: pd.DataFrame,
    *,
    scales: pd.Series,
    weights: pd.Series,
) -> float:
    """Compute WRMSSE from predeclared training scales and revenue weights."""
    required = {"node_id", "target_time", "actual", "prediction"}
    if missing := required.difference(evaluation.columns):
        raise ValueError(f"evaluation is missing columns: {sorted(missing)}")
    if evaluation[["node_id", "target_time"]].duplicated().any():
        raise ValueError("evaluation contains duplicate node/time rows")
    squared_error = evaluation.assign(
        squared_error=(evaluation["actual"] - evaluation["prediction"]) ** 2
    ).groupby("node_id", sort=False)["squared_error"].mean()
    aligned_scales = scales.reindex(squared_error.index)
    aligned_weights = weights.reindex(squared_error.index)
    if aligned_scales.isna().any() or (aligned_scales <= 0).any():
        raise ValueError("every evaluated node requires a positive training scale")
    if aligned_weights.isna().any() or (aligned_weights < 0).any():
        raise ValueError("every evaluated node requires a non-negative weight")
    if not np.isclose(aligned_weights.sum(), 1.0):
        raise ValueError("weights for evaluated nodes must sum to one")
    rmsse = np.sqrt(squared_error / aligned_scales)
    return float(np.sum(aligned_weights * rmsse))
