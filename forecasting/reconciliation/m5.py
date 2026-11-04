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


@dataclass(frozen=True)
class M5UpperHierarchy:
    """Levels 1–9 reconciled over the 70 store/department groups."""

    full_hierarchy: M5Hierarchy
    groups: pd.DataFrame
    nodes: pd.DataFrame
    summing_matrix: sparse.csr_matrix
    bottom_group_index: np.ndarray

    @classmethod
    def from_hierarchy(cls, hierarchy: M5Hierarchy) -> "M5UpperHierarchy":
        metadata = hierarchy.bottom.copy()
        consistency = metadata.groupby(["store_id", "dept_id"], observed=True).agg(
            state_count=("state_id", "nunique"),
            category_count=("cat_id", "nunique"),
        )
        if (consistency != 1).any().any():
            raise ValueError("store/department groups must map to one state and category")
        groups = (
            metadata.groupby(["store_id", "dept_id"], as_index=False, observed=True)
            .agg(state_id=("state_id", "first"), cat_id=("cat_id", "first"))
            .sort_values(["store_id", "dept_id"], ignore_index=True)
        )
        level9 = M5_LEVELS[8]
        groups["group_id"] = [
            _node_id(level9, (row.store_id, row.dept_id))
            for row in groups.itertuples(index=False)
        ]
        group_lookup = {
            (row.store_id, row.dept_id): index
            for index, row in enumerate(groups.itertuples(index=False))
        }
        bottom_group_index = np.array(
            [
                group_lookup[(row.store_id, row.dept_id)]
                for row in metadata.itertuples(index=False)
            ],
            dtype=np.int32,
        )

        row_indices: list[int] = []
        column_indices: list[int] = []
        records: list[dict[str, object]] = []
        row_index = 0
        for level in M5_LEVELS[:9]:
            if not level.group_columns:
                level_groups = [((), groups.index.to_numpy())]
            else:
                grouper: str | list[str] = (
                    level.group_columns[0]
                    if len(level.group_columns) == 1
                    else list(level.group_columns)
                )
                level_groups = [
                    (
                        key if isinstance(key, tuple) else (key,),
                        group.index.to_numpy(),
                    )
                    for key, group in groups.groupby(grouper, sort=True, observed=True)
                ]
            for values, members in level_groups:
                records.append(
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
            shape=(len(records), len(groups)),
        ).tocsr()
        nodes = pd.DataFrame.from_records(records)
        expected_nodes = hierarchy.nodes.loc[
            hierarchy.nodes["level"] <= 9, "node_id"
        ].tolist()
        if nodes["node_id"].tolist() != expected_nodes:
            raise ValueError("upper hierarchy does not align with the full M5 hierarchy")
        return cls(
            full_hierarchy=hierarchy,
            groups=groups,
            nodes=nodes,
            summing_matrix=matrix,
            bottom_group_index=bottom_group_index,
        )

    def reconcile(
        self, base_forecasts: np.ndarray, variances: pd.Series
    ) -> np.ndarray:
        """Project incoherent level 1–9 forecasts onto the 70-group hierarchy."""
        base = np.asarray(base_forecasts, dtype=np.float64)
        if base.ndim != 2 or base.shape[0] != len(self.nodes):
            raise ValueError("upper base forecasts must have node-by-horizon shape")
        variance = variances.reindex(self.nodes["node_id"])
        if variance.isna().any() or (variance <= 0).any():
            raise ValueError("every upper node requires a positive variance")
        inverse_scale = 1.0 / np.sqrt(variance.to_numpy(dtype=float))
        weighted_s = sparse.diags(inverse_scale) @ self.summing_matrix
        reconciled = np.empty_like(base)
        for horizon in range(base.shape[1]):
            solution = lsq_linear(
                weighted_s,
                inverse_scale * base[:, horizon],
                bounds=(0.0, np.inf),
                lsmr_tol="auto",
            )
            if not solution.success:
                raise RuntimeError(f"upper reconciliation failed: {solution.message}")
            reconciled[:, horizon] = self.summing_matrix @ solution.x
        return reconciled

    def disaggregate_to_bottom(
        self,
        reconciled_upper: np.ndarray,
        bottom_forecast: np.ndarray,
        *,
        fallback_shares: np.ndarray | None = None,
    ) -> np.ndarray:
        """Scale item/store forecasts within level-9 groups, then aggregate fully."""
        upper = np.asarray(reconciled_upper, dtype=np.float64)
        bottom = np.asarray(bottom_forecast, dtype=np.float64)
        if upper.ndim != 2 or upper.shape[0] != len(self.nodes):
            raise ValueError("reconciled upper forecast has invalid shape")
        if bottom.shape != (len(self.full_hierarchy.bottom), upper.shape[1]):
            raise ValueError("bottom forecast has invalid shape")
        if (bottom < 0).any() or not np.isfinite(bottom).all():
            raise ValueError("bottom forecasts must be finite and non-negative")
        level9_targets = upper[self.nodes["level"].to_numpy() == 9]
        adjusted = np.zeros_like(bottom)
        fallback = (
            np.ones(len(bottom), dtype=np.float64)
            if fallback_shares is None
            else np.asarray(fallback_shares, dtype=np.float64)
        )
        if fallback.shape != (len(bottom),) or (fallback < 0).any():
            raise ValueError("fallback shares must be a non-negative bottom vector")
        for horizon in range(bottom.shape[1]):
            group_totals = np.bincount(
                self.bottom_group_index,
                weights=bottom[:, horizon],
                minlength=len(self.groups),
            )
            positive = group_totals > 0
            factors = np.zeros(len(self.groups), dtype=np.float64)
            factors[positive] = level9_targets[positive, horizon] / group_totals[positive]
            adjusted[:, horizon] = bottom[:, horizon] * factors[self.bottom_group_index]
            for group_index in np.flatnonzero(~positive & (level9_targets[:, horizon] > 0)):
                members = self.bottom_group_index == group_index
                shares = fallback[members]
                if shares.sum() <= 0:
                    shares = np.ones(members.sum(), dtype=np.float64)
                adjusted[members, horizon] = (
                    level9_targets[group_index, horizon] * shares / shares.sum()
                )
        return self.full_hierarchy.summing_matrix @ adjusted


@dataclass(frozen=True)
class M5ReconciliationResult:
    bottom_up: np.ndarray
    incoherent_base: np.ndarray
    reconciled: np.ndarray
    aggregate_blend_weights: pd.Series
    residual_variances: pd.Series
    base_coherence_max_abs: float
    reconciled_coherence_max_abs: float


def reconcile_m5_upper_forecasts(
    upper: M5UpperHierarchy,
    *,
    bottom_validation_forecast: np.ndarray,
    bottom_evaluation_forecast: np.ndarray,
    aggregate_validation_forecast: np.ndarray,
    aggregate_evaluation_forecast: np.ndarray,
    upper_validation_actual: np.ndarray,
    fallback_shares: np.ndarray,
) -> M5ReconciliationResult:
    """Fit level blends and WLS variances on validation, then reconcile evaluation."""
    hierarchy = upper.full_hierarchy
    bottom_validation = np.asarray(bottom_validation_forecast, dtype=np.float64)
    bottom_evaluation = np.asarray(bottom_evaluation_forecast, dtype=np.float64)
    aggregate_validation = np.asarray(aggregate_validation_forecast, dtype=np.float64)
    aggregate_evaluation = np.asarray(aggregate_evaluation_forecast, dtype=np.float64)
    validation_actual = np.asarray(upper_validation_actual, dtype=np.float64)
    horizon = bottom_evaluation.shape[1]
    if bottom_validation.shape != (len(hierarchy.bottom), horizon):
        raise ValueError("bottom validation forecast has invalid shape")
    if bottom_evaluation.shape != (len(hierarchy.bottom), horizon):
        raise ValueError("bottom evaluation forecast has invalid shape")
    upper_shape = (len(upper.nodes), horizon)
    if any(
        values.shape != upper_shape
        for values in (aggregate_validation, aggregate_evaluation, validation_actual)
    ):
        raise ValueError("aggregate forecasts and validation actuals must align")

    bottom_up_validation = hierarchy.summing_matrix @ bottom_validation
    bottom_up_evaluation = hierarchy.summing_matrix @ bottom_evaluation
    upper_count = len(upper.nodes)
    upper_bottom_up_validation = bottom_up_validation[:upper_count]
    upper_bottom_up_evaluation = bottom_up_evaluation[:upper_count]
    level_values = upper.nodes["level"].to_numpy()
    blend_weights = np.empty(upper_count, dtype=np.float64)
    for level in range(1, 10):
        mask = level_values == level
        aggregate_mse = np.mean(
            (aggregate_validation[mask] - validation_actual[mask]) ** 2
        )
        bottom_up_mse = np.mean(
            (upper_bottom_up_validation[mask] - validation_actual[mask]) ** 2
        )
        denominator = aggregate_mse + bottom_up_mse
        blend_weights[mask] = bottom_up_mse / denominator if denominator > 0 else 0.5
    validation_base = (
        blend_weights[:, None] * aggregate_validation
        + (1.0 - blend_weights[:, None]) * upper_bottom_up_validation
    )
    evaluation_base = (
        blend_weights[:, None] * aggregate_evaluation
        + (1.0 - blend_weights[:, None]) * upper_bottom_up_evaluation
    )
    residual_variance = np.mean((validation_base - validation_actual) ** 2, axis=1)
    for level in range(1, 10):
        mask = level_values == level
        level_median = float(np.median(residual_variance[mask]))
        residual_variance[mask] = (
            0.5 * residual_variance[mask] + 0.5 * level_median
        )
    floor = max(float(np.median(residual_variance)) * 1e-6, 1e-6)
    residual_variance = np.maximum(residual_variance, floor)
    variances = pd.Series(residual_variance, index=upper.nodes["node_id"])
    reconciled_upper = upper.reconcile(evaluation_base, variances)
    reconciled_full = upper.disaggregate_to_bottom(
        reconciled_upper,
        bottom_evaluation,
        fallback_shares=fallback_shares,
    )
    incoherent_full = bottom_up_evaluation.copy()
    incoherent_full[:upper_count] = evaluation_base
    bottom_mask = hierarchy.nodes["level"].to_numpy() == 12
    coherent_check = hierarchy.summing_matrix @ reconciled_full[bottom_mask]
    return M5ReconciliationResult(
        bottom_up=bottom_up_evaluation,
        incoherent_base=incoherent_full,
        reconciled=reconciled_full,
        aggregate_blend_weights=pd.Series(
            blend_weights, index=upper.nodes["node_id"], name="aggregate_blend_weight"
        ),
        residual_variances=variances,
        base_coherence_max_abs=float(
            np.max(np.abs(incoherent_full - bottom_up_evaluation))
        ),
        reconciled_coherence_max_abs=float(
            np.max(np.abs(reconciled_full - coherent_check))
        ),
    )


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
