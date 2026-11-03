"""Declared time-regime labels for retail forecast diagnostics."""

from __future__ import annotations

import pandas as pd


MRTS_REGIME_BOUNDARIES = {
    "pre_covid_end": pd.Timestamp("2020-02-01"),
    "disruption_end": pd.Timestamp("2021-06-01"),
}


def assign_mrts_regime(target_time: object) -> pd.Series:
    """Label MRTS targets as pre-COVID, disruption, or post-disruption."""
    target = pd.to_datetime(pd.Series(target_time), errors="raise")
    if target.isna().any():
        raise ValueError("target times cannot be null")
    labels = pd.Series("post_disruption", index=target.index, dtype="string")
    labels.loc[target <= MRTS_REGIME_BOUNDARIES["pre_covid_end"]] = "pre_covid"
    labels.loc[
        (target > MRTS_REGIME_BOUNDARIES["pre_covid_end"])
        & (target <= MRTS_REGIME_BOUNDARIES["disruption_end"])
    ] = "disruption"
    return labels
