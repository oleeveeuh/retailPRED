import pandas as pd

from forecasting.evaluation.regimes import assign_mrts_regime


def test_mrts_regimes_use_declared_target_month_boundaries() -> None:
    result = assign_mrts_regime(
        ["2020-02-01", "2020-03-01", "2021-06-01", "2021-07-01"]
    )
    assert result.tolist() == [
        "pre_covid",
        "disruption",
        "disruption",
        "post_disruption",
    ]
    assert str(result.dtype) == "string"


def test_mrts_regimes_preserve_input_index() -> None:
    target = pd.Series(["2019-01-01", "2022-01-01"], index=[5, 9])
    assert assign_mrts_regime(target).index.tolist() == [5, 9]
