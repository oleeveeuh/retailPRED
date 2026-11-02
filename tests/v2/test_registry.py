import pytest

from forecasting.registry import MRTS_BY_CODE, MRTS_BY_ID, MRTS_SERIES, get_series


def test_registry_has_one_unique_mapping_for_each_v2_category() -> None:
    assert len(MRTS_SERIES) == 11
    assert len(MRTS_BY_ID) == len(MRTS_SERIES)
    assert len(MRTS_BY_CODE) == len(MRTS_SERIES)
    assert get_series("electronics_and_appliances").source_code == "443"
    assert get_series("building_material_and_garden").source_code == "444"


def test_registry_fails_explicitly_for_unknown_series() -> None:
    with pytest.raises(KeyError, match="unknown MRTS series_id"):
        get_series("made_up_category")
