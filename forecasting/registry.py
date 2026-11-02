"""The authoritative v2 MRTS category registry.

Codes are Census MRTS category codes. The v2 scope deliberately retains the
11 series shown in the original project while correcting their mappings.
"""

from __future__ import annotations

from forecasting.contracts import SeriesMetadata


MRTS_SERIES: tuple[SeriesMetadata, ...] = (
    SeriesMetadata(series_id="retail_total", source="census_mrts", source_code="44000", name="Retail sales, total", unit="millions_usd", seasonal_adjustment="seasonally_adjusted"),
    SeriesMetadata(series_id="motor_vehicle_and_parts", source="census_mrts", source_code="441", name="Motor vehicle and parts dealers", unit="millions_usd", seasonal_adjustment="seasonally_adjusted"),
    SeriesMetadata(series_id="furniture_and_home_furnishings", source="census_mrts", source_code="442", name="Furniture and home furnishings stores", unit="millions_usd", seasonal_adjustment="seasonally_adjusted"),
    SeriesMetadata(series_id="electronics_and_appliances", source="census_mrts", source_code="443", name="Electronics and appliance stores", unit="millions_usd", seasonal_adjustment="seasonally_adjusted"),
    SeriesMetadata(series_id="building_material_and_garden", source="census_mrts", source_code="444", name="Building material and garden equipment and supplies dealers", unit="millions_usd", seasonal_adjustment="seasonally_adjusted"),
    SeriesMetadata(series_id="food_and_beverage", source="census_mrts", source_code="445", name="Food and beverage stores", unit="millions_usd", seasonal_adjustment="seasonally_adjusted"),
    SeriesMetadata(series_id="health_and_personal_care", source="census_mrts", source_code="446", name="Health and personal care stores", unit="millions_usd", seasonal_adjustment="seasonally_adjusted"),
    SeriesMetadata(series_id="gasoline_stations", source="census_mrts", source_code="447", name="Gasoline stations", unit="millions_usd", seasonal_adjustment="seasonally_adjusted"),
    SeriesMetadata(series_id="clothing_and_accessories", source="census_mrts", source_code="448", name="Clothing and clothing accessories stores", unit="millions_usd", seasonal_adjustment="seasonally_adjusted"),
    SeriesMetadata(series_id="sporting_goods_and_hobby", source="census_mrts", source_code="451", name="Sporting goods, hobby, musical instrument, and book stores", unit="millions_usd", seasonal_adjustment="seasonally_adjusted"),
    SeriesMetadata(series_id="general_merchandise", source="census_mrts", source_code="452", name="General merchandise stores", unit="millions_usd", seasonal_adjustment="seasonally_adjusted"),
)


def _build_index(field: str) -> dict[str, SeriesMetadata]:
    index = {getattr(item, field): item for item in MRTS_SERIES}
    if len(index) != len(MRTS_SERIES):
        raise RuntimeError(f"duplicate {field} in MRTS registry")
    return index


MRTS_BY_ID = _build_index("series_id")
MRTS_BY_CODE = _build_index("source_code")


def get_series(series_id: str) -> SeriesMetadata:
    try:
        return MRTS_BY_ID[series_id]
    except KeyError as exc:
        raise KeyError(f"unknown MRTS series_id: {series_id}") from exc
