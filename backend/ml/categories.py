"""
Retail Categories Configuration
Lightweight module with category definitions - no heavy imports
"""

from typing import List, Dict, Any

# Retail categories mapping
RETAIL_CATEGORIES = {
    "total_sales": "Total Retail Sales",
    "building_material_and_garden_equipment": "Building Materials & Garden",
    "automobile_dealers": "Automobile Dealers",
    "gasoline_stations": "Gasoline Stations",
    "food_and_beverage_stores": "Food & Beverage Stores",
    "health_and_personal_care_stores": "Health & Personal Care",
    "general_merchandise_stores": "General Merchandise",
    "furniture_and_home_furnishings_stores": "Furniture & Home Furnishings",
    "clothing_and_clothing_accessories_stores": "Clothing & Accessories",
    "sporting_goods_hobby_and_musical_instrument_stores": "Sporting Goods & Hobby",
    "electronics_and_appliance_stores": "Electronics & Appliances",
}


def get_available_categories() -> List[Dict[str, str]]:
    """
    Get list of available retail categories

    Returns:
        List of dictionaries with category keys and display names
    """
    return [
        {"key": key, "display_name": name}
        for key, name in RETAIL_CATEGORIES.items()
    ]


def get_category_display_name(category_key: str) -> str:
    """
    Get display name for a category key

    Args:
        category_key: Category key (e.g., 'total_sales')

    Returns:
        Display name (e.g., 'Total Retail Sales')
    """
    return RETAIL_CATEGORIES.get(category_key, category_key)
