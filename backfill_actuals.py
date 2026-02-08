#!/usr/bin/env python3
"""
Backfill Actual Values from MRTS API

This script fetches actual monthly retail sales data from the Census Bureau
MRTS API and updates all matching predictions in the database.

The Census Bureau typically releases monthly data with a 6-7 week lag.
As of February 2025, data is available through November 2025.

Usage:
    python backfill_actuals.py [--dry-run]

Options:
    --dry-run: Show what would be updated without making changes
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple
import sqlite3
import requests

# Import from config
try:
    from config import (
        DATABASE_PATH,
        MRTS_BASE_URL,
        MRTS_API_KEY,
        RETAIL_CATEGORIES,
        CENSUS_SCALING_FACTORS
    )
except ImportError:
    print("ERROR: Could not import config. Please run from repository root.")
    sys.exit(2)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# MRTS category codes mapping
MRTS_CATEGORY_CODES = {
    'total_retail_sales': '4400A',
    'automobile_dealers': '441',
    'building_materials_garden': '443',
    'clothing_accessories': '452',
    'electronics_appliances': '44X72',
    'food_beverage_stores': '445',
    'furniture_home_furnishings': '442',
    'gasoline_stations': '448',
    'general_merchandise': '454',
    'health_personal_care': '447',
    'sporting_goods_hobby': '453',
    'nonstore_retailers': '444',
}


def fetch_monthly_actuals(year: int, month: int) -> Dict[str, float]:
    """
    Fetch actual retail sales for a specific month from MRTS API.

    Args:
        year: Year (e.g., 2025)
        month: Month (1-12)

    Returns:
        Dictionary mapping category_key to actual sales value (millions)
    """
    month_str = f"{year}-{month:02d}"
    logger.info(f"Fetching MRTS data for {month_str}")

    params = {
        "get": "data_type_code,time_slot_id,seasonally_adj,category_code,cell_value",
        "time": month_str,
        "key": MRTS_API_KEY
    }

    try:
        response = requests.get(MRTS_BASE_URL, params=params, timeout=60)

        if response.status_code != 200:
            logger.error(f"API error: HTTP {response.status_code}")
            return {}

        data = response.json()
        if len(data) <= 1:
            logger.warning(f"No data available for {month_str}")
            return {}

        headers = data[0]
        rows = data[1:]

        # Find column indices
        try:
            category_idx = headers.index('category_code')
            value_idx = headers.index('cell_value')
            data_type_idx = headers.index('data_type_code')
            seasonally_adj_idx = headers.index('seasonally_adj')
        except ValueError as e:
            logger.error(f"Unexpected API response format: {e}")
            return {}

        actuals = {}
        for category_key, category_code in MRTS_CATEGORY_CODES.items():
            for row in rows:
                if len(row) <= max(category_idx, value_idx, data_type_idx, seasonally_adj_idx):
                    continue

                if (row[category_idx] == category_code and
                    row[data_type_idx] == 'SM' and
                    row[seasonally_adj_idx] == 'yes'):

                    try:
                        value_str = row[value_idx]
                        if value_str and value_str not in ['M', '0', '']:
                            value = float(value_str)
                            if value > 0:
                                # Apply category-specific scaling factor
                                scaling_factor = CENSUS_SCALING_FACTORS.get(category_key, 70.0)
                                scaled_value = value / scaling_factor
                                actuals[category_key] = scaled_value
                                break
                    except (ValueError, TypeError):
                        continue

        logger.info(f"  Fetched {len(actuals)}/{len(MRTS_CATEGORY_CODES)} categories")
        return actuals

    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return {}


def get_months_to_backfill() -> List[Tuple[int, int]]:
    """
    Determine which months need actual values backfilled.

    Checks the database for predictions without actuals and returns
    the unique year-month combinations that are likely to have Census data.

    The Census Bureau releases data with a lag, so we only check months
    that are at least 2 months in the past.

    Returns:
        List of (year, month) tuples
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    # Get unique year-month from predictions without actuals
    cursor.execute("""
        SELECT DISTINCT
            CAST(strftime('%Y', prediction_date) AS INTEGER) as year,
            CAST(strftime('%m', prediction_date) AS INTEGER) as month
        FROM prediction_log
        WHERE actual_value IS NULL
        AND (model_name LIKE '%randomforest%' OR model_name LIKE '%lgbm%')
        ORDER BY year, month
    """)

    all_months = [(row[0], row[1]) for row in cursor.fetchall()]
    conn.close()

    # Filter to only include months that likely have Census data
    # Census data is typically available 2 months after month end
    from datetime import datetime
    current_date = datetime.now()
    current_year = current_date.year
    current_month = current_date.month

    # Only process months that are at least 2 months old
    available_months = []
    for year, month in all_months:
        if year < current_year or (year == current_year and month <= current_month - 2):
            available_months.append((year, month))

    return available_months


def update_predictions_for_month(year: int, month: int, actuals: Dict[str, float],
                                 dry_run: bool = False) -> int:
    """
    Update all predictions for a specific month with actual values.

    Args:
        year: Year
        month: Month
        actuals: Dictionary of category_key -> actual_value
        dry_run: If True, don't make actual changes

    Returns:
        Number of predictions updated
    """
    if not actuals:
        return 0

    month_str = f"{year}-{month:02d}"
    logger.info(f"Updating predictions for {month_str}")

    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    updated_count = 0

    # Mapping from config category_key to model name patterns
    # Model names use "and" instead of underscore for some categories
    CATEGORY_KEY_TO_MODEL_PATTERN = {
        'electronics_appliances': 'electronics_and_appliances',
        'health_personal_care': 'health_and_personal_care',
        'food_beverage_stores': 'food_and_beverage_stores',
        'building_materials_garden': 'building_materials_and_garden',
        'furniture_home_furnishings': 'furniture_and_home_furnishings',
        'sporting_goods_hobby': 'sporting_goods_and_hobby',
        'clothing_accessories': 'clothing_and_accessories',
    }

    for category_key, actual_value in actuals.items():
        try:
            # Get the pattern to match in model names
            model_pattern = CATEGORY_KEY_TO_MODEL_PATTERN.get(category_key, category_key)

            # Find predictions for this category and month
            cursor.execute("""
                UPDATE prediction_log
                SET actual_value = ?,
                    error_absolute = ABS(predicted_value - ?),
                    error_percentage = ABS(predicted_value - ?) / ? * 100,
                    is_validated = 1
                WHERE strftime('%Y-%m', prediction_date) = ?
                AND model_name LIKE ?
                AND actual_value IS NULL
            """, (actual_value, actual_value, actual_value, actual_value,
                  month_str, f'%{model_pattern}%'))

            rows_updated = cursor.rowcount
            updated_count += rows_updated

            if rows_updated > 0:
                logger.info(f"  {category_key}: Updated {rows_updated} predictions")

        except Exception as e:
            logger.error(f"  Error updating {category_key}: {e}")

    if not dry_run:
        conn.commit()

    conn.close()
    return updated_count


def main():
    parser = argparse.ArgumentParser(
        description="Backfill actual values from MRTS API"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be updated without making changes'
    )
    parser.add_argument(
        '--year',
        type=int,
        help='Only process specific year'
    )
    parser.add_argument(
        '--month',
        type=int,
        help='Only process specific month (1-12)'
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("BACKFILL ACTUALS FROM MRTS API")
    logger.info("=" * 80)

    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be made")

    # Determine months to process
    if args.year and args.month:
        months_to_process = [(args.year, args.month)]
    else:
        months_to_process = get_months_to_backfill()

    if not months_to_process:
        logger.info("No months need backfilling")
        return

    logger.info(f"Months to process: {len(months_to_process)}")
    for year, month in months_to_process:
        logger.info(f"  {year}-{month:02d}")
    logger.info("")

    total_updated = 0

    for year, month in months_to_process:
        month_str = f"{year}-{month:02d}"
        logger.info("-" * 80)
        logger.info(f"Processing {month_str}")

        # Fetch actuals for this month
        actuals = fetch_monthly_actuals(year, month)

        if not actuals:
            logger.warning(f"  No data available for {month_str}, skipping")
            continue

        # Update predictions
        updated = update_predictions_for_month(year, month, actuals, args.dry_run)
        total_updated += updated

        logger.info(f"  Updated {updated} predictions for {month_str}")

    logger.info("")
    logger.info("=" * 80)
    logger.info("BACKFILL COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total predictions updated: {total_updated}")

    if args.dry_run:
        logger.info("(Dry run - no actual changes made)")


if __name__ == "__main__":
    main()
