#!/usr/bin/env python3
"""
Generate historical predictions for January 2026
Populates the database with predictions for all categories and models
"""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from db.database import RetailPREDDatabase
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database path
db_path = Path(__file__).parent.parent.parent / "data" / "retailpred.db"
db = RetailPREDDatabase(db_path=str(db_path.absolute()))

# Categories
categories = [
    "total_sales",
    "building_material_and_garden_equipment",
    "automobile_dealers",
    "gasoline_stations",
    "food_and_beverage_stores",
    "health_and_personal_care_stores",
    "general_merchandise_stores",
    "furniture_and_home_furnishings_stores",
    "clothing_and_clothing_accessories_stores",
    "sporting_goods_hobby_and_musical_instrument_stores",
    "electronics_and_appliance_stores"
]

# Models to generate predictions for
models = ["LGBM", "PatchTST", "TimesNet", "AutoARIMA", "AutoETS", "SeasonalNaive"]

# Weekly dates for January 2026
january_dates = [
    "2026-01-01",
    "2026-01-08",
    "2026-01-15",
    "2026-01-22",
    "2026-01-29"
]

def clear_predictions():
    """Clear all existing predictions"""
    logger.info("Clearing existing predictions...")
    conn = db.get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM prediction_log")
        conn.commit()
        logger.info(f"Cleared all predictions from database")
    except Exception as e:
        logger.error(f"Error clearing predictions: {e}")
        conn.rollback()
    finally:
        conn.close()

def generate_prediction(category, model_name, start_date):
    """Generate a single prediction via API call"""
    import requests

    api_url = "http://localhost:8000/api/predict"

    params = {
        "category": category,
        "model_name": model_name,
        "weeks_ahead": 4,  # Generate 4 weeks of forecasts
        "granularity": "weekly",
        "start_date": start_date
    }

    try:
        logger.info(f"Generating prediction for {category} - {model_name} - {start_date}")
        response = requests.get(api_url, params=params, timeout=60)

        if response.status_code == 200:
            data = response.json()
            logger.info(f"✓ Generated {len(data.get('forecasts', []))} forecasts for {category} - {model_name} - {start_date}")
            return True
        else:
            logger.error(f"✗ Error {response.status_code}: {response.text}")
            return False

    except Exception as e:
        logger.error(f"✗ Exception: {e}")
        return False

def main():
    """Main function to populate predictions"""
    logger.info("=" * 80)
    logger.info("Starting January 2026 Prediction Population")
    logger.info("=" * 80)

    # Step 1: Clear existing predictions
    clear_predictions()

    # Step 2: Generate predictions for all combinations
    total_combinations = len(categories) * len(models) * len(january_dates)
    logger.info(f"Total predictions to generate: {total_combinations}")
    logger.info(f"Categories: {len(categories)}")
    logger.info(f"Models: {len(models)}")
    logger.info(f"Dates: {len(january_dates)}")

    success_count = 0
    fail_count = 0

    for date in january_dates:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Processing date: {date}")
        logger.info(f"{'=' * 80}")

        for category in categories:
            for model in models:
                success = generate_prediction(category, model, date)
                if success:
                    success_count += 1
                else:
                    fail_count += 1

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("POPULATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Successful: {success_count}/{total_combinations}")
    logger.info(f"Failed: {fail_count}/{total_combinations}")

    # Verify results
    conn = db.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM prediction_log")
    total_predictions = cursor.fetchone()[0]
    conn.close()

    logger.info(f"Total predictions in database: {total_predictions}")

if __name__ == "__main__":
    main()
