#!/usr/bin/env python3
"""
Generate historical predictions for entire 2026 year
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

def generate_weekly_dates_2026():
    """Generate all weekly dates for 2026"""
    start_date = datetime(2026, 1, 1)
    end_date = datetime(2026, 12, 31)

    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=7)

    logger.info(f"Generated {len(dates)} weekly dates for 2026")
    return dates

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
        "weeks_ahead": 1,  # Just 1 week ahead for each weekly prediction
        "granularity": "weekly",
        "start_date": start_date
    }

    try:
        response = requests.get(api_url, params=params, timeout=120)

        if response.status_code == 200:
            data = response.json()
            forecast_count = len(data.get('forecasts', []))
            return True, forecast_count
        else:
            logger.error(f"✗ Error {response.status_code}: {response.text[:200]}")
            return False, 0

    except Exception as e:
        logger.error(f"✗ Exception for {category} - {model_name} - {start_date}: {str(e)[:100]}")
        return False, 0

def main():
    """Main function to populate predictions"""
    logger.info("=" * 80)
    logger.info("Starting 2026 Yearly Prediction Population")
    logger.info("=" * 80)

    # Step 1: Clear existing predictions
    clear_predictions()

    # Step 2: Generate all weekly dates for 2026
    weekly_dates = generate_weekly_dates_2026()

    # Step 3: Generate predictions for all combinations
    total_combinations = len(categories) * len(models) * len(weekly_dates)
    logger.info(f"Total predictions to generate: {total_combinations}")
    logger.info(f"Categories: {len(categories)}")
    logger.info(f"Models: {len(models)}")
    logger.info(f"Weeks: {len(weekly_dates)}")

    success_count = 0
    fail_count = 0
    total_forecasts = 0

    # Process week by week
    for week_num, date in enumerate(weekly_dates, 1):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Week {week_num}/{len(weekly_dates)}: Processing date {date}")
        logger.info(f"{'=' * 80}")

        week_success = 0
        week_fail = 0

        for category in categories:
            for model in models:
                print(f"  {category} - {model} - {date}...", end='\r', flush=True)
                success, forecast_count = generate_prediction(category, model, date)
                if success:
                    success_count += 1
                    total_forecasts += forecast_count
                    week_success += 1
                else:
                    fail_count += 1
                    week_fail += 1

        logger.info(f"  Week {week_num} complete: {week_success} success, {week_fail} failed")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("POPULATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Successful predictions: {success_count}/{total_combinations}")
    logger.info(f"Failed predictions: {fail_count}/{total_combinations}")
    logger.info(f"Total forecasts generated: {total_forecasts}")

    # Verify results
    conn = db.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM prediction_log")
    total_predictions = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT prediction_date) FROM prediction_log")
    unique_dates = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT model_name) FROM prediction_log")
    unique_models = cursor.fetchone()[0]

    cursor.execute("SELECT MIN(prediction_date), MAX(prediction_date) FROM prediction_log")
    min_date, max_date = cursor.fetchone()

    conn.close()

    logger.info(f"\nDatabase Statistics:")
    logger.info(f"  Total predictions: {total_predictions}")
    logger.info(f"  Unique dates: {unique_dates}")
    logger.info(f"  Unique models: {unique_models}")
    logger.info(f"  Date range: {min_date} to {max_date}")

if __name__ == "__main__":
    main()
