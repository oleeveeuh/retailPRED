#!/usr/bin/env python3
"""
Generate historical predictions for entire 2026 year - Direct backend version
Much faster by calling backend functions directly instead of HTTP API
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

def normalize_category(category):
    """Normalize category name"""
    category_mapping = {
        "total_retail_sales": "total_sales",
        "building_materials": "building_material_and_garden_equipment",
        "automobiles": "automobile_dealers",
        "gasoline": "gasoline_stations",
        "food_beverage": "food_and_beverage_stores",
        "health_personal_care": "health_and_personal_care_stores",
        "general_merchandise": "general_merchandise_stores",
        "furniture": "furniture_and_home_furnishings_stores",
        "clothing": "clothing_and_clothing_accessories_stores",
        "sporting_goods": "sporting_goods_hobby_and_musical_instrument_stores",
        "electronics": "electronics_and_appliance_stores",
    }
    return category_mapping.get(category, category)

def generate_prediction_direct(category, model_name, start_date):
    """Generate prediction by calling backend functions directly"""
    import json

    # Import after sys.path update
    from ml.monthly_model_inference import generate_monthly_forecast
    from ml.multi_resolution_inference import generate_forecast

    normalized_category = normalize_category(category)
    forecast_start_date = start_date

    try:
        # Normalize model name
        model_name_mapping = {
            "lightgbm": "LGBM", "lgbm": "LGBM",
            "randomforest": "RandomForest", "random_forest": "RandomForest",
            "patchtst": "PatchTST",
            "timesnet": "TimesNet",
            "autoarima": "AutoARIMA",
            "autoets": "AutoETS",
            "seasonalnaive": "SeasonalNaive",
        }
        normalized_model_name = model_name_mapping.get(model_name.lower(), model_name)

        # Check if requesting monthly models
        monthly_models = ["autoarima", "autoets", "seasonalnaive"]
        if normalized_model_name and normalized_model_name.lower() in monthly_models:
            # Import monthly model inference
            forecast_list, metadata = generate_monthly_forecast(
                category=normalized_category,
                model_type=normalized_model_name.lower(),
                weeks_ahead=1,  # Just 1 week
                granularity="weekly",
                start_date=forecast_start_date
            )
        else:
            # Import multi-resolution inference
            forecast_list, metadata = generate_forecast(
                category=normalized_category,
                model_type=normalized_model_name,
                weeks_ahead=1,  # Just 1 week
                granularity="weekly",
                start_date=forecast_start_date
            )

        # Log predictions to database
        prediction_ids = []
        for point in forecast_list:
            try:
                pred_id = db.log_prediction(
                    model_name=metadata["model_name"],
                    prediction_date=point["date"],
                    predicted_value=point["predicted_value"],
                    features={},
                    confidence_interval_lower=point.get("confidence_interval_lower"),
                    confidence_interval_upper=point.get("confidence_interval_upper"),
                    shap_values=None,  # Skip SHAP for speed
                )
                prediction_ids.append(pred_id)
            except Exception as e:
                logger.error(f"Error logging prediction: {e}")
                return False, 0

        return True, len(forecast_list)

    except Exception as e:
        logger.error(f"✗ Exception for {category} - {model_name} - {start_date}: {str(e)[:100]}")
        return False, 0

def main():
    """Main function to populate predictions"""
    logger.info("=" * 80)
    logger.info("Starting 2026 Yearly Prediction Population (Direct Mode)")
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
        logger.info(f"\nWeek {week_num}/{len(weekly_dates)}: Processing date {date}")
        week_success = 0
        week_fail = 0

        for category in categories:
            for model in models:
                print(f"  {category} - {model}...", end='\r', flush=True)
                success, forecast_count = generate_prediction_direct(category, model, date)
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
