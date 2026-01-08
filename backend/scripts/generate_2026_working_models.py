"""
Generate REAL 2026 predictions using the 5 working trained models
(skipping PatchTST/TimesNet which are broken)

Models:
- LGBM (sklearn, feature-based)
- RandomForest (sklearn, feature-based)
- AutoARIMA (statsforecast)
- AutoETS (statsforecast)
- SeasonalNaive (statsforecast)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.unified_inference import generate_forecast
from datetime import datetime, timedelta
import sqlite3
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

db_path = "/Users/olivialiau/retailPRED/data/retailpred.db"

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

# Only working models (skip PatchTST/TimesNet)
model_types = [
    "LGBM",
    "RandomForest",
    "AutoARIMA",
    "AutoETS",
    "SeasonalNaive",
]

def generate_weekly_dates_2026():
    """Generate all weekly dates for 2026"""
    start_date = datetime(2026, 1, 1)
    end_date = datetime(2026, 12, 31)

    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(weeks=1)

    return dates

def batch_insert_predictions(predictions_batch):
    """Insert a batch of predictions in a single transaction"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        for pred in predictions_batch:
            cursor.execute("""
                INSERT INTO prediction_log (
                    model_name, prediction_date, predicted_value,
                    confidence_interval_lower, confidence_interval_upper,
                    store_id, product_id
                ) VALUES (?, ?, ?, ?, ?, NULL, NULL)
            """, (
                pred['model_name'],
                pred['prediction_date'],
                pred['predicted_value'],
                pred.get('confidence_interval_lower'),
                pred.get('confidence_interval_upper')
            ))
        conn.commit()
        return len(predictions_batch)
    except Exception as e:
        conn.rollback()
        logger.error(f"Batch insert failed: {e}")
        raise
    finally:
        conn.close()

def main():
    logger.info("=" * 80)
    logger.info("Generating REAL 2026 Predictions (5 Working Models)")
    logger.info("=" * 80)
    logger.info("Models: LGBM, RandomForest, AutoARIMA, AutoETS, SeasonalNaive")
    logger.info("(Skipping PatchTST/TimesNet due to technical issues)")
    logger.info("")

    # Clear existing 2026 predictions
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM prediction_log WHERE prediction_date >= '2026-01-01'")
    conn.commit()
    conn.close()
    logger.info("Cleared existing 2026 predictions")

    dates = generate_weekly_dates_2026()
    logger.info(f"Generating predictions for {len(dates)} weeks")
    logger.info(f"Categories: {len(categories)}")
    logger.info(f"Models: {len(model_types)}")
    logger.info(f"Expected total: {len(dates) * len(categories) * len(model_types)} predictions")
    logger.info("")

    total_predictions = 0
    all_predictions = []

    for model_type in model_types:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Model: {model_type}")
        logger.info(f"{'=' * 80}")

        model_start = datetime.now()

        for date in dates:
            for category in categories:
                try:
                    # Use unified_inference.generate_forecast for all model types
                    forecast_list, metadata = generate_forecast(
                        category=category,
                        model_type=model_type,
                        weeks_ahead=1,
                        start_date=date
                    )

                    # Collect predictions
                    for point in forecast_list:
                        all_predictions.append({
                            'model_name': metadata['model_name'],
                            'prediction_date': point['date'],
                            'predicted_value': point['predicted_value'],
                            'confidence_interval_lower': point.get('confidence_interval_lower'),
                            'confidence_interval_upper': point.get('confidence_interval_upper'),
                        })

                        total_predictions += 1

                        # Progress logging every 100 predictions
                        if total_predictions % 100 == 0:
                            logger.info(f"  Progress: {total_predictions} predictions generated")

                except Exception as e:
                    logger.error(f"  ✗ {category} - {date}: {str(e)[:100]}")

        # Batch insert every 500 predictions
        if len(all_predictions) >= 500:
            inserted = batch_insert_predictions(all_predictions)
            logger.info(f"  → Inserted {inserted} predictions")
            all_predictions = []

        model_elapsed = (datetime.now() - model_start).total_seconds()
        logger.info(f"✓ {model_type} complete in {int(model_elapsed//60)}m {int(model_elapsed%60)}s")

    # Insert remaining
    if all_predictions:
        inserted = batch_insert_predictions(all_predictions)
        logger.info(f"→ Final batch: {inserted} predictions")

    logger.info("\n" + "=" * 80)
    logger.info(f"✓ Generated {total_predictions} total predictions for 2026")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
