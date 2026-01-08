#!/usr/bin/env python3
"""
Generate remaining 2026 predictions for AutoARIMA, AutoETS, SeasonalNaive, PatchTST, TimesNet
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging
import sqlite3

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from ml.multi_resolution_inference import generate_forecast

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database path
db_path = Path(__file__).parent.parent.parent / "data" / "retailpred.db"

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

# Remaining models (excluding LGBM which is already done)
models = ["AutoARIMA", "AutoETS", "SeasonalNaive", "PatchTST", "TimesNet"]

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
        logger.error(f"Error batch inserting predictions: {e}")
        return 0
    finally:
        conn.close()

def main():
    logger.info("=" * 80)
    logger.info("Starting 2026 Remaining Models Prediction Population")
    logger.info("=" * 80)

    # Generate dates
    dates = generate_weekly_dates_2026()

    total_predictions = len(categories) * len(models) * len(dates)
    logger.info(f"Total predictions to generate: {total_predictions}")
    logger.info(f"Categories: {len(categories)}")
    logger.info(f"Models: {len(models)} - {models}")
    logger.info(f"Weeks: {len(dates)}")
    logger.info("")

    # Collect all predictions in memory first
    all_predictions = []
    prediction_count = 0
    success_count = 0
    error_count = 0
    start_time = datetime.now()

    # Process by model type to track progress better
    for model_idx, model_name in enumerate(models, 1):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Model {model_idx}/{len(models)}: {model_name}")
        logger.info(f"{'=' * 80}")

        model_start = datetime.now()
        model_predictions = 0

        for week_idx, date in enumerate(dates, 1):
            if week_idx % 10 == 0:
                logger.info(f"  Week {week_idx}/{len(dates)} ({week_idx/len(dates)*100:.0f}%)")

            for category in categories:
                try:
                    # Generate forecast
                    forecast_list, metadata = generate_forecast(
                        category=category,
                        model_type=model_name,
                        weeks_ahead=1,
                        granularity="weekly",
                        start_date=date
                    )

                    # Add to batch
                    for point in forecast_list:
                        all_predictions.append({
                            'model_name': metadata["model_name"],
                            'prediction_date': point["date"],
                            'predicted_value': point["predicted_value"],
                            'confidence_interval_lower': point.get("confidence_interval_lower"),
                            'confidence_interval_upper': point.get("confidence_interval_upper"),
                        })
                        prediction_count += 1
                        success_count += 1
                        model_predictions += 1

                except Exception as e:
                    logger.error(f"  ✗ {category} - {date}: {str(e)[:100]}")
                    error_count += 1

            # Batch insert every week
            if all_predictions:
                inserted = batch_insert_predictions(all_predictions)
                all_predictions = []

        # Model complete
        model_elapsed = (datetime.now() - model_start).total_seconds()
        logger.info(f"\n✓ {model_name} complete: {model_predictions} predictions in {int(model_elapsed//60)}m {int(model_elapsed%60)}s")

    # Insert remaining predictions
    if all_predictions:
        inserted = batch_insert_predictions(all_predictions)
        logger.info(f"→ Final batch inserted {inserted} predictions")

    elapsed_total = (datetime.now() - start_time).total_seconds()

    logger.info("")
    logger.info("=" * 80)
    logger.info("GENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total time: {int(elapsed_total//60)}m {int(elapsed_total%60)}s")
    logger.info(f"Total predictions generated: {prediction_count}/{total_predictions}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Errors: {error_count}")

if __name__ == "__main__":
    main()
