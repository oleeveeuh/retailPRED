#!/usr/bin/env python3
"""
Generate 2026 predictions for LGBM models only (best performing models)
Much faster without slow PatchTST/TimesNet models
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

# Only use LGBM (best model by MAPE)
models = ["LGBM"]

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
    logger.info("Starting 2026 LGBM Prediction Population (Fast Mode)")
    logger.info("=" * 80)

    # Clear existing 2026 predictions
    logger.info("Clearing existing 2026 predictions...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM prediction_log WHERE prediction_date >= '2026-01-01'")
        conn.commit()
        logger.info("✓ Cleared existing 2026 predictions")
    except Exception as e:
        logger.error(f"Error clearing predictions: {e}")
        conn.rollback()
    finally:
        conn.close()

    # Generate dates
    dates = generate_weekly_dates_2026()

    total_predictions = len(categories) * len(models) * len(dates)
    logger.info(f"Total predictions to generate: {total_predictions}")
    logger.info(f"Categories: {len(categories)}")
    logger.info(f"Models: {len(models)} (LGBM only)")
    logger.info(f"Weeks: {len(dates)}")
    logger.info("")

    # Collect all predictions in memory first
    all_predictions = []
    prediction_count = 0
    success_count = 0
    error_count = 0
    start_time = datetime.now()

    for week_idx, date in enumerate(dates, 1):
        logger.info(f"Week {week_idx}/{len(dates)}: Processing date {date}")

        for category in categories:
            try:
                # Generate forecast
                forecast_list, metadata = generate_forecast(
                    category=category,
                    model_type="LGBM",
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

                logger.info(f"  ✓ {category}: ${forecast_list[0]['predicted_value']:,.2f}")

            except Exception as e:
                logger.error(f"  ✗ {category}: {str(e)[:100]}")
                error_count += 1

        # Batch insert every week
        if all_predictions:
            inserted = batch_insert_predictions(all_predictions)
            logger.info(f"  → Batch inserted {inserted} predictions (total: {prediction_count}/{total_predictions})")
            all_predictions = []

        # Progress update every 10 weeks
        if week_idx % 10 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = week_idx / elapsed
            remaining = (len(dates) - week_idx) / rate
            logger.info(f"  Progress: {week_idx}/{len(dates)} weeks ({week_idx/len(dates)*100:.1f}%) - ETA: {int(remaining//60)}m {int(remaining%60)}s")

    # Insert remaining predictions
    if all_predictions:
        inserted = batch_insert_predictions(all_predictions)
        logger.info(f"  → Final batch inserted {inserted} predictions")

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
