"""
Generate 2026 Predictions for PatchTST and TimesNet Models

This script generates predictions for all 11 categories using PatchTST and TimesNet models
for the year 2026 and inserts them into the database.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import sqlite3
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Any

# Import the inference module
from ml.multi_resolution_inference import generate_forecast

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = "/Users/olivialiau/retailPRED/data/retailpred.db"

# All 11 retail categories
CATEGORIES = [
    "total_sales",
    "building_materials",
    "automobile_dealers",
    "gasoline_stations",
    "food_beverage",
    "health_personal_care",
    "general_merchandise",
    "furniture_home_furnishings",
    "clothing_accessories",
    "sporting_goods_hobby",
    "electronics_and_appliances",
]

# NeuralForecast models
MODEL_TYPES = ["PatchTST", "TimesNet"]


def generate_2026_predictions() -> Dict[str, int]:
    """Generate 2026 predictions for all categories and models"""
    stats = {"total": 0, "success": 0, "failed": 0}

    # Generate all 52 weeks of 2026
    start_date = "2026-01-01"
    weeks_ahead = 52

    for category in CATEGORIES:
        for model_type in MODEL_TYPES:
            stats["total"] += 1

            try:
                logger.info(f"Generating {category} - {model_type} for 2026...")

                # Generate forecast for all 52 weeks of 2026
                forecast_list, metadata = generate_forecast(
                    category=category,
                    model_type=model_type,
                    weeks_ahead=weeks_ahead,
                    granularity="weekly",
                    start_date=start_date
                )

                # Insert into database
                insert_predictions_to_db(
                    category=category,
                    model_type=model_type,
                    forecast_list=forecast_list,
                    metadata=metadata
                )

                stats["success"] += 1
                logger.info(f"  ✓ Inserted {len(forecast_list)} predictions")

            except Exception as e:
                stats["failed"] += 1
                logger.error(f"  ✗ Failed: {str(e)[:100]}")

    return stats


def insert_predictions_to_db(
    category: str,
    model_type: str,
    forecast_list: List[Dict[str, Any]],
    metadata: Dict[str, Any]
):
    """Insert predictions into database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    model_name = metadata["model_name"]
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Batch insert for performance
    predictions_to_insert = []

    for forecast in forecast_list:
        predictions_to_insert.append((
            model_name,
            None,  # store_id (not used for category-level predictions)
            None,  # product_id (not used for category-level predictions)
            forecast["date"],
            forecast["predicted_value"],
            None,  # actual_value (will be filled in later)
            forecast.get("confidence_interval_lower"),
            forecast.get("confidence_interval_upper"),
            None,  # features (NeuralForecast models use raw time series)
            None,  # shap_values (NeuralForecast models don't have SHAP)
            created_at
        ))

    # Batch insert
    cursor.executemany("""
        INSERT INTO prediction_log (
            model_name, store_id, product_id, prediction_date, predicted_value,
            actual_value, confidence_interval_lower, confidence_interval_upper,
            features, shap_values, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, predictions_to_insert)

    conn.commit()
    conn.close()


def main():
    logger.info("=" * 80)
    logger.info("Generating 2026 Predictions for PatchTST and TimesNet")
    logger.info("=" * 80)
    logger.info("")

    stats = generate_2026_predictions()

    logger.info("")
    logger.info("=" * 80)
    logger.info(f"✓ Total: {stats['total']}")
    logger.info(f"✓ Success: {stats['success']}")
    logger.info(f"✗ Failed: {stats['failed']}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
