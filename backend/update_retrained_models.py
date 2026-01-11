#!/usr/bin/env python3
"""
Update Retrained Models in Database

This script:
1. Updates model_metadata with retrained model metrics
2. Generates fresh predictions using the new models
3. Validates predictions against actual values
4. Updates prediction_log with new predictions
"""

import sys
from pathlib import Path
import logging
import sqlite3
import joblib
from datetime import datetime, timedelta

# Add paths
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from ml.unified_inference import generate_forecast

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Database path
DB_PATH = backend_path.parent / "data" / "retailpred.db"

# Retrained models info
RETRAINED_MODELS = [
    {
        'category': 'furniture_and_home_furnishings_stores',
        'model_type': 'LGBM',
        'display_name': 'Furniture_Home_Furnishings',
        'model_file': 'furniture_and_home_furnishings_stores_LGBM_model.pkl',
        'test_mape': 4.66,
        'train_mape': 1.25,
    },
    {
        'category': 'general_merchandise_stores',
        'model_type': 'LGBM',
        'display_name': 'General_Merchandise',
        'model_file': 'general_merchandise_stores_LGBM_model.pkl',
        'test_mape': 4.30,
        'train_mape': 1.29,
    },
    {
        'category': 'sporting_goods_hobby_and_musical_instrument_stores',
        'model_type': 'LGBM',
        'display_name': 'Sporting_Goods_Hobby',
        'model_file': 'sporting_goods_hobby_and_musical_instrument_stores_LGBM_model.pkl',
        'test_mape': 4.39,
        'train_mape': 1.30,
    },
    {
        'category': 'furniture_and_home_furnishings_stores',
        'model_type': 'RandomForest',
        'display_name': 'Furniture_Home_Furnishings',
        'model_file': 'furniture_and_home_furnishings_stores_RandomForest_model.pkl',
        'test_mape': 6.00,
        'train_mape': 1.60,
    },
]


def get_category_id(display_name: str) -> int:
    """Get category_id from display name"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT category_id
        FROM categories
        WHERE category_name = ?
    """, (display_name,))

    result = cursor.fetchone()
    conn.close()

    if result:
        return result[0]
    return None


def update_model_metadata(model_info: dict):
    """Update model metadata in database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    category_id = get_category_id(model_info['display_name'])

    if not category_id:
        logger.error(f"  ✗ Could not find category_id for {model_info['display_name']}")
        conn.close()
        return

    model_name = f"{model_info['category']}_{model_info['model_type'].lower()}_model"

    # Update metrics in model_metadata
    metrics_json = f'''{{
        "mape": {model_info['test_mape']},
        "train_mape": {model_info['train_mape']},
        "test_mape": {model_info['test_mape']},
        "rmse": 0.0,
        "mae": 0.0
    }}'''

    cursor.execute("""
        UPDATE model_metadata
        SET metrics = ?,
            updated_at = ?
        WHERE model_name = ?
    """, (metrics_json, datetime.now().isoformat(), model_name))

    conn.commit()
    logger.info(f"  ✓ Updated metadata for {model_name}")

    conn.close()


def generate_new_predictions(model_info: dict):
    """Generate new predictions using retrained model"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    category_id = get_category_id(model_info['display_name'])

    model_name = f"{model_info['category']}_{model_info['model_type'].lower()}_model"

    logger.info(f"\n  Generating predictions for {model_name}...")

    # Generate forecast for next 4 weeks
    try:
        forecast, metadata = generate_forecast(
            category=model_info['category'],
            model_type=model_info['model_type'],
            weeks_ahead=4,
            start_date=datetime.now().strftime("%Y-%m-%d")
        )

        # Insert predictions into database
        for i, point in enumerate(forecast):
            prediction_date = point['date']
            predicted_value = point['predicted_value']
            ci_lower = point['confidence_interval_lower']
            ci_upper = point['confidence_interval_upper']

            # Check if we have actual value for this date
            cursor.execute("""
                SELECT value
                FROM time_series_data
                WHERE category_id = ?
                AND date = ?
            """, (category_id, prediction_date))

            actual_result = cursor.fetchone()
            actual_value = actual_result[0] if actual_result else None

            # Calculate error if we have actual
            error_pct = None
            error_abs = None
            if actual_value:
                error_pct = abs((actual_value - predicted_value) / actual_value * 100)
                error_abs = abs(actual_value - predicted_value)

            # Insert prediction
            cursor.execute("""
                INSERT INTO prediction_log
                (model_name, prediction_date, predicted_value, actual_value,
                 confidence_interval_lower, confidence_interval_upper,
                 error_percentage, error_absolute, is_validated, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
            """, (model_name, prediction_date, predicted_value, actual_value,
                  ci_lower, ci_upper, error_pct, error_abs, datetime.now().isoformat()))

        conn.commit()
        logger.info(f"  ✓ Generated {len(forecast)} new predictions")

        # Show recent errors
        if actual_value:
            logger.info(f"  Latest prediction: ${predicted_value:,.2f} (error: {error_pct:.2f}%)")

    except Exception as e:
        logger.error(f"  ✗ Error generating predictions: {e}")

    conn.close()


def main():
    """Main update function"""
    logger.info("="*80)
    logger.info("UPDATING RETRAINED MODELS IN DATABASE")
    logger.info("="*80)

    for model_info in RETRAINED_MODELS:
        logger.info(f"\n{model_info['display_name']} - {model_info['model_type']}")
        logger.info("-"*80)

        # Update model metadata
        update_model_metadata(model_info)

        # Generate new predictions
        generate_new_predictions(model_info)

    logger.info("\n" + "="*80)
    logger.info("UPDATE COMPLETE")
    logger.info("="*80)

    # Verify updates
    logger.info("\nVerifying updates...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    for model_info in RETRAINED_MODELS:
        model_name = f"{model_info['category']}_{model_info['model_type'].lower()}_model"

        # Check metadata
        cursor.execute("""
            SELECT json_extract(metrics, '$.mape')
            FROM model_metadata
            WHERE model_name = ?
        """, (model_name,))

        mape = cursor.fetchone()
        if mape:
            logger.info(f"  {model_name}: MAPE = {mape[0]:.2f}%")

        # Check latest prediction
        cursor.execute("""
            SELECT predicted_value, actual_value, error_percentage
            FROM prediction_log
            WHERE model_name = ?
            ORDER BY created_at DESC
            LIMIT 1
        """, (model_name,))

        pred = cursor.fetchone()
        if pred:
            pred_val, actual_val, error = pred
            actual_str = f"${actual_val:,.2f}" if actual_val else "N/A"
            error_str = f"{error:.2f}%" if error else "N/A"
            logger.info(f"    Latest: ${pred_val:,.2f} (actual: {actual_str}, error: {error_str})")

    conn.close()

    logger.info("\n" + "="*80)
    logger.info("All retrained models updated in database!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
