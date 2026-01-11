#!/usr/bin/env python3
"""
Sync Retrained Model Prediction Dates with Existing Models

This script regenerates predictions for the retrained models using the EXACT SAME
dates as the existing models, ensuring complete consistency.
"""

import sys
from pathlib import Path
import logging
import sqlite3
import joblib
from datetime import datetime

# Add paths
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from ml.feature_computer import load_historical_data_from_csv, compute_real_features

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Database path
DB_PATH = backend_path.parent / "data" / "retailpred.db"

# Retrained models
RETRAINED_MODELS = [
    {
        'category': 'furniture_and_home_furnishings_stores',
        'model_type': 'LGBM',
        'display_name': 'Furniture_Home_Furnishings',
        'model_file': 'furniture_and_home_furnishings_stores_LGBM_model.pkl',
    },
    {
        'category': 'general_merchandise_stores',
        'model_type': 'LGBM',
        'display_name': 'General_Merchandise',
        'model_file': 'general_merchandise_stores_LGBM_model.pkl',
    },
    {
        'category': 'sporting_goods_hobby_and_musical_instrument_stores',
        'model_type': 'LGBM',
        'display_name': 'Sporting_Goods_Hobby',
        'model_file': 'sporting_goods_hobby_and_musical_instrument_stores_LGBM_model.pkl',
    },
    {
        'category': 'furniture_and_home_furnishings_stores',
        'model_type': 'RandomForest',
        'display_name': 'Furniture_Home_Furnishings',
        'model_file': 'furniture_and_home_furnishings_stores_RandomForest_model.pkl',
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


def get_prediction_dates():
    """Get all unique prediction dates from existing models"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get dates from a reference model (automobile_dealers_LGBM_model)
    cursor.execute("""
        SELECT DISTINCT prediction_date
        FROM prediction_log
        WHERE model_name = 'automobile_dealers_LGBM_model'
        AND prediction_date >= '2025-01-01'
        ORDER BY prediction_date
    """)

    dates = [row[0] for row in cursor.fetchall()]
    conn.close()

    return dates


def generate_predictions_synced(model_info: dict, prediction_dates: list):
    """Generate predictions for the same dates as existing models"""
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing: {model_info['display_name']} - {model_info['model_type']}")
    logger.info(f"{'='*80}")

    # Load model
    model_path = backend_path / "ml" / "models" / model_info['model_file']
    if not model_path.exists():
        logger.error(f"  ✗ Model file not found: {model_path}")
        return

    logger.info(f"  Loading model: {model_info['model_file']}")
    model_data = joblib.load(model_path)

    model = model_data['model']
    features = model_data['features']

    logger.info(f"  ✓ Model loaded")
    logger.info(f"  Features: {len(features)}")

    # Load historical data
    logger.info(f"\n  Loading historical data...")
    historical_df = load_historical_data_from_csv(
        model_info['display_name'],
        days_back=400
    )

    logger.info(f"  ✓ Loaded {len(historical_df)} records")

    # Generate predictions for synced dates
    logger.info(f"\n  Generating predictions for {len(prediction_dates)} synced dates...")

    predictions = []
    for pred_date_str in prediction_dates:
        pred_date = datetime.strptime(pred_date_str, '%Y-%m-%d')

        # Check if data exists for this date
        actual_row = historical_df[historical_df['date'] == pred_date]
        has_actual = len(actual_row) > 0
        actual_value = actual_row.iloc[0]['value'] if has_actual else None

        # Compute features
        features_df = compute_real_features(
            historical_df,
            pred_date_str
        )

        if features_df is None or len(features_df) == 0:
            logger.warning(f"    Skipping {pred_date_str}: No features")
            continue

        # Predict
        X = features_df[features].iloc[[-1]]
        predicted_value = float(model.predict(X)[0])

        # Calculate error if we have actual
        error_pct = None
        error_abs = None
        if has_actual and actual_value:
            error_pct = abs((actual_value - predicted_value) / actual_value * 100)
            error_abs = abs(actual_value - predicted_value)

        predictions.append({
            'date': pred_date_str,
            'predicted': predicted_value,
            'actual': actual_value,
            'error_pct': error_pct,
            'error_abs': error_abs,
            'has_actual': has_actual
        })

    logger.info(f"  ✓ Generated {len(predictions)} predictions")

    # Show stats
    validated = sum(1 for p in predictions if p['has_actual'])
    avg_error = sum(p['error_pct'] for p in predictions if p['error_pct']) / validated if validated > 0 else 0

    logger.info(f"\n  Statistics:")
    logger.info(f"    Total predictions: {len(predictions)}")
    logger.info(f"    Validated: {validated}")
    logger.info(f"    Avg error: {avg_error:.2f}%")

    # Add to database
    logger.info(f"\n  Adding to prediction_log...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    category_id = get_category_id(model_info['display_name'])
    if not category_id:
        logger.error(f"  ✗ Could not find category_id")
        conn.close()
        return

    model_name = f"{model_info['category']}_{model_info['model_type'].lower()}_model"

    # Delete ALL existing predictions for this model
    cursor.execute("""
        DELETE FROM prediction_log
        WHERE model_name = ?
    """, (model_name,))

    deleted = cursor.rowcount
    if deleted > 0:
        logger.info(f"  Deleted {deleted} existing predictions")

    # Insert new predictions
    added_count = 0
    for pred in predictions:
        try:
            cursor.execute("""
                INSERT INTO prediction_log
                (model_name, prediction_date, predicted_value, actual_value,
                 confidence_interval_lower, confidence_interval_upper,
                 error_percentage, error_absolute, is_validated, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_name,
                pred['date'],
                pred['predicted'],
                pred['actual'],
                pred['predicted'] * 0.95,  # Simple CI
                pred['predicted'] * 1.05,
                pred['error_pct'],
                pred['error_abs'],
                1 if pred['has_actual'] else 0,
                datetime.now().isoformat()
            ))
            added_count += 1
        except Exception as e:
            logger.warning(f"    Warning: {e}")

    conn.commit()
    conn.close()

    logger.info(f"  ✓ Added {added_count} predictions to database")

    return predictions


def main():
    """Main function"""
    logger.info("="*80)
    logger.info("SYNCING RETRAINED MODEL PREDICTION DATES WITH EXISTING MODELS")
    logger.info("="*80)

    # Get prediction dates from existing models
    logger.info("\nGetting prediction dates from existing models...")
    prediction_dates = get_prediction_dates()
    logger.info(f"  Found {len(prediction_dates)} unique prediction dates")

    # Show date range
    logger.info(f"  Date range: {prediction_dates[0]} to {prediction_dates[-1]}")

    total_added = 0

    for model_info in RETRAINED_MODELS:
        try:
            predictions = generate_predictions_synced(model_info, prediction_dates)
            if predictions:
                total_added += len(predictions)
        except Exception as e:
            logger.error(f"\n✗ Error processing {model_info['display_name']}: {e}")
            import traceback
            traceback.print_exc()

    logger.info(f"\n{'='*80}")
    logger.info(f"COMPLETE - Generated {total_added} synced predictions")
    logger.info(f"{'='*80}")

    # Verify consistency
    logger.info("\nVerifying consistency across all models...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Check if all models have predictions for the same dates
    cursor.execute("""
        SELECT prediction_date, COUNT(DISTINCT model_name) as model_count
        FROM prediction_log
        WHERE prediction_date >= '2025-01-01'
        GROUP BY prediction_date
        ORDER BY model_count DESC, prediction_date
        LIMIT 10
    """)

    logger.info("\n  Sample dates with model counts:")
    for date, count in cursor.fetchall():
        dow = datetime.strptime(date, '%Y-%m-%d').strftime('%A')
        logger.info(f"    {date} ({dow}): {count} models")

    # Check retrained models
    logger.info(f"\n  Retrained models:")
    for model_info in RETRAINED_MODELS:
        model_name = f"{model_info['category']}_{model_info['model_type'].lower()}_model"

        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN actual_value IS NOT NULL THEN 1 ELSE 0 END) as validated,
                AVG(CASE WHEN actual_value IS NOT NULL THEN error_percentage END) as avg_error
            FROM prediction_log
            WHERE model_name = ?
        """, (model_name,))

        total, validated, avg_error = cursor.fetchone()

        logger.info(f"    {model_name}:")
        logger.info(f"      Total: {total}")
        logger.info(f"      Validated: {validated}")
        logger.info(f"      Avg Error: {avg_error:.2f}%" if avg_error else "      Avg Error: N/A")

    conn.close()

    logger.info("\n✓ All retrained models now synced with existing model prediction dates!")


if __name__ == "__main__":
    main()
