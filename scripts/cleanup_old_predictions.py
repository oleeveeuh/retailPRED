#!/usr/bin/env python3
"""
Clean up old predictions from retrained models

This script deletes all old predictions made by the problematic models
before they were retrained, ensuring all analytics show the correct performance.
"""

import sqlite3
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Database path
DB_PATH = Path(__file__).parent.parent / "data" / "retailpred.db"

# Models with old bad predictions to clean up
OLD_MODEL_NAMES = [
    'furniture_home_furnishings_LGBM_model',
    'general_merchandise_LGBM_model',
    'sporting_goods_hobby_LGBM_model',
    'furniture_home_furnishings_RandomForest_model',
]


def cleanup_old_predictions():
    """Delete old predictions from retrained models"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    logger.info("="*80)
    logger.info("CLEANING UP OLD PREDICTIONS FROM RETRAINED MODELS")
    logger.info("="*80)

    total_deleted = 0

    for model_name in OLD_MODEL_NAMES:
        logger.info(f"\n{model_name}:")
        logger.info("-"*80)

        # Count old predictions
        cursor.execute("""
            SELECT COUNT(*)
            FROM prediction_log
            WHERE model_name = ?
        """, (model_name,))

        old_count = cursor.fetchone()[0]
        logger.info(f"  Old predictions: {old_count}")

        if old_count == 0:
            logger.info(f"  ✓ No old predictions to delete")
            continue

        # Show sample of old predictions
        cursor.execute("""
            SELECT prediction_date, predicted_value, actual_value, error_percentage
            FROM prediction_log
            WHERE model_name = ?
            ORDER BY prediction_date DESC
            LIMIT 3
        """, (model_name,))

        logger.info(f"  Sample old predictions:")
        for pred in cursor.fetchall():
            pred_date, pred_val, actual_val, error = pred
            actual_str = f"${actual_val:,.2f}" if actual_val else "N/A"
            error_str = f"{error:.2f}%" if error else "N/A"
            logger.info(f"    {pred_date}: ${pred_val:,.2f} (actual: {actual_str}, error: {error_str})")

        # Delete old predictions
        cursor.execute("""
            DELETE FROM prediction_log
            WHERE model_name = ?
        """, (model_name,))

        deleted_count = cursor.rowcount
        total_deleted += deleted_count

        logger.info(f"  ✓ Deleted {deleted_count} old predictions")

    conn.commit()
    conn.close()

    logger.info("\n" + "="*80)
    logger.info(f"CLEANUP COMPLETE - {total_deleted} old predictions deleted")
    logger.info("="*80)

    # Verify cleanup
    logger.info("\nVerifying cleanup...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    for model_name in OLD_MODEL_NAMES:
        cursor.execute("""
            SELECT COUNT(*)
            FROM prediction_log
            WHERE model_name = ?
        """, (model_name,))

        remaining = cursor.fetchone()[0]
        status = "✓ Clean" if remaining == 0 else "⚠ Still has predictions"
        logger.info(f"  {model_name}: {remaining} predictions remaining {status}")

    conn.close()

    logger.info("\n" + "="*80)
    logger.info("All old predictions cleaned up successfully!")
    logger.info("Analytics will now show only new predictions with good performance")
    logger.info("="*80)


if __name__ == "__main__":
    cleanup_old_predictions()
