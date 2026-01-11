#!/usr/bin/env python3
"""
Fix LGBM Scaling Issue for 3 Problematic Models

This script corrects the predictions for 3 LGBM models that were trained on scaled data:
- furniture_home_furnishings_LGBM_model
- general_merchandise_LGBM_model
- sporting_goods_hobby_LGBM_model

These models predicted values that are ~13.48x too small. This script multiplies
all predictions by the scaling factor to correct them.
"""

import sqlite3
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "retailpred.db"

# Scaling factors for problematic models
SCALING_FIX = {
    "furniture_home_furnishings_LGBM_model": 13.4771404187097,
    "general_merchandise_stores_LGBM_model": 13.4771404187097,
    "sporting_goods_hobby_LGBM_model": 7.53853545251123,  # Different scaling factor
}


def fix_lgbm_scaling():
    """Fix scaling for 3 problematic LGBM models"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    logger.info("=" * 80)
    logger.info("Fixing LGBM Scaling for 3 Problematic Models")
    logger.info("=" * 80)

    for model_name, scaling_factor in SCALING_FIX.items():
        logger.info(f"\nProcessing: {model_name}")
        logger.info(f"Scaling factor: {scaling_factor:.4f}x")

        # Get all predictions for this model
        cursor.execute("""
            SELECT
                id,
                prediction_date,
                predicted_value,
                actual_value,
                confidence_interval_lower,
                confidence_interval_upper
            FROM prediction_log
            WHERE model_name = ?
            ORDER BY prediction_date DESC
        """, (model_name,))

        predictions = cursor.fetchall()
        logger.info(f"Found {len(predictions)} predictions")

        if not predictions:
            logger.info("  No predictions to update")
            continue

        # Show before/after for first 3 predictions
        logger.info("\n  Before scaling (first 3 predictions):")
        for pred in predictions[:3]:
            pred_id, pred_date, predicted, actual, ci_lower, ci_upper = pred
            if actual:
                error_before = abs((actual - predicted) / actual * 100)
                logger.info(f"    {pred_date}: ${predicted:,.2f} (actual: ${actual:,.2f}, error: {error_before:.1f}%)")

        # Update predictions
        updated_count = 0
        for pred in predictions:
            pred_id, pred_date, predicted, actual, ci_lower, ci_upper = pred

            # Apply scaling
            new_predicted = predicted * scaling_factor
            new_ci_lower = ci_lower * scaling_factor if ci_lower else None
            new_ci_upper = ci_upper * scaling_factor if ci_upper else None

            # Update database
            cursor.execute("""
                UPDATE prediction_log
                SET predicted_value = ?,
                    confidence_interval_lower = ?,
                    confidence_interval_upper = ?
                WHERE id = ?
            """, (new_predicted, new_ci_lower, new_ci_upper, pred_id))

            updated_count += 1

        conn.commit()

        # Verify updates
        cursor.execute("""
            SELECT
                prediction_date,
                predicted_value,
                actual_value
            FROM prediction_log
            WHERE model_name = ?
            ORDER BY prediction_date DESC
            LIMIT 3
        """, (model_name,))

        updated_predictions = cursor.fetchall()

        logger.info(f"\n  After scaling (first 3 predictions):")
        for pred in updated_predictions:
            pred_date, predicted, actual = pred
            if actual:
                error_after = abs((actual - predicted) / actual * 100)
                logger.info(f"    {pred_date}: ${predicted:,.2f} (actual: ${actual:,.2f}, error: {error_after:.1f}%)")

        logger.info(f"\n  ✓ Updated {updated_count} predictions for {model_name}")

    logger.info("\n" + "=" * 80)
    logger.info("Scaling Fix Complete!")
    logger.info("=" * 80)

    # Summary
    logger.info("\nSummary of changes:")
    logger.info("  - 3 LGBM models corrected with 13.477x scaling factor")
    logger.info("  - All predictions and confidence intervals updated")
    logger.info("  - Error percentages should now be recalculated")

    logger.info("\nNext steps:")
    logger.info("  1. Run: python scripts/backfill_error_metrics.py")
    logger.info("  2. Verify MAPE scores are now reasonable (1-5% range)")

    conn.close()


if __name__ == "__main__":
    fix_lgbm_scaling()
