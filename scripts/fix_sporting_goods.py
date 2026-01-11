#!/usr/bin/env python3
"""
Fix sporting_goods_hobby_LGBM_model with correct scaling factor

This model was incorrectly scaled by 13.48x when it should be 7.54x
This script reverts and applies the correct scaling.
"""

import sqlite3
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "retailpred.db"

MODEL_NAME = "sporting_goods_hobby_LGBM_model"
WRONG_SCALING = 13.4771404187097
CORRECT_SCALING = 7.53853545251123


def fix_sporting_goods():
    """Fix sporting goods model with correct scaling"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    logger.info("=" * 80)
    logger.info(f"Fixing {MODEL_NAME}")
    logger.info("=" * 80)

    # Get all predictions
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
    """, (MODEL_NAME,))

    predictions = cursor.fetchall()
    logger.info(f"\nFound {len(predictions)} predictions")

    if not predictions:
        logger.info("No predictions to update")
        conn.close()
        return

    # Show current state
    logger.info("\nCurrent predictions (first 3):")
    for pred in predictions[:3]:
        pred_id, pred_date, predicted, actual, ci_lower, ci_upper = pred
        if actual:
            ratio = actual / predicted
            logger.info(f"  {pred_date}: predicted=${predicted:,.2f}, actual=${actual:,.2f}, ratio={ratio:.2f}")

    # Revert wrong scaling and apply correct scaling
    # Current values were scaled by WRONG_SCALING, need to scale by CORRECT_SCALING
    # So: new_value = (current_value / WRONG_SCALING) * CORRECT_SCALING
    # Simplified: new_value = current_value * (CORRECT_SCALING / WRONG_SCALING)
    adjustment_factor = CORRECT_SCALING / WRONG_SCALING

    logger.info(f"\nAdjustment factor: {adjustment_factor:.4f}")
    logger.info(f"(Correcting from {WRONG_SCALING:.4f}x to {CORRECT_SCALING:.4f}x)")

    updated_count = 0
    for pred in predictions:
        pred_id, pred_date, predicted, actual, ci_lower, ci_upper = pred

        # Apply adjustment
        new_predicted = predicted * adjustment_factor
        new_ci_lower = ci_lower * adjustment_factor if ci_lower else None
        new_ci_upper = ci_upper * adjustment_factor if ci_upper else None

        cursor.execute("""
            UPDATE prediction_log
            SET predicted_value = ?,
                confidence_interval_lower = ?,
                confidence_interval_upper = ?
            WHERE id = ?
        """, (new_predicted, new_ci_lower, new_ci_upper, pred_id))

        updated_count += 1

    conn.commit()
    logger.info(f"\n✓ Updated {updated_count} predictions")

    # Verify
    cursor.execute("""
        SELECT prediction_date, predicted_value, actual_value
        FROM prediction_log
        WHERE model_name = ?
        ORDER BY prediction_date DESC
        LIMIT 3
    """, (MODEL_NAME,))

    updated_predictions = cursor.fetchall()
    logger.info("\nAfter fix (first 3 predictions):")
    for pred in updated_predictions:
        pred_date, predicted, actual = pred
        if actual:
            error_pct = abs((actual - predicted) / actual * 100)
            logger.info(f"  {pred_date}: ${predicted:,.2f} (actual: ${actual:,.2f}, error: {error_pct:.1f}%)")

    # Calculate new average MAPE
    cursor.execute("""
        SELECT AVG(error_percentage)
        FROM prediction_log
        WHERE model_name = ?
        AND actual_value IS NOT NULL
    """, (MODEL_NAME,))

    avg_mape = cursor.fetchone()[0]
    logger.info(f"\nNew average MAPE: {avg_mape:.2f}%")

    logger.info("\n" + "=" * 80)
    logger.info("Fix Complete!")
    logger.info("=" * 80)

    conn.close()


if __name__ == "__main__":
    fix_sporting_goods()
