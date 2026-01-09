#!/usr/bin/env python3
"""
Backfill error metrics for predictions that already have actual_value
Updates error_percentage and error_absolute for validated predictions
"""

import sqlite3
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "retailpred.db"

def backfill_error_metrics():
    """Backfill error metrics for predictions with actual values"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    logger.info("=" * 80)
    logger.info("Backfilling Error Metrics for Validated Predictions")
    logger.info("=" * 80)

    # Get all predictions with actual_value but missing error_percentage
    cursor.execute("""
        SELECT
            id,
            model_name,
            prediction_date,
            predicted_value,
            actual_value
        FROM prediction_log
        WHERE actual_value IS NOT NULL
        AND (error_percentage IS NULL OR error_absolute IS NULL)
        ORDER BY model_name, prediction_date DESC
    """)

    predictions_to_update = cursor.fetchall()
    logger.info(f"\nFound {len(predictions_to_update)} predictions with missing error metrics\n")

    if not predictions_to_update:
        logger.info("No error metrics to update")
        conn.close()
        return

    # Group by model
    updates_by_model = {}
    for pred in predictions_to_update:
        pred_id, model_name, pred_date, predicted_value, actual_value = pred

        if model_name not in updates_by_model:
            updates_by_model[model_name] = []

        # Calculate error metrics
        # MAPE = |actual - predicted| / actual * 100 (industry standard)
        error_pct = abs((actual_value - predicted_value) / actual_value * 100)
        error_abs = abs(actual_value - predicted_value)

        updates_by_model[model_name].append({
            'id': pred_id,
            'error_pct': error_pct,
            'error_abs': error_abs,
            'pred_date': pred_date,
            'predicted': predicted_value,
            'actual': actual_value
        })

    # Update predictions
    total_updated = 0
    for model_name, updates in sorted(updates_by_model.items()):
        logger.info(f"Updating {model_name}:")
        logger.info(f"  {len(updates)} predictions")

        avg_error_pct = sum(u['error_pct'] for u in updates) / len(updates)
        avg_error_abs = sum(u['error_abs'] for u in updates) / len(updates)

        for update in updates[:3]:  # Show first 3 examples
            logger.info(f"    {update['pred_date']}: "
                       f"error_pct={update['error_pct']:.2f}%, "
                       f"error_abs=${update['error_abs']:.2f}")

        logger.info(f"  Average error: {avg_error_pct:.2f}% (${avg_error_abs:.2f})\n")

        for update in updates:
            cursor.execute("""
                UPDATE prediction_log
                SET error_percentage = ?,
                    error_absolute = ?
                WHERE id = ?
            """, (
                update['error_pct'],
                update['error_abs'],
                update['id']
            ))
            total_updated += 1

    conn.commit()

    # Verify updates
    cursor.execute("""
        SELECT
            model_name,
            COUNT(*) as total,
            SUM(CASE WHEN actual_value IS NOT NULL THEN 1 ELSE 0 END) as validated,
            SUM(CASE WHEN error_percentage IS NOT NULL THEN 1 ELSE 0 END) as with_error_pct,
            AVG(error_percentage) as avg_error_pct
        FROM prediction_log
        WHERE actual_value IS NOT NULL
        GROUP BY model_name
        ORDER BY model_name
    """)

    logger.info("\n" + "=" * 80)
    logger.info("Error Metrics Status After Backfill:")
    logger.info("=" * 80)
    for row in cursor.fetchall():
        model_name, total, validated, with_error_pct, avg_error_pct = row
        logger.info(f"{model_name:50} | Validated: {validated:4} | With Errors: {with_error_pct:4} | Avg Error: {avg_error_pct:6.2f}%")

    conn.close()
    logger.info(f"\n✓ Successfully updated {total_updated} predictions with error metrics")

if __name__ == "__main__":
    backfill_error_metrics()
