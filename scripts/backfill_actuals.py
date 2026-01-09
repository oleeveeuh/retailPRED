#!/usr/bin/env python3
"""
Backfill actual values for predictions that should be validated
Updates prediction_log table with actual values from time_series_data for dates that have passed
"""

import sqlite3
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "retailpred.db"

def backfill_actuals():
    """Backfill actual values for predictions"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    logger.info("Backfilling actual values for predictions...")

    # Get all predictions without actual values for past dates
    cursor.execute("""
        SELECT
            pl.id,
            pl.model_name,
            pl.prediction_date,
            pl.predicted_value,
            t.value as actual_value,
            t.category_id
        FROM prediction_log pl
        LEFT JOIN time_series_data t ON t.date = pl.prediction_date
        WHERE pl.actual_value IS NULL
        AND pl.prediction_date < date('now')
        ORDER BY pl.prediction_date DESC, pl.model_name
    """)

    predictions_to_update = cursor.fetchall()
    logger.info(f"Found {len(predictions_to_update)} predictions to update")

    if not predictions_to_update:
        logger.info("No predictions to update")
        conn.close()
        return

    # Group by model and date
    updates_by_model = {}
    for pred in predictions_to_update:
        pred_id, model_name, pred_date, predicted_value, actual_value, category_id = pred

        if actual_value is None:
            logger.warning(f"  ⚠ No actual value found for {model_name} on {pred_date}")
            continue

        if model_name not in updates_by_model:
            updates_by_model[model_name] = []

        # Calculate error metrics
        error_pct = abs((actual_value - predicted_value) / actual_value * 100)
        error_abs = abs(actual_value - predicted_value)

        # Simple confidence score based on error percentage
        confidence_score = max(0, min(100, 100 - error_pct * 2))

        updates_by_model[model_name].append({
            'id': pred_id,
            'actual_value': actual_value,
            'error_pct': error_pct,
            'error_abs': error_abs,
            'confidence_score': confidence_score
        })

    # Update predictions
    total_updated = 0
    for model_name, updates in updates_by_model.items():
        logger.info(f"\nUpdating {model_name}:")
        logger.info(f"  {len(updates)} predictions")

        for update in updates:
            cursor.execute("""
                UPDATE prediction_log
                SET actual_value = ?,
                    error_percentage = ?,
                    error_absolute = ?,
                    confidence_score = ?,
                    validated_at = datetime('now')
                WHERE id = ?
            """, (
                update['actual_value'],
                update['error_pct'],
                update['error_abs'],
                update['confidence_score'],
                update['id']
            ))
            total_updated += 1

    conn.commit()

    # Verify updates
    cursor.execute("""
        SELECT
            model_name,
            COUNT(*) as total,
            COUNT(actual_value) as validated,
            COUNT(*) - COUNT(actual_value) as unvalidated
        FROM prediction_log
        WHERE prediction_date < date('now')
        GROUP BY model_name
        ORDER BY model_name
    """)

    logger.info("\n" + "=" * 80)
    logger.info("Validation Status After Backfill:")
    logger.info("=" * 80)
    for row in cursor.fetchall():
        model_name, total, validated, unvalidated = row
        logger.info(f"{model_name:50} | Total: {validated+unvalidated:4} | Validated: {validated:4} | Unvalidated: {unvalidated:4}")

    conn.close()
    logger.info(f"\n✓ Successfully updated {total_updated} predictions")

if __name__ == "__main__":
    backfill_actuals()
