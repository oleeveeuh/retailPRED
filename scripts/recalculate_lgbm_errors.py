#!/usr/bin/env python3
"""
Recalculate error metrics for the 3 fixed LGBM models
"""

import sqlite3
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "retailpred.db"

MODELS_TO_FIX = [
    "furniture_home_furnishings_LGBM_model",
    "general_merchandise_stores_LGBM_model",
    "sporting_goods_hobby_LGBM_model",
]


def recalculate_errors():
    """Recalculate error metrics for fixed models"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    logger.info("=" * 80)
    logger.info("Recalculating Error Metrics for Fixed LGBM Models")
    logger.info("=" * 80)

    for model_name in MODELS_TO_FIX:
        logger.info(f"\nProcessing: {model_name}")

        # Get all predictions with actual values
        cursor.execute("""
            SELECT
                id,
                prediction_date,
                predicted_value,
                actual_value
            FROM prediction_log
            WHERE model_name = ?
            AND actual_value IS NOT NULL
            ORDER BY prediction_date DESC
        """, (model_name,))

        predictions = cursor.fetchall()
        logger.info(f"Found {len(predictions)} predictions with actual values")

        if not predictions:
            logger.info("  No validated predictions")
            continue

        # Update error metrics
        for pred_id, pred_date, predicted, actual in predictions:
            # Calculate error metrics
            error_pct = abs((actual - predicted) / actual * 100)
            error_abs = abs(actual - predicted)

            cursor.execute("""
                UPDATE prediction_log
                SET error_percentage = ?,
                    error_absolute = ?
                WHERE id = ?
            """, (error_pct, error_abs, pred_id))

        conn.commit()

        # Calculate average error
        cursor.execute("""
            SELECT AVG(error_percentage) as avg_mape
            FROM prediction_log
            WHERE model_name = ?
            AND actual_value IS NOT NULL
        """, (model_name,))

        avg_mape = cursor.fetchone()[0]
        logger.info(f"  New average MAPE: {avg_mape:.2f}%")

    logger.info("\n" + "=" * 80)
    logger.info("Error metrics recalculated!")
    logger.info("=" * 80)

    conn.close()


if __name__ == "__main__":
    recalculate_errors()
