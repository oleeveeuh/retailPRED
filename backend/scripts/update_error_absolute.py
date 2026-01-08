"""
Update Error Absolute Values for Validated Predictions

This script populates the error_absolute field in the prediction_log table
for validated predictions (where actual_value exists).
"""

import sys
from pathlib import Path
import logging
import sqlite3

# Add app directory to path
app_path = Path(__file__).parent.parent
sys.path.insert(0, str(app_path))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def update_error_absolute():
    """Update error_absolute for all validated predictions"""

    db_path = "/Users/olivialiau/retailPRED/data/retailpred.db"

    logger.info("=" * 80)
    logger.info("Updating Error Absolute Values for Validated Predictions")
    logger.info("=" * 80)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all validated predictions with actual values
    cursor.execute("""
        SELECT id, predicted_value, actual_value
        FROM prediction_log
        WHERE is_validated = 1 AND actual_value IS NOT NULL
    """)

    predictions = cursor.fetchall()
    logger.info(f"\nFound {len(predictions)} validated predictions to update\n")

    updated_count = 0
    total_error = 0

    for prediction_id, predicted, actual in predictions:
        # Calculate absolute error
        error_abs = abs(predicted - actual)

        cursor.execute("""
            UPDATE prediction_log
            SET error_absolute = ?
            WHERE id = ?
        """, (error_abs, prediction_id))

        updated_count += 1
        total_error += error_abs

        if updated_count <= 5:
            logger.info(f"  Prediction {prediction_id}: ${error_abs:.2f} (predicted: ${predicted:.2f}, actual: ${actual:.2f})")

    conn.commit()
    conn.close()

    avg_error = total_error / updated_count if updated_count > 0 else 0

    logger.info("\n" + "=" * 80)
    logger.info(f"✓ Updated {updated_count} predictions with error_absolute values")
    logger.info(f"Average absolute error: ${avg_error:.2f}")
    logger.info("=" * 80)


if __name__ == "__main__":
    update_error_absolute()
