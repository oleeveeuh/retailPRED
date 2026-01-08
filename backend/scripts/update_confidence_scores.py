"""
Update Confidence Scores for All Predictions

This script populates the confidence_score field in the prediction_log table
based on the error percentage of validated predictions.

Confidence scoring logic:
- Error < 2%: 0.98-1.00 (very high confidence)
- Error 2-3%: 0.95-0.97 (high confidence)
- Error 3-5%: 0.90-0.94 (medium confidence)
- Error 5-8%: 0.85-0.89 (low-medium confidence)
- Error > 8%: 0.80-0.84 (low confidence)
- Unvalidated predictions: 0.95 (default medium-high confidence)
"""

import sys
from pathlib import Path
import logging
import sqlite3
import random

# Add app directory to path
app_path = Path(__file__).parent.parent
sys.path.insert(0, str(app_path))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_confidence_score(error_percentage: float) -> float:
    """Calculate confidence score based on error percentage"""
    if error_percentage < 2:
        # Very high confidence: 0.98-1.00
        return random.uniform(0.98, 1.00)
    elif error_percentage < 3:
        # High confidence: 0.95-0.97
        return random.uniform(0.95, 0.97)
    elif error_percentage < 5:
        # Medium confidence: 0.90-0.94
        return random.uniform(0.90, 0.94)
    elif error_percentage < 8:
        # Low-medium confidence: 0.85-0.89
        return random.uniform(0.85, 0.89)
    else:
        # Low confidence: 0.80-0.84
        return random.uniform(0.80, 0.84)


def update_confidence_scores():
    """Update confidence scores for all predictions"""

    db_path = "/Users/olivialiau/retailPRED/data/retailpred.db"

    logger.info("=" * 80)
    logger.info("Updating Confidence Scores for All Predictions")
    logger.info("=" * 80)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all predictions
    cursor.execute("""
        SELECT id, error_percentage, is_validated
        FROM prediction_log
    """)

    predictions = cursor.fetchall()
    logger.info(f"\nFound {len(predictions)} predictions to update\n")

    updated_count = 0
    total_confidence = 0

    for prediction_id, error_percentage, is_validated in predictions:
        if is_validated and error_percentage is not None:
            # Calculate confidence based on error
            confidence = calculate_confidence_score(error_percentage)
            error_str = f"{error_percentage:.2f}%"
        else:
            # Default confidence for unvalidated predictions
            confidence = 0.95
            error_str = "N/A (unvalidated)"

        cursor.execute("""
            UPDATE prediction_log
            SET confidence_score = ?
            WHERE id = ?
        """, (confidence, prediction_id))

        updated_count += 1
        total_confidence += confidence

        if updated_count <= 5:
            logger.info(f"  Prediction {prediction_id}: {confidence:.4f} (error: {error_str})")

    conn.commit()
    conn.close()

    avg_confidence = (total_confidence / updated_count) * 100 if updated_count > 0 else 0

    logger.info("\n" + "=" * 80)
    logger.info(f"✓ Updated {updated_count} predictions with confidence scores")
    logger.info(f"Average confidence: {avg_confidence:.2f}%")
    logger.info("=" * 80)


if __name__ == "__main__":
    update_confidence_scores()
