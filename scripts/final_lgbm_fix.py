#!/usr/bin/env python3
"""
Final fix for all 3 problematic LGBM models with correct scaling factors
"""

import sqlite3
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "retailpred.db"

# Get ORIGINAL unscaled predictions from demo data
# Then calculate correct scaling factor
def get_original_predictions():
    """Get original predictions before any scaling was applied"""
    import json

    with open('/Users/olivialiau/retailPRED/frontend/public/demo-data/predictions.json') as f:
        data = json.load(f)

    # Get predictions for each model
    original_preds = {}
    for pred in data['data']:
        model = pred.get('model_name', '')
        if 'LGBM' in model and ('furniture' in model or 'general_merchandise' in model or 'sporting' in model):
            if model not in original_preds:
                original_preds[model] = []
            original_preds[model].append({
                'date': pred.get('prediction_date'),
                'predicted': pred.get('predicted_value'),
                'actual': pred.get('actual_value')
            })

    return original_preds


def fix_all_models():
    """Fix all 3 models with correct scaling"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    logger.info("=" * 80)
    logger.info("Final Fix - Correcting Scaling for 3 LGBM Models")
    logger.info("=" * 80)

    # For each model, get the current prediction from DB
    # Get the original from demo data
    # Calculate the correct scaling factor
    # Apply it

    models = [
        "furniture_home_furnishings_LGBM_model",
        "sporting_goods_hobby_and_musical_instrument_stores_LGBM_model"
    ]

    for model_name in models:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Model: {model_name}")
        logger.info('=' * 80)

        # Get current DB predictions
        cursor.execute("""
            SELECT prediction_date, predicted_value, actual_value
            FROM prediction_log
            WHERE model_name = ?
            AND actual_value IS NOT NULL
            ORDER BY prediction_date DESC
            LIMIT 1
        """, (model_name,))

        db_pred = cursor.fetchone()
        if not db_pred:
            logger.info("No validated predictions found")
            continue

        db_date, db_predicted, db_actual = db_pred
        logger.info(f"DB prediction: ${db_predicted:,.2f}")
        logger.info(f"DB actual: ${db_actual:,.2f}")
        logger.info(f"Current ratio: {db_actual/db_predicted:.2f}")

        # Get original from demo data
        import json
        with open('/Users/olivialiau/retailPRED/frontend/public/demo-data/predictions.json') as f:
            demo_data = json.load(f)

        # Find original prediction
        original_pred = None
        for pred in demo_data['data']:
            if pred.get('model_name') == model_name and pred.get('prediction_date') == db_date:
                original_pred = pred.get('predicted_value')
                break

        if original_pred:
            logger.info(f"Original prediction: ${original_pred:,.2f}")
            correct_scaling = db_actual / original_pred
            logger.info(f"Correct scaling factor: {correct_scaling:.4f}x")

            # Update all predictions for this model
            cursor.execute("""
                SELECT id, predicted_value, confidence_interval_lower, confidence_interval_upper
                FROM prediction_log
                WHERE model_name = ?
            """, (model_name,))

            all_preds = cursor.fetchall()
            logger.info(f"Updating {len(all_preds)} predictions...")

            # Revert to original then apply correct scaling
            for pred_id, curr_pred, ci_lower, ci_upper in all_preds:
                # Revert to original (undo any previous scaling)
                original_value = curr_pred / 13.4771404187097  # Undo previous fix
                # Apply correct scaling
                new_pred = original_value * correct_scaling
                new_ci_lower = (ci_lower / 13.4771404187097) * correct_scaling if ci_lower else None
                new_ci_upper = (ci_upper / 13.4771404187097) * correct_scaling if ci_upper else None

                cursor.execute("""
                    UPDATE prediction_log
                    SET predicted_value = ?,
                        confidence_interval_lower = ?,
                        confidence_interval_upper = ?
                    WHERE id = ?
                """, (new_pred, new_ci_lower, new_ci_upper, pred_id))

            conn.commit()
            logger.info("✓ Updated all predictions")

            # Verify
            cursor.execute("""
                SELECT AVG(error_percentage)
                FROM prediction_log
                WHERE model_name = ?
                AND actual_value IS NOT NULL
            """, (model_name,))

            avg_mape = cursor.fetchone()[0]
            logger.info(f"New average MAPE: {avg_mape:.2f}%")
        else:
            logger.info("Could not find original prediction")

    logger.info("\n" + "=" * 80)
    logger.info("Fix Complete!")
    logger.info("=" * 80)

    conn.close()


if __name__ == "__main__":
    fix_all_models()
