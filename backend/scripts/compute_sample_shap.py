#!/usr/bin/env python3
"""
Pre-compute SHAP values for a sample of predictions
Computes SHAP for first week of each month for total_sales LGBM model
"""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from db.database import RetailPREDDatabase
import logging
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database path
db_path = Path(__file__).parent.parent.parent / "data" / "retailpred.db"
db = RetailPREDDatabase(db_path=str(db_path.absolute()))

def compute_and_save_shap(prediction_id):
    """Compute SHAP values for a prediction and save to database"""
    from ml.feature_computer import compute_shap_values
    from ml.multi_resolution_inference import load_model, prepare_features

    # Get prediction details
    conn = db.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM prediction_log WHERE id = ?", (prediction_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        logger.error(f"Prediction {prediction_id} not found")
        return False

    prediction = dict(row)
    model_name = prediction["model_name"]
    prediction_date = prediction["prediction_date"]

    # Extract category
    category_mapping = {
        "total_sales": "total_sales",
        "building_material_and_garden_equipment": "building_material_and_garden_equipment",
        "automobile_dealers": "automobile_dealers",
        "gasoline_stations": "gasoline_stations",
        "food_and_beverage_stores": "food_and_beverage_stores",
        "health_and_personal_care_stores": "health_and_personal_care_stores",
        "general_merchandise_stores": "general_merchandise_stores",
        "furniture_and_home_furnishings_stores": "furniture_and_home_furnishings_stores",
        "clothing_and_clothing_accessories_stores": "clothing_and_clothing_accessories_stores",
        "sporting_goods_hobby_and_musical_instrument_stores": "sporting_goods_hobby_and_musical_instrument_stores",
        "electronics_and_appliance_stores": "electronics_and_appliance_stores",
    }

    category = None
    for cat in category_mapping:
        if cat in model_name:
            category = category_mapping[cat]
            break

    if not category:
        logger.warning(f"Could not determine category for {model_name}")
        return False

    # Determine model type
    model_type = None
    for mt in ["LGBM", "PatchTST", "TimesNet"]:
        if mt in model_name:
            model_type = mt
            break

    if not model_type:
        logger.warning(f"Model {model_name} does not support SHAP")
        return False

    try:
        logger.info(f"Computing SHAP for prediction {prediction_id} ({category} - {model_type} - {prediction_date})")

        # Load model and features
        model_obj = load_model(category, model_type)
        features_df = prepare_features(category, prediction_date)

        # Compute SHAP values
        shap_results = compute_shap_values(
            model_obj,
            features_df,
            features_df.columns.tolist(),
            top_n=15
        )

        # Convert to dict format
        shap_dict = {}
        for result in shap_results:
            shap_dict[result["feature"]] = result["value"]

        # Update database
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE prediction_log SET shap_values = ? WHERE id = ?",
            (json.dumps(shap_dict), prediction_id)
        )
        conn.commit()
        conn.close()

        logger.info(f"✓ Saved {len(shap_dict)} SHAP values for prediction {prediction_id}")
        return True

    except Exception as e:
        logger.error(f"✗ Error computing SHAP for prediction {prediction_id}: {e}")
        return False

def main():
    """Main function"""
    logger.info("=" * 80)
    logger.info("Pre-computing SHAP values for sample predictions")
    logger.info("=" * 80)

    # Get sample predictions (first prediction of each month for total_sales LGBM)
    conn = db.get_connection()
    cursor = conn.cursor()

    # Get first prediction of each month
    cursor.execute("""
        SELECT id, model_name, prediction_date
        FROM prediction_log
        WHERE model_name = 'total_sales_LGBM_model'
        AND id IN (
            SELECT MIN(id)
            FROM prediction_log
            WHERE model_name = 'total_sales_LGBM_model'
            GROUP BY substr(prediction_date, 1, 7)
        )
        ORDER BY prediction_date
        LIMIT 12
    """)

    predictions = cursor.fetchall()
    conn.close()

    logger.info(f"Found {len(predictions)} predictions to process")

    success_count = 0
    for pred in predictions:
        prediction_id = pred[0]
        if compute_and_save_shap(prediction_id):
            success_count += 1

    logger.info("\n" + "=" * 80)
    logger.info(f"Complete: {success_count}/{len(predictions)} successful")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
