#!/usr/bin/env python3
"""
Pre-compute SHAP values for LGBM predictions only
LGBM models have proper SHAP support via TreeExplainer
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
        return False, "Prediction not found"

    prediction = dict(row)
    model_name = prediction["model_name"]
    prediction_date = prediction["prediction_date"]

    # Skip if already has SHAP values
    if prediction.get("shap_values"):
        return True, "Already has SHAP"

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
        return False, f"Could not determine category for {model_name}"

    try:
        # Load model and features
        model_obj = load_model(category, "LGBM")
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

        return True, f"Saved {len(shap_dict)} SHAP values"

    except Exception as e:
        return False, f"Error: {str(e)[:100]}"

def main():
    """Main function"""
    logger.info("=" * 80)
    logger.info("Pre-computing SHAP values for LGBM predictions")
    logger.info("=" * 80)

    # Get all LGBM predictions that need SHAP values
    conn = db.get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, model_name, prediction_date
        FROM prediction_log
        WHERE model_name LIKE '%LGBM%'
        AND (shap_values IS NULL OR shap_values = '{}' OR length(shap_values) < 50)
        ORDER BY id
    """)

    predictions = cursor.fetchall()
    conn.close()

    logger.info(f"Found {len(predictions)} LGBM predictions that need SHAP values")

    if not predictions:
        logger.info("All LGBM predictions already have SHAP values!")
        return

    success_count = 0
    fail_count = 0

    # Process predictions
    for i, pred in enumerate(predictions, 1):
        prediction_id = pred[0]
        print(f"\rProcessing {i}/{len(predictions)}...", end='', flush=True)
        success, message = compute_and_save_shap(prediction_id)
        if success:
            success_count += 1
        else:
            fail_count += 1

    print()  # New line after progress

    logger.info("=" * 80)
    logger.info(f"COMPLETE: {success_count}/{len(predictions)} successful")
    logger.info(f"Failed: {fail_count}/{len(predictions)}")
    logger.info("=" * 80)

    # Show final stats
    conn = db.get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN shap_values IS NOT NULL AND length(shap_values) > 50 THEN 1 ELSE 0 END) as with_shap
        FROM prediction_log
        WHERE model_name LIKE '%LGBM%'
    """)
    stats = cursor.fetchone()
    conn.close()

    logger.info(f"\nLGBM Model Statistics:")
    logger.info(f"  Total LGBM predictions: {stats[0]}")
    logger.info(f"  With real SHAP values: {stats[1]}")
    logger.info(f"  Coverage: {stats[1]/stats[0]*100:.1f}%")

if __name__ == "__main__":
    main()
