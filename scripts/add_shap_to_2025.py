"""
Add SHAP values to existing 2025 predictions for LGBM and RandomForest models

This script updates existing 2025 predictions to include SHAP values
without regenerating the predictions themselves.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))

from ml.unified_inference import load_model
from ml.feature_computer_full import compute_full_features
from ml.feature_computer import load_historical_data_from_csv
from datetime import datetime, timedelta
import sqlite3
import json
import logging
import shap
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

db_path = "/Users/olivialiau/retailPRED/data/retailpred.db"

# Categories
categories = [
    "total_sales",
    "building_material_and_garden_equipment",
    "automobile_dealers",
    "gasoline_stations",
    "food_and_beverage_stores",
    "health_and_personal_care_stores",
    "general_merchandise_stores",
    "furniture_and_home_furnishings_stores",
    "clothing_and_clothing_accessories_stores",
    "sporting_goods_hobby_and_musical_instrument_stores",
    "electronics_and_appliance_stores"
]

# Only sklearn models support SHAP
model_types = ["LGBM", "RandomForest"]

# Category display names
category_display_names = {
    "total_sales": "Total Retail Sales",
    "building_material_and_garden_equipment": "Building Materials & Garden",
    "automobile_dealers": "Automobile Dealers",
    "gasoline_stations": "Gasoline Stations",
    "food_and_beverage_stores": "Food & Beverage Stores",
    "health_and_personal_care_stores": "Health & Personal Care",
    "general_merchandise_stores": "General Merchandise",
    "furniture_and_home_furnishings_stores": "Furniture & Home Furnishings",
    "clothing_and_clothing_accessories_stores": "Clothing & Accessories",
    "sporting_goods_hobby_and_musical_instrument_stores": "Sporting Goods & Hobby",
    "electronics_and_appliance_stores": "Electronics & Appliances",
}

def calculate_shap_values(model, features_df, model_type):
    """Calculate SHAP values for sklearn models"""
    try:
        # Create explainer based on model type
        if model_type == "LGBM":
            explainer = shap.TreeExplainer(model)
        elif model_type == "RandomForest":
            explainer = shap.TreeExplainer(model)
        else:
            logger.warning(f"SHAP not supported for model type: {model_type}")
            return None

        # Calculate SHAP values
        shap_values = explainer.shap_values(features_df)

        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # For multi-class

        # Create feature->shap_value dict
        shap_dict = {}
        for i, fname in enumerate(features_df.columns):
            try:
                shap_dict[fname] = float(shap_values[0][i])
            except:
                shap_dict[fname] = 0.0

        # Return as JSON string
        return json.dumps(shap_dict)

    except Exception as e:
        logger.warning(f"SHAP calculation failed: {e}")
        return None

def main():
    logger.info("=" * 80)
    logger.info("Adding SHAP Values to 2025 Predictions")
    logger.info("=" * 80)
    logger.info("")

    # Get existing 2025 predictions for LGBM and RandomForest
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, model_name, prediction_date
        FROM prediction_log
        WHERE prediction_date >= '2025-01-01'
          AND prediction_date < '2026-01-01'
          AND (model_name LIKE '%LGBM%' OR model_name LIKE '%RandomForest%')
          AND shap_values IS NULL
        ORDER BY prediction_date, model_name
    """)

    predictions = cursor.fetchall()
    logger.info(f"Found {len(predictions)} 2025 predictions without SHAP values")

    if len(predictions) == 0:
        logger.info("All 2025 predictions already have SHAP values!")
        return

    # Process predictions
    updated = 0
    for pred_id, model_name, prediction_date in predictions:
        try:
            # Parse model info
            if model_name.endswith('_model'):
                model_name_base = model_name[:-6]  # Remove '_model'
            else:
                model_name_base = model_name

            # Extract category and model type
            parts = model_name_base.rsplit('_', 1)
            if len(parts) != 2:
                logger.warning(f"Cannot parse model name: {model_name}")
                continue

            category, model_type = parts

            if model_type not in model_types:
                logger.warning(f"Unsupported model type: {model_type}")
                continue

            logger.info(f"Processing: {category} - {model_type} - {prediction_date}")

            # Load model
            model = load_model(category, model_type)

            # Load historical data
            display_name = category_display_names.get(category, category.replace("_", " ").title())
            historical_df = load_historical_data_from_csv(display_name, days_back=400)

            # Compute features
            features_df = compute_full_features(historical_df, prediction_date, category)

            # Align features to model expectations
            if hasattr(model, 'feature_names_in_'):
                expected_features = model.feature_names_in_
                aligned_data = {}
                for feat in expected_features:
                    if feat in features_df.columns:
                        aligned_data[feat] = features_df[feat].values[0]
                    else:
                        aligned_data[feat] = 0.0
                features_df = pd.DataFrame([aligned_data])

            # Calculate SHAP values
            shap_values_json = calculate_shap_values(model, features_df, model_type)

            if shap_values_json:
                # Update database
                cursor.execute("""
                    UPDATE prediction_log
                    SET shap_values = ?
                    WHERE id = ?
                """, (shap_values_json, pred_id))

                updated += 1

                if updated % 50 == 0:
                    conn.commit()
                    logger.info(f"  Progress: {updated}/{len(predictions)} updated")

        except Exception as e:
            logger.error(f"  ✗ {model_name} - {prediction_date}: {str(e)[:100]}")

    # Final commit
    conn.commit()
    conn.close()

    logger.info("")
    logger.info("=" * 80)
    logger.info(f"✓ Updated {updated} predictions with SHAP values")
    logger.info("=" * 80)

    # Verify
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            SUBSTR(prediction_date, 1, 4) as year,
            COUNT(*) as total,
            SUM(CASE WHEN shap_values IS NOT NULL THEN 1 ELSE 0 END) as with_shap,
            ROUND(CAST(SUM(CASE WHEN shap_values IS NOT NULL THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) * 100, 1) as pct_shap
        FROM prediction_log
        WHERE prediction_date >= '2025-01-01' AND prediction_date < '2026-01-01'
          AND (model_name LIKE '%LGBM%' OR model_name LIKE '%RandomForest%')
        GROUP BY year
    """)

    results = cursor.fetchall()
    conn.close()

    logger.info("")
    logger.info("2025 SHAP Value Summary:")
    for year, total, with_shap, pct_shap in results:
        logger.info(f"  {year}: {with_shap}/{total} ({pct_shap}%) with SHAP values")

if __name__ == "__main__":
    main()
