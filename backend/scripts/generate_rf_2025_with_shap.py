"""
Generate RandomForest predictions for 2025 with SHAP values

Creates validated predictions for all 11 categories
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.unified_inference import load_model, generate_forecast
from ml.feature_computer_full import compute_full_features
from ml.feature_computer import load_historical_data_from_csv
import shap
import json
from datetime import datetime, timedelta
import sqlite3
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

db_path = "/Users/olivialiau/retailPRED/data/retailpred.db"

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

# Map to old naming for 2025
category_to_old_name = {
    "total_sales": "total_sales",
    "building_material_and_garden_equipment": "building_materials",
    "automobile_dealers": "automobile_dealers",
    "gasoline_stations": "gasoline_stations",
    "food_and_beverage_stores": "food_beverage",
    "health_and_personal_care_stores": "health_personal_care",
    "general_merchandise_stores": "general_merchandise",
    "furniture_and_home_furnishings_stores": "furniture_home_furnishings",
    "clothing_and_clothing_accessories_stores": "clothing_accessories",
    "sporting_goods_hobby_and_musical_instrument_stores": "sporting_goods_hobby",
    "electronics_and_appliance_stores": "electronics_and_appliances",
}

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


def generate_weekly_dates_2025():
    """Generate all weekly dates for 2025"""
    start_date = datetime(2025, 1, 3)  # Thursday (first prediction date)
    end_date = datetime(2025, 12, 31)

    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(weeks=1)

    return dates


def calculate_shap_values(model, features_df):
    """Calculate SHAP values for RandomForest"""
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features_df)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_dict = {}
        for i, fname in enumerate(features_df.columns):
            try:
                shap_dict[fname] = float(shap_values[0][i])
            except:
                shap_dict[fname] = 0.0

        return json.dumps(shap_dict)
    except Exception as e:
        logger.warning(f"SHAP calculation failed: {e}")
        return None


def get_actual_value(category_id, date):
    """Get actual value from time_series_data table"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Try exact date match
    cursor.execute("""
        SELECT value FROM time_series_data
        WHERE category_id = ? AND date = ?
        AND value > 100
        ORDER BY date DESC LIMIT 1
    """, (category_id, date))

    row = cursor.fetchone()
    conn.close()

    return float(row[0]) if row else None


def generate_prediction_with_shap(category, category_old_name, date):
    """Generate prediction with SHAP values"""
    try:
        display_name = category_display_names.get(category, category)

        # Load model using unified_inference
        model = load_model(category, "RandomForest")

        # Load historical data
        historical_df = load_historical_data_from_csv(display_name, days_back=400)

        # Compute features
        features_df = compute_full_features(historical_df, date, category)

        # Align features to model expectations
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_
            aligned_data = {}
            for feat in expected_features:
                if feat in features_df.columns:
                    aligned_data[feat] = features_df[feat].values[0]
                else:
                    aligned_data[feat] = 0.0
            features_df = __import__('pandas').DataFrame([aligned_data])

        # Make prediction
        prediction = float(model.predict(features_df)[0])

        # Calculate SHAP values
        shap_values_json = calculate_shap_values(model, features_df)

        # Estimate confidence interval
        ci_lower = prediction * 0.98
        ci_upper = prediction * 1.02

        # Get actual value
        category_id_map = {
            "total_sales": "4400",
            "building_material_and_garden_equipment": "443",
            "automobile_dealers": "441",
            "gasoline_stations": "448",
            "food_and_beverage_stores": "445",
            "health_and_personal_care_stores": "447",
            "general_merchandise_stores": "454",
            "furniture_and_home_furnishings_stores": "442",
            "clothing_and_clothing_accessories_stores": "452",
            "sporting_goods_hobby_and_musical_instrument_stores": "453",
            "electronics_and_appliance_stores": "4431",
        }

        actual_value = get_actual_value(category_id_map[category], date)

        return {
            "predicted_value": round(prediction, 2),
            "confidence_interval_lower": round(ci_lower, 2),
            "confidence_interval_upper": round(ci_upper, 2),
            "shap_values": shap_values_json,
            "actual_value": actual_value
        }

    except Exception as e:
        logger.error(f"  Error generating prediction for {category} on {date}: {e}")
        return None


def batch_insert_predictions(predictions_batch):
    """Insert predictions in batch"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        for pred in predictions_batch:
            model_name = f"{pred['category_old_name']}_RandomForest_model"

            # Delete if exists
            cursor.execute("""
                DELETE FROM prediction_log
                WHERE model_name = ? AND prediction_date = ?
            """, (model_name, pred['date']))

            # Insert
            cursor.execute("""
                INSERT INTO prediction_log (
                    model_name, prediction_date, predicted_value,
                    confidence_interval_lower, confidence_interval_upper,
                    shap_values, actual_value, store_id, product_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?)
            """, (
                model_name,
                pred['date'],
                pred['predicted_value'],
                pred['ci_lower'],
                pred['ci_upper'],
                pred['shap_values'],
                pred['actual_value'],
                datetime.now().isoformat()
            ))
        conn.commit()
        return len(predictions_batch)
    except Exception as e:
        conn.rollback()
        logger.error(f"Batch insert failed: {e}")
        raise
    finally:
        conn.close()


def main():
    logger.info("=" * 80)
    logger.info("Generating RandomForest 2025 Predictions with SHAP")
    logger.info("=" * 80)

    dates = generate_weekly_dates_2025()
    logger.info(f"Dates: {len(dates)} weeks")
    logger.info(f"Categories: {len(categories)}")
    logger.info(f"Expected: {len(dates) * len(categories)} predictions")
    logger.info("")

    total_generated = 0
    all_predictions = []

    for category in categories:
        category_old_name = category_to_old_name[category]
        display_name = category_display_names.get(category, category)

        logger.info(f"\n{display_name}")

        for date in dates:
            try:
                result = generate_prediction_with_shap(category, category_old_name, date)

                if result:
                    all_predictions.append({
                        'category_old_name': category_old_name,
                        'date': date,
                        'predicted_value': result['predicted_value'],
                        'ci_lower': result['confidence_interval_lower'],
                        'ci_upper': result['confidence_interval_upper'],
                        'shap_values': result['shap_values'],
                        'actual_value': result['actual_value']
                    })

                    total_generated += 1

                    if total_generated % 100 == 0:
                        logger.info(f"  Progress: {total_generated} predictions")

            except Exception as e:
                logger.error(f"  ✗ {date}: {str(e)[:100]}")

        # Batch insert per category
        if len(all_predictions) >= 100:
            inserted = batch_insert_predictions(all_predictions)
            logger.info(f"  Inserted {inserted} predictions")
            all_predictions = []

    # Insert remaining
    if all_predictions:
        inserted = batch_insert_predictions(all_predictions)
        logger.info(f"Final batch: {inserted} predictions")

    logger.info("\n" + "=" * 80)
    logger.info(f"✓ Generated {total_generated} RandomForest 2025 predictions with SHAP")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
