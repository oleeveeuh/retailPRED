#!/usr/bin/env python3
"""
Generate Mock SHAP Values for Demo Data

Generates mock SHAP (SHapley Additive exPlanations) values for tree-based models
to enable model explainability in the demo interface.

Note: This generates DEMO/ MOCK values. For production, install shap library:
    pip install shap
"""

import sys
import sqlite3
import json
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = Path('data/retailpred.db')
CSV_DIR = Path('project_root/data_multi_resolution')

def load_category_data(category_id: str):
    """Load data for a category"""
    # Map category IDs to CSV files
    category_files = {
        '4400': 'retail_total_sales_multi_resolution.csv',
        '441': 'retail_automobile_dealers_multi_resolution.csv',
        '442': 'retail_furniture_and_home_furnishings_multi_resolution.csv',
        '443': 'retail_building_material_and_garden_equipment_multi_resolution.csv',
        '4431': 'retail_electronics_and_appliances_multi_resolution.csv',
        '445': 'retail_food_and_beverage_stores_multi_resolution.csv',
        '447': 'retail_health_and_personal_care_stores_multi_resolution.csv',
        '448': 'retail_gasoline_stations_multi_resolution.csv',
        '452': 'retail_clothing_and_clothing_accessories_stores_multi_resolution.csv',
        '453': 'retail_sporting_goods_and_hobby_multi_resolution.csv',
        '454': 'retail_general_merchandise_stores_multi_resolution.csv',
    }

    csv_file = category_files.get(category_id)
    if not csv_file:
        logger.warning(f"No CSV file found for category {category_id}")
        return None

    csv_path = CSV_DIR / csv_file
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['index'])
    df = df.sort_values('date')

    return df

def generate_mock_shap_values(feature_cols: list, n_predictions: int = 10):
    """Generate mock SHAP values for demo purposes"""
    shap_values = []

    # Time series feature importance (typically higher for recent lags)
    lag_importance = {
        'lag_1': 45.2,
        'lag_2': 28.7,
        'lag_3': 18.4,
        'lag_4': 12.1,
        'lag_5': 8.5,
        'lag_6': 6.2,
        'lag_7': 4.8,
        'lag_8': 3.5,
        'lag_9': 2.6,
        'lag_10': 2.0,
        'lag_11': 1.5,
        'lag_12': 1.2,
    }

    for i in range(n_predictions):
        feature_shap = {}

        for feature in feature_cols:
            # Generate realistic SHAP values
            if feature.startswith('lag_'):
                # Extract lag number
                parts = feature.split('_')
                lag_num = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
                base_value = lag_importance.get(f'lag_{min(lag_num, 12)}', 5.0)

                # Add some variation per prediction
                variation = np.random.normal(0, base_value * 0.1)
                feature_shap[feature] = float(base_value + variation)

            elif feature.startswith('rolling_'):
                # Rolling features have moderate importance
                base_value = np.random.uniform(3, 8)
                feature_shap[feature] = float(base_value)

            elif 'month' in feature or 'quarter' in feature:
                # Seasonal features
                base_value = np.random.uniform(2, 6)
                feature_shap[feature] = float(base_value)

            elif 'trend' in feature:
                # Trend features
                base_value = np.random.uniform(5, 12)
                feature_shap[feature] = float(base_value)

            else:
                # Other features
                base_value = np.random.uniform(0.5, 3)
                feature_shap[feature] = float(base_value)

        shap_values.append(feature_shap)

    return shap_values

def update_predictions_with_shap():
    """Update prediction_log table with mock SHAP values"""
    logger.info("\n" + "="*80)
    logger.info("GENERATING MOCK SHAP VALUES FOR DEMO DATA")
    logger.info("="*80)
    logger.info("Note: Using mock SHAP values. For production, install shap library:")
    logger.info("      pip install shap")

    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get all unique tree-based models
    cursor.execute("""
        SELECT DISTINCT model_name FROM prediction_log
        WHERE model_name LIKE '%lgbm%' OR model_name LIKE '%randomforest%'
        ORDER BY model_name
    """)

    models = cursor.fetchall()
    logger.info(f"\nFound {len(models)} tree-based models")

    updated_count = 0
    for (model_name,) in models:
        logger.info(f"Processing {model_name}")

        # Extract category ID from model name
        # Model names are like "total_sales_lgbm_model"
        parts = model_name.replace('_lgbm_model', '').replace('_randomforest_model', '').split('_')

        # Map to category ID
        category_map = {
            'total': '4400',
            'sales': '4400',
            'automobile': '441',
            'dealers': '441',
            'furniture': '442',
            'home': '442',
            'furnishings': '442',
            'building': '443',
            'materials': '443',
            'garden': '443',
            'electronics': '4431',
            'appliances': '4431',
            'food': '445',
            'beverage': '445',
            'health': '447',
            'personal': '447',
            'care': '447',
            'gasoline': '448',
            'stations': '448',
            'clothing': '452',
            'accessories': '452',
            'sporting': '453',
            'goods': '453',
            'hobby': '453',
            'general': '454',
            'merchandise': '454',
        }

        category_id = None
        for part in parts:
            if part in category_map:
                category_id = category_map[part]
                break

        if not category_id:
            logger.warning(f"  Could not determine category ID for {model_name}")
            continue

        # Load category data to get feature names
        df = load_category_data(category_id)
        if df is None:
            continue

        # Get feature columns (exclude date, index, year, y)
        exclude_cols = ['date', 'index', 'year', 'y']
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # Generate mock SHAP values
        shap_values_list = generate_mock_shap_values(feature_cols, n_predictions=10)

        # Get predictions for this model (last 10)
        cursor.execute("""
            SELECT id FROM prediction_log
        WHERE model_name = ?
        ORDER BY prediction_date DESC
        LIMIT 10
        """, (model_name,))

        rows = cursor.fetchall()

        if not rows:
            logger.warning(f"  No predictions found for {model_name}")
            continue

        # Update each prediction with SHAP values
        for i, row in enumerate(rows):
            prediction_id = row[0]
            if i < len(shap_values_list):
                shap_json = json.dumps(shap_values_list[i])
                cursor.execute("""
                    UPDATE prediction_log
                    SET shap_values = ?
                    WHERE id = ?
                """, (shap_json, prediction_id))
                updated_count += 1

        logger.info(f"  Updated {len(rows)} predictions for {model_name}")

    # Commit changes
    conn.commit()
    conn.close()

    logger.info("\n" + "="*80)
    logger.info(f"✓ Updated {updated_count} predictions with mock SHAP values")
    logger.info("="*80)

    return updated_count

if __name__ == '__main__':
    count = update_predictions_with_shap()
    sys.exit(0 if count > 0 else 1)
