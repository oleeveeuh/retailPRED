#!/usr/bin/env python3
"""
Update Predictions with Newly Deployed Models

This script regenerates predictions for the 15 deployed models:
- 11 RandomForest models (all)
- 4 LGBM models (overfitting ones only)
"""

import sys
from pathlib import Path
import sqlite3
import pandas as pd
import logging
from datetime import datetime, timedelta
import joblib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Models that were deployed
DEPLOYED_MODELS = {
    'randomforest': [
        'automobile_dealers_RandomForest_model',
        'building_materials_RandomForest_model',
        'clothing_accessories_RandomForest_model',
        'electronics_and_appliances_RandomForest_model',
        'food_beverage_RandomForest_model',
        'furniture_home_furnishings_RandomForest_model',
        'gasoline_stations_RandomForest_model',
        'general_merchandise_RandomForest_model',
        'health_personal_care_RandomForest_model',
        'sporting_goods_hobby_RandomForest_model',
        'total_sales_RandomForest_model',
    ],
    'lgbm': [
        'sporting_goods_hobby_LGBM_model',
        'furniture_home_furnishings_LGBM_model',
        'building_materials_LGBM_model',
        'general_merchandise_LGBM_model',
    ]
}

# Category mapping for data loading
CATEGORY_MAP = {
    'automobile_dealers': '441',
    'building_materials': '443',
    'clothing_accessories': '452',
    'electronics_and_appliances': '4431',
    'food_beverage': '445',
    'furniture_home_furnishings': '442',
    'gasoline_stations': '448',
    'general_merchandise': '454',
    'health_personal_care': '447',
    'sporting_goods_hobby': '453',
    'total_sales': '4400',
}


def load_category_data(category: str):
    """Load category-specific data from database"""
    db_path = Path(__file__).parent.parent / "data/retailpred.db"
    conn = sqlite3.connect(db_path)

    query = """
        SELECT date as prediction_date, value as actual_value
        FROM time_series_data
        WHERE category_id = ?
        AND data_type = 'retail_sales'
        AND date >= '2020-01-01'
        ORDER BY date
    """

    df = pd.read_sql_query(query, conn, params=[category])
    conn.close()

    if df.empty:
        logger.warning(f"No data found for category: {category}")
        return None

    df['prediction_date'] = pd.to_datetime(df['prediction_date'])
    df = df.sort_values('prediction_date')

    logger.info(f"Loaded {len(df)} data points for {category}")
    return df


def create_features(df):
    """Create time series features"""
    import numpy as np

    # Lag features
    for lag in range(1, 13):
        df[f'lag_{lag}'] = df['actual_value'].shift(lag)

    # Rolling features
    df['rolling_mean_7'] = df['actual_value'].rolling(window=7, min_periods=1).mean()
    df['rolling_std_7'] = df['actual_value'].rolling(window=7, min_periods=1).std()
    df['rolling_min_7'] = df['actual_value'].rolling(window=7, min_periods=1).min()
    df['rolling_max_7'] = df['actual_value'].rolling(window=7, min_periods=1).max()

    # Time features
    df['month_sin'] = np.sin(2 * np.pi * df['prediction_date'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['prediction_date'].dt.month / 12)
    df['quarter_sin'] = np.sin(2 * np.pi * df['prediction_date'].dt.quarter / 4)
    df['quarter_cos'] = np.cos(2 * np.pi * df['prediction_date'].dt.quarter / 4)

    # Trend
    df['trend'] = range(len(df))

    # Drop rows with NaN
    df_clean = df.dropna()

    logger.info(f"Created {len(df_clean)} samples with features")
    return df_clean


def load_model(model_name: str):
    """Load a trained model"""
    models_dir = Path(__file__).parent.parent / "backend/ml/models"
    model_path = models_dir / f"{model_name}.pkl"

    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return None

    model = joblib.load(model_path)
    logger.info(f"Loaded model: {model_name}")
    return model


def update_predictions_for_model(model_name: str):
    """Update predictions for a single model"""
    try:
        # Extract category from model name
        category = None
        for cat_name, cat_id in CATEGORY_MAP.items():
            if cat_name in model_name.lower():
                category = cat_id
                break

        if not category:
            logger.warning(f"Could not determine category for {model_name}")
            return False

        # Load data
        df = load_category_data(category)
        if df is None:
            return False

        # Create features
        df = create_features(df)

        # Load model
        model = load_model(model_name)
        if model is None:
            return False

        # Prepare features
        feature_cols = [c for c in df.columns if c.startswith('lag_') or
                       c.startswith('rolling_') or c.startswith('month_') or
                       c.startswith('quarter_') or c == 'trend']

        X = df[feature_cols].values

        # Make predictions
        predictions = model.predict(X)

        # Update database
        db_path = Path(__file__).parent.parent / "data/retailpred.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        updated_count = 0
        for i, (date, actual) in enumerate(zip(df['prediction_date'], df['actual_value'])):
            pred_date = date.strftime('%Y-%m-%d')
            pred_value = float(predictions[i])

            # Check if prediction exists
            cursor.execute("""
                SELECT id FROM prediction_log
                WHERE model_name = ?
                AND prediction_date = ?
            """, (model_name, pred_date))

            result = cursor.fetchone()

            if result:
                # Update existing prediction
                cursor.execute("""
                    UPDATE prediction_log
                    SET predicted_value = ?
                    WHERE model_name = ?
                    AND prediction_date = ?
                """, (pred_value, model_name, pred_date))
                updated_count += 1
            else:
                # Insert new prediction
                cursor.execute("""
                    INSERT INTO prediction_log
                    (model_name, prediction_date, predicted_value, actual_value)
                    VALUES (?, ?, ?, ?)
                """, (model_name, pred_date, pred_value, float(actual)))

        conn.commit()
        conn.close()

        logger.info(f"✅ Updated {updated_count} predictions for {model_name}")
        return True

    except Exception as e:
        logger.error(f"❌ Error updating predictions for {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Update predictions for all deployed models"""

    print("=" * 80)
    print("UPDATING PREDICTIONS FOR DEPLOYED MODELS")
    print("=" * 80)
    print()

    all_models = DEPLOYED_MODELS['randomforest'] + DEPLOYED_MODELS['lgbm']
    print(f"Total models to update: {len(all_models)}")
    print(f"  RandomForest: {len(DEPLOYED_MODELS['randomforest'])}")
    print(f"  LGBM: {len(DEPLOYED_MODELS['lgbm'])}")
    print()
    print("=" * 80)

    success_count = 0
    failed_count = 0

    for model_name in all_models:
        logger.info(f"\nProcessing: {model_name}")
        if update_predictions_for_model(model_name):
            success_count += 1
        else:
            failed_count += 1

    print()
    print("=" * 80)
    print("UPDATE SUMMARY")
    print("=" * 80)
    print(f"Total models: {len(all_models)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failed_count}")
    print()
    print("✅ Predictions updated successfully!")


if __name__ == "__main__":
    main()
