#!/usr/bin/env python3
"""
Train RandomForest and LGBM models with proper 73 features (excluding 'year')

This ensures proper time series forecasting without data leakage from the 'year' feature.
Models are trained on pre-computed features from multi-resolution CSV files.
"""

import sys
from pathlib import Path
import logging
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Category mappings
CATEGORIES = {
    'automobile_dealers': 'Automobile_Dealers',
    'building_material_and_garden_equipment': 'Building_Materials_Garden',
    'clothing_and_clothing_accessories_stores': 'Clothing_Accessories',
    'electronics_and_appliance_stores': 'Electronics_and_Appliances',
    'food_and_beverage_stores': 'Food_Beverage_Stores',
    'furniture_and_home_furnishings_stores': 'Furniture_Home_Furnishings',
    'gasoline_stations': 'Gasoline_Stations',
    'general_merchandise_stores': 'General_Merchandise',
    'health_and_personal_care_stores': 'Health_Personal_Care',
    'sporting_goods_hobby_and_musical_instrument_stores': 'Sporting_Goods_Hobby',
    'total_sales': 'Total_Retail_Sales',
}

# Model types to train
MODEL_TYPES = ['RandomForest', 'LGBM']

# Data directory
DATA_DIR = Path(__file__).parent.parent.parent / "project_root" / "data_multi_resolution"
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True, parents=True)


def load_training_data(category_key: str, category_display: str) -> pd.DataFrame:
    """Load pre-computed feature data from CSV"""
    # Map category_key to CSV filename
    # The CSV files use category_key with underscores (not display names)
    csv_name = category_key  # Use category_key directly as it matches CSV filenames

    csv_path = DATA_DIR / f"retail_{csv_name}_multi_resolution.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)

    # Exclude 'y' (target), 'index' (date string), and 'year' (data leakage)
    # This leaves 73 features for proper time series forecasting
    exclude_cols = ['y', 'index', 'year']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    logger.info(f"Loaded {len(df)} rows with {len(feature_cols)} features (excluding 'year')")

    return df, feature_cols


def prepare_train_test_split(df: pd.DataFrame, test_size: int = 52):
    """Split data into train/test sets (time-series aware)"""
    # Use last 52 weeks (1 year) for testing
    train_df = df.iloc[:-test_size].copy()
    test_df = df.iloc[-test_size:].copy()

    logger.info(f"Train: {len(train_df)} samples, Test: {len(test_df)} samples")

    return train_df, test_df


def train_randomforest(X_train, y_train, X_test, y_test):
    """Train RandomForest model"""
    logger.info("Training RandomForest...")

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)

    logger.info(f"  Train MAE: ${train_mae:,.2f}")
    logger.info(f"  Test MAE: ${test_mae:,.2f}")

    return model, {
        'train_mae': train_mae,
        'test_mae': test_mae,
    }


def train_lgbm(X_train, y_train, X_test, y_test):
    """Train LGBM model"""
    logger.info("Training LGBM...")

    model = LGBMRegressor(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    model.fit(X_train, y_train)

    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)

    logger.info(f"  Train MAE: ${train_mae:,.2f}")
    logger.info(f"  Test MAE: ${test_mae:,.2f}")

    return model, {
        'train_mae': train_mae,
        'test_mae': test_mae,
    }


def train_category(category_key: str, category_display: str):
    """Train all models for a category"""
    logger.info(f"\n{'='*80}")
    logger.info(f"Training: {category_display}")
    logger.info(f"{'='*80}\n")

    results = {}

    try:
        # Load data
        df, feature_cols = load_training_data(category_key, category_display)

        # Prepare features and target
        X = df[feature_cols]
        y = df['y']

        # Split train/test
        train_df, test_df = prepare_train_test_split(df)

        X_train = train_df[feature_cols]
        y_train = train_df['y']
        X_test = test_df[feature_cols]
        y_test = test_df['y']

        # Train each model type
        for model_type in MODEL_TYPES:
            try:
                if model_type == 'RandomForest':
                    model, metrics = train_randomforest(X_train, y_train, X_test, y_test)
                elif model_type == 'LGBM':
                    model, metrics = train_lgbm(X_train, y_train, X_test, y_test)
                else:
                    continue

                # Save model
                model_filename = f"{category_key}_{model_type}_model.pkl"
                model_path = MODELS_DIR / model_filename

                model_data = {
                    'model': model,
                    'model_name': model_filename,
                    'is_trained': True,
                    'training_time': datetime.now().isoformat(),
                    'features': feature_cols,
                    'n_features': len(feature_cols),
                    'metrics': metrics,
                }

                joblib.dump(model_data, model_path)
                logger.info(f"  ✅ Saved: {model_filename}")

                results[model_type] = {
                    'status': 'success',
                    'metrics': metrics,
                    'n_features': len(feature_cols),
                }

            except Exception as e:
                logger.error(f"  ❌ {model_type}: {e}")
                results[model_type] = {'status': 'failed', 'error': str(e)}

        return results

    except Exception as e:
        logger.error(f"Failed to train {category_display}: {e}")
        return {'status': 'failed', 'error': str(e)}


def main():
    """Main training function"""
    logger.info("="*80)
    logger.info("TRAINING MODELS WITH 73 FEATURES (EXCLUDING 'YEAR')")
    logger.info("="*80)
    logger.info(f"Categories: {len(CATEGORIES)}")
    logger.info(f"Model types: {MODEL_TYPES}")
    logger.info("="*80)

    all_results = {}
    start_time = datetime.now()

    for category_key, category_display in CATEGORIES.items():
        results = train_category(category_key, category_display)
        all_results[category_key] = results

    # Save summary
    duration = (datetime.now() - start_time).total_seconds()

    summary = {
        'training_date': datetime.now().isoformat(),
        'duration_seconds': duration,
        'n_features': 73,
        'features_excluded': ['y', 'index', 'year'],
        'categories_trained': len(CATEGORIES),
        'model_types': MODEL_TYPES,
        'results': all_results,
    }

    summary_path = Path(__file__).parent.parent.parent / "training_outputs" / "training_73features_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"Duration: {duration:.1f} seconds")
    logger.info(f"Summary: {summary_path}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
