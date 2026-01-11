#!/usr/bin/env python3
"""
Retrain Furniture RandomForest Model Using Unified Pipeline

This will fix the Furniture RandomForest model that currently has 88.3% MAPE.
Uses the same 74-feature pipeline as the successfully retrained LGBM models.
"""

import sys
from pathlib import Path
import logging
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor

# Add paths
project_root = Path(__file__).parent.parent / "project_root"
backend_path = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(backend_path))

from ml.feature_computer import compute_real_features, load_historical_data_from_csv
from ml.data_loader import RETAIL_CATEGORIES

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Category display names
CATEGORY_DISPLAY_NAMES = {
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


def create_training_data(category_key: str, display_name: str, n_samples: int = 100):
    """
    Create training dataset using compute_real_features (74 features)
    """
    logger.info(f"\nCreating training data for {display_name}...")
    logger.info(f"  Loading historical data from CSV...")

    # Load historical data from CSV (same as inference)
    historical_df = load_historical_data_from_csv(display_name, days_back=400)

    if len(historical_df) < n_samples:
        logger.warning(f"  Only {len(historical_df)} records available")
        n_samples = len(historical_df) - 10

    logger.info(f"  Loaded {len(historical_df)} records")
    logger.info(f"  Generating {n_samples} training samples...")

    # Generate samples starting from most recent
    samples = []
    dates = []

    for i in range(n_samples):
        # Work backwards from most recent
        idx = len(historical_df) - 1 - i
        if idx < 0:
            break

        sample_date = historical_df.iloc[idx]['date']
        dates.append(sample_date)

        # Compute features using the unified feature computer (74 features)
        try:
            features_df = compute_real_features(
                historical_df,
                sample_date.strftime("%Y-%m-%d")
            )

            # Get the target value (actual sales for next week)
            if idx + 1 < len(historical_df):
                target = historical_df.iloc[idx + 1]['value']
            else:
                # Use current value as fallback
                target = historical_df.iloc[idx]['value']

            samples.append({
                'features': features_df.iloc[0].to_dict(),
                'target': target,
                'date': sample_date
            })

        except Exception as e:
            logger.warning(f"  Warning: Could not compute features for {sample_date}: {e}")
            continue

    logger.info(f"  ✓ Generated {len(samples)} valid samples")

    # Convert to DataFrame
    feature_list = [s['features'] for s in samples]
    targets = [s['target'] for s in samples]

    X = pd.DataFrame(feature_list)
    y = np.array(targets)

    # Fill NaN values with 0
    X = X.fillna(0)

    # Remove any infinite values
    X = X.replace([np.inf, -np.inf], 0)

    logger.info(f"  ✓ Feature matrix shape: {X.shape}")
    logger.info(f"  ✓ Target range: ${y.min():,.2f} - ${y.max():,.2f}")

    return X, y, dates


def train_randomforest_model(category_key: str, display_name: str):
    """
    Train RandomForest model using unified pipeline
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"TRAINING: {display_name} (RandomForest)")
    logger.info(f"{'='*80}")

    # Create training data
    X, y, dates = create_training_data(category_key, display_name)

    if X is None or len(X) < 10:
        logger.error(f"Not enough training samples ({len(X)}). Minimum 10 required.")
        return None

    # Train-test split (80-20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    logger.info(f"\nTraining split: {len(X_train)} samples")
    logger.info(f"Test split: {len(X_test)} samples")

    # Initialize RandomForest model
    # Using params that work well for other models
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )

    logger.info(f"\nTraining RandomForest model...")
    logger.info(f"  n_estimators: 100")
    logger.info(f"  max_depth: 10")
    logger.info(f"  min_samples_split: 5")

    # Train model
    model.fit(X_train, y_train)

    logger.info(f"  ✓ Training complete")

    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Calculate metrics
    train_mape = np.mean(np.abs((y_train - train_pred) / y_train)) * 100
    test_mape = np.mean(np.abs((y_test - test_pred) / y_test)) * 100

    train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))
    test_rmse = np.sqrt(np.mean((y_test - test_pred) ** 2))

    logger.info(f"\nModel Performance:")
    logger.info(f"  Train MAPE: {train_mape:.2f}%")
    logger.info(f"  Test MAPE:  {test_mape:.2f}%")
    logger.info(f"  Train RMSE: ${train_rmse:,.2f}")
    logger.info(f"  Test RMSE:  ${test_rmse:,.2f}")

    # Test on recent predictions
    logger.info(f"\nRecent predictions (last 3 weeks):")
    for i in range(min(3, len(test_pred))):
        actual = y_test[-(i+1)]
        pred = test_pred[-(i+1)]
        error_pct = abs((actual - pred) / actual) * 100
        logger.info(f"  {dates[-(i+1)]:%Y-%m-%d}: ${pred:,.2f} (actual: ${actual:,.2f}, error: {error_pct:.1f}%)")

    # Prepare model data for saving
    model_data = {
        'model': model,
        'model_type': 'RandomForest',
        'category': category_key,
        'category_display': display_name,
        'features': list(X.columns),
        'feature_count': len(X.columns),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'metrics': {
            'train_mape': float(train_mape),
            'test_mape': float(test_mape),
            'train_rmse': float(train_rmse),
            'test_rmse': float(test_rmse),
        },
        'trained_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    return model_data


def main():
    """Main retraining function"""
    logger.info("="*80)
    logger.info("RETRAINING FURNITURE RANDOMFOREST MODEL")
    logger.info("Using unified 74-feature pipeline")
    logger.info("="*80)

    category_key = "furniture_and_home_furnishings_stores"
    display_name = CATEGORY_DISPLAY_NAMES.get(category_key, category_key)

    # Output directory
    models_dir = backend_path / "ml" / "models"
    models_dir.mkdir(exist_ok=True, parents=True)

    try:
        # Train model
        model_data = train_randomforest_model(category_key, display_name)

        if model_data is None:
            logger.error(f"Failed to train model for {display_name}")
            return

        # Save model
        model_filename = f"{category_key}_RandomForest_model.pkl"
        model_path = models_dir / model_filename

        joblib.dump(model_data, model_path)
        logger.info(f"\n✓ Saved model: {model_filename}")

        results = {
            category_key: {
                'status': 'success',
                'model_file': model_filename,
                'metrics': model_data['metrics']
            }
        }

        # Save summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = backend_path / f"furniture_randomforest_retraining_summary_{timestamp}.json"

        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Print summary
        logger.info(f"\n{'='*80}")
        logger.info("RETRAINING SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"\n{display_name}:")
        logger.info(f"  ✓ Status: Success")
        logger.info(f"  ✓ Test MAPE: {model_data['metrics']['test_mape']:.2f}%")
        logger.info(f"  ✓ Model file: {model_filename}")

        logger.info(f"\n{'='*80}")
        logger.info("Model saved to: " + str(models_dir))
        logger.info("Summary saved to: " + str(summary_path))
        logger.info(f"{'='*80}\n")

        return results

    except Exception as e:
        logger.error(f"\n✗ Error training {display_name}: {e}")
        import traceback
        traceback.print_exc()

        return {
            category_key: {
                'status': 'failed',
                'error': str(e)
            }
        }


if __name__ == "__main__":
    main()
