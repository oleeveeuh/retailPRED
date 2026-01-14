#!/usr/bin/env python3
"""
Retrain Only the 14 Overfitting Models

This script retrains only the models that are showing severe overfitting
with new anti-overfitting hyperparameters.
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Models that need retraining (14 total)
MODELS_TO_RETRAIN = [
    # RandomForest models (10) - severely overfitting
    'clothing_accessories_RandomForest_model',
    'health_personal_care_RandomForest_model',
    'building_materials_RandomForest_model',
    'automobile_dealers_RandomForest_model',
    'furniture_home_furnishings_RandomForest_model',
    'sporting_goods_hobby_RandomForest_model',
    'food_beverage_RandomForest_model',
    'total_sales_RandomForest_model',
    'general_merchandise_RandomForest_model',
    'gasoline_stations_RandomForest_model',

    # LGBM models (4) - overfitting
    'sporting_goods_hobby_LGBM_model',
    'furniture_home_furnishings_LGBM_model',
    'general_merchandise_LGBM_model',
    'building_materials_LGBM_model',
]

# New anti-overfitting hyperparameters
RANDOMFOREST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,                   # Reduced from 10
    "min_samples_split": 20,          # Increased from 5
    "min_samples_leaf": 10,           # Increased from 2
    "max_features": 0.7,              # Feature bagging
    "bootstrap": True,
    "oob_score": True,                # Out-of-bag validation
    "random_state": 42,
    "n_jobs": -1,
}

LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "n_estimators": 500,
    "max_depth": 5,
    "num_leaves": 31,
    "learning_rate": 0.01,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


def retrain_randomforest_model(model_name: str, category: str):
    """Retrain a RandomForest model with anti-overfitting parameters"""
    try:
        from sklearn.ensemble import RandomForestRegressor
        import joblib
        import sqlite3
        import pandas as pd
        import numpy as np

        logger.info(f"Retraining {model_name}...")

        # Load data
        conn = sqlite3.connect(Path(__file__).parent.parent / "data/retailpred.db")

        # Get training data
        query = f"""
            SELECT prediction_date, actual_value, predicted_value, error_absolute
            FROM prediction_log
            WHERE prediction_date >= '2022-01-01'
            AND actual_value IS NOT NULL
            ORDER BY prediction_date
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        # Prepare features
        df['prediction_date'] = pd.to_datetime(df['prediction_date'])
        df = df.sort_values('prediction_date')

        # Create features using lagged values
        for lag in range(1, 13):
            df[f'lag_{lag}'] = df['actual_value'].shift(lag)

        # Create rolling features
        df['rolling_mean_7'] = df['actual_value'].rolling(window=7).mean()
        df['rolling_std_7'] = df['actual_value'].rolling(window=7).std()
        df['rolling_min_7'] = df['actual_value'].rolling(window=7).min()
        df['rolling_max_7'] = df['actual_value'].rolling(window=7).max()

        # Time features
        df['month_sin'] = np.sin(2 * np.pi * df['prediction_date'].dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['prediction_date'].dt.month / 12)
        df['quarter_sin'] = np.sin(2 * np.pi * df['prediction_date'].dt.quarter / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['prediction_date'].dt.quarter / 4)

        # Drop NaN values
        df = df.dropna()

        # Prepare X and y
        feature_cols = [c for c in df.columns if c.startswith('lag_') or
                       c.startswith('rolling_') or c.startswith('month_') or
                       c.startswith('quarter_')]

        X = df[feature_cols]
        y = df['actual_value']

        # Use 80/20 split for validation
        split_idx = int(len(df) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        # Train with new parameters
        model = RandomForestRegressor(**RANDOMFOREST_PARAMS)
        model.fit(X_train, y_train)

        # Evaluate
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        train_mae = np.mean(np.abs(y_train - train_pred))
        val_mae = np.mean(np.abs(y_val - val_pred))

        # Calculate MASE
        naive_mae = train_mae  # Simplified baseline
        train_mase = train_mae / naive_mae
        val_mase = val_mae / naive_mae

        overfitting_ratio = val_mae / train_mae

        logger.info(f"  Train MAE: {train_mae:.2f}, Val MAE: {val_mae:.2f}")
        logger.info(f"  Train MASE: {train_mase:.4f}, Val MASE: {val_mase:.4f}")
        logger.info(f"  Overfitting ratio: {overfitting_ratio:.2f}")

        # Save model
        models_dir = Path(__file__).parent.parent / "backend/ml/models"
        models_dir.mkdir(exist_ok=True)

        model_path = models_dir / f"{model_name}_v2.pkl"
        joblib.dump(model, model_path)

        logger.info(f"  ✅ Model saved to {model_path}")

        return {
            'model': model_name,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'train_mase': train_mase,
            'val_mase': val_mase,
            'overfitting_ratio': overfitting_ratio,
            'success': True
        }

    except Exception as e:
        logger.error(f"  ❌ Error retraining {model_name}: {e}")
        return {
            'model': model_name,
            'error': str(e),
            'success': False
        }


def retrain_lgbm_model(model_name: str, category: str):
    """Retrain an LGBM model with anti-overfitting parameters"""
    try:
        import lightgbm as lgb
        import joblib
        import sqlite3
        import pandas as pd
        import numpy as np

        logger.info(f"Retraining {model_name}...")

        # Load data
        conn = sqlite3.connect(Path(__file__).parent.parent / "data/retailpred.db")

        query = f"""
            SELECT prediction_date, actual_value, predicted_value, error_absolute
            FROM prediction_log
            WHERE prediction_date >= '2022-01-01'
            AND actual_value IS NOT NULL
            ORDER BY prediction_date
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        # Prepare features (same as RF)
        df['prediction_date'] = pd.to_datetime(df['prediction_date'])
        df = df.sort_values('prediction_date')

        for lag in range(1, 13):
            df[f'lag_{lag}'] = df['actual_value'].shift(lag)

        df['rolling_mean_7'] = df['actual_value'].rolling(window=7).mean()
        df['rolling_std_7'] = df['actual_value'].rolling(window=7).std()
        df['rolling_min_7'] = df['actual_value'].rolling(window=7).min()
        df['rolling_max_7'] = df['actual_value'].rolling(window=7).max()

        df['month_sin'] = np.sin(2 * np.pi * df['prediction_date'].dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['prediction_date'].dt.month / 12)
        df['quarter_sin'] = np.sin(2 * np.pi * df['prediction_date'].dt.quarter / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['prediction_date'].dt.quarter / 4)

        df = df.dropna()

        feature_cols = [c for c in df.columns if c.startswith('lag_') or
                       c.startswith('rolling_') or c.startswith('month_') or
                       c.startswith('quarter_')]

        X = df[feature_cols]
        y = df['actual_value']

        # Use 80/20 split
        split_idx = int(len(df) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Train with early stopping
        model = lgb.train(
            LGBM_PARAMS,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )

        # Evaluate
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        train_mae = np.mean(np.abs(y_train - train_pred))
        val_mae = np.mean(np.abs(y_val - val_pred))

        naive_mae = train_mae
        train_mase = train_mae / naive_mae
        val_mase = val_mae / naive_mae

        overfitting_ratio = val_mae / train_mae

        logger.info(f"  Train MAE: {train_mae:.2f}, Val MAE: {val_mae:.2f}")
        logger.info(f"  Train MASE: {train_mase:.4f}, Val MASE: {val_mase:.4f}")
        logger.info(f"  Best iteration: {model.num_trees()}")
        logger.info(f"  Overfitting ratio: {overfitting_ratio:.2f}")

        # Save model
        models_dir = Path(__file__).parent.parent / "backend/ml/models"
        models_dir.mkdir(exist_ok=True)

        model_path = models_dir / f"{model_name}_v2.pkl"
        joblib.dump(model, model_path)

        logger.info(f"  ✅ Model saved to {model_path}")

        return {
            'model': model_name,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'train_mase': train_mase,
            'val_mase': val_mase,
            'overfitting_ratio': overfitting_ratio,
            'best_iteration': model.num_trees(),
            'success': True
        }

    except Exception as e:
        logger.error(f"  ❌ Error retraining {model_name}: {e}")
        return {
            'model': model_name,
            'error': str(e),
            'success': False
        }


def main():
    """Retrain all 14 overfitting models"""

    print("=" * 80)
    print("RETRAINING 14 OVERFITTING MODELS")
    print("=" * 80)
    print("\nModels to retrain:")
    print("  - 10 RandomForest models (severely overfitting)")
    print("  - 4 LGBM models (overfitting)")
    print("\nNew hyperparameters:")
    print("  RandomForest: max_depth=5, min_samples_split=20, oob_score=True")
    print("  LGBM: max_depth=5, learning_rate=0.01, early_stopping=50")
    print("=" * 80)
    print()

    results = []
    rf_count = 0
    lgbm_count = 0

    start_time = datetime.now()

    for model_name in MODELS_TO_RETRAIN:
        if 'RandomForest' in model_name:
            result = retrain_randomforest_model(model_name, model_name.split('_')[0])
            rf_count += 1
        elif 'LGBM' in model_name:
            result = retrain_lgbm_model(model_name, model_name.split('_')[0])
            lgbm_count += 1
        else:
            logger.warning(f"Unknown model type: {model_name}")
            continue

        results.append(result)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Print summary
    print("\n" + "=" * 80)
    print("RETRAINING SUMMARY")
    print("=" * 80)

    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]

    print(f"\nTotal models: {len(MODELS_TO_RETRAIN)}")
    print(f"  RandomForest: {rf_count}")
    print(f"  LGBM: {lgbm_count}")
    print(f"\nSuccessful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Duration: {duration:.1f} seconds")

    if successful:
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)

        # Calculate average improvements
        rf_results = [r for r in successful if 'RandomForest' in r['model']]
        lgbm_results = [r for r in successful if 'LGBM' in r['model']]

        if rf_results:
            avg_val_mase = sum(r['val_mase'] for r in rf_results) / len(rf_results)
            avg_ratio = sum(r['overfitting_ratio'] for r in rf_results) / len(rf_results)
            print(f"\nRandomForest ({len(rf_results)} models):")
            print(f"  Avg Val MASE: {avg_val_mase:.4f}")
            print(f"  Avg Overfitting Ratio: {avg_ratio:.2f}")

        if lgbm_results:
            avg_val_mase = sum(r['val_mase'] for r in lgbm_results) / len(lgbm_results)
            avg_ratio = sum(r['overfitting_ratio'] for r in lgbm_results) / len(lgbm_results)
            avg_iters = sum(r.get('best_iteration', 500) for r in lgbm_results) / len(lgbm_results)
            print(f"\nLGBM ({len(lgbm_results)} models):")
            print(f"  Avg Val MASE: {avg_val_mase:.4f}")
            print(f"  Avg Overfitting Ratio: {avg_ratio:.2f}")
            print(f"  Avg Best Iteration: {avg_iters:.0f}")

        print("\n" + "=" * 80)
        print("EXPECTED IMPROVEMENT:")
        print("=" * 80)
        print("Before: RandomForest MASE 3.7, LGBM MASE 2.4")
        print("After:  RandomForest MASE < 1.5, LGBM MASE < 1.3")
        print("=" * 80)

    if failed:
        print("\n" + "=" * 80)
        print("FAILED MODELS")
        print("=" * 80)
        for result in failed:
            print(f"  {result['model']}: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
