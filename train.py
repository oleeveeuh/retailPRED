#!/usr/bin/env python3
"""
RetailPRED Unified Training Script

This script runs the complete training pipeline for retail sales forecasting.
It reuses existing training modules and provides a single entry point for model training.

Usage:
    python train.py                    # Train all models for all categories
    python train.py --category total   # Train specific category
    python train.py --model lgbm       # Train specific model type
    python train.py --quick            # Quick training with subset

Exit codes:
    0 - Success
    1 - Failure
"""

import sys
import os
import logging
import argparse
import json
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import centralized config
from config import (
    DATABASE_PATH,
    MODELS_DIR,
    BACKEND_MODELS_DIR,
    PROCESSED_DATA_DIR,
    MULTI_RESOLUTION_DATA_DIR,
    ensure_directories,
)

# Import existing training modules
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT / "backend" / "ml"))

try:
    from ml.data_loader import (
        RetailDataLoader,
        load_real_data_for_training,
        RETAIL_CATEGORIES,
        CATEGORY_DISPLAY_NAMES,
    )
except ImportError as e:
    print(f"Warning: Could not import data_loader: {e}")
    RETAIL_CATEGORIES = {}
    CATEGORY_DISPLAY_NAMES = {}

# Ensure directories exist
ensure_directories()
(PROJECT_ROOT / "logs").mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "logs" / "training.log")
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# Model Training Functions (reuse existing code)
# ============================================================================

def check_dependencies() -> Dict[str, bool]:
    """Check which ML libraries are available"""
    deps = {
        'lightgbm': False,
        'xgboost': False,
        'sklearn': False,
        'statsforecast': False,
        'shap': False,
    }

    try:
        import lightgbm
        deps['lightgbm'] = True
        logger.info("✓ LightGBM available")
    except ImportError:
        logger.warning("✗ LightGBM not installed")

    try:
        import xgboost
        deps['xgboost'] = True
        logger.info("✓ XGBoost available")
    except ImportError:
        logger.warning("✗ XGBoost not installed")

    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        deps['sklearn'] = True
        logger.info("✓ scikit-learn available")
    except ImportError:
        logger.warning("✗ scikit-learn not installed")

    try:
        from statsforecast import StatsForecast
        from statsforecast.models import SeasonalNaive, AutoARIMA, AutoETS
        deps['statsforecast'] = True
        logger.info("✓ statsforecast available")
    except ImportError:
        logger.warning("✗ statsforecast not installed")

    try:
        import shap
        deps['shap'] = True
        logger.info("✓ SHAP available")
    except ImportError:
        logger.warning("✗ SHAP not installed")

    return deps


def load_multi_resolution_data(category_key: str) -> Tuple[Any, List[str]]:
    """
    Load training data from multi-resolution CSV files

    Args:
        category_key: Category key (e.g., 'total_sales')

    Returns:
        Tuple of (DataFrame, feature_columns)
    """
    import pandas as pd

    # Map category keys to file names
    category_file_map = {
        'total_sales': 'retail_total_sales_multi_resolution.csv',
        'automobile_dealers': 'retail_automobile_dealers_multi_resolution.csv',
        'building_material_and_garden_equipment': 'retail_building_material_and_garden_equipment_multi_resolution.csv',
        'clothing_and_clothing_accessories_stores': 'retail_clothing_and_clothing_accessories_stores_multi_resolution.csv',
        'electronics_and_appliance_stores': 'retail_electronics_and_appliance_stores_multi_resolution.csv',
        'food_and_beverage_stores': 'retail_food_and_beverage_stores_multi_resolution.csv',
        'furniture_and_home_furnishings_stores': 'retail_furniture_and_home_furnishings_stores_multi_resolution.csv',
        'gasoline_stations': 'retail_gasoline_stations_multi_resolution.csv',
        'general_merchandise_stores': 'retail_general_merchandise_stores_multi_resolution.csv',
        'health_and_personal_care_stores': 'retail_health_and_personal_care_stores_multi_resolution.csv',
        'nonstore_retailers': 'retail_nonstore_retailers_multi_resolution.csv',
        'sporting_goods_hobby_and_musical_instrument_stores': 'retail_sporting_goods_hobby_and_musical_instrument_stores_multi_resolution.csv',
    }

    filename = category_file_map.get(category_key, category_file_map['total_sales'])
    csv_path = MULTI_RESOLUTION_DATA_DIR / filename

    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    logger.info(f"Loading data from {csv_path}")
    import pandas as pd
    df = pd.read_csv(csv_path)

    # Exclude target, date, and year (to prevent data leakage)
    exclude_cols = ['y', 'index', 'year']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    logger.info(f"Loaded {len(df)} rows with {len(feature_cols)} features")

    return df, feature_cols


def prepare_train_test_split(
    df: Any,
    test_size: int = 52
) -> Tuple[Any, Any, Any, Any]:
    """
    Split data into train/test sets (time-series aware)

    Args:
        df: DataFrame with features and target
        test_size: Number of samples for testing

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    import pandas as pd

    # Exclude columns
    exclude_cols = ['y', 'index', 'year']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols]
    y = df['y']

    # Time-series split (last test_size samples for testing)
    split_idx = len(df) - test_size

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    logger.info(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")

    return X_train, X_test, y_train, y_test


def calculate_metrics(y_true: Any, y_pred: Any) -> Dict[str, float]:
    """Calculate evaluation metrics"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # MAPE (avoiding division by zero)
    y_true_array = np.array(y_true)
    y_pred_array = np.array(y_pred)
    mask = y_true_array != 0
    mape = np.mean(np.abs((y_true_array[mask] - y_pred_array[mask]) / y_true_array[mask])) * 100 if mask.any() else 0

    # Accuracy (1 - MAPE/100)
    accuracy = max(0, 100 - mape)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "mape": float(mape),
        "accuracy": float(accuracy),
    }


def train_lgbm(
    X_train: Any,
    y_train: Any,
    X_test: Any,
    y_test: Any
) -> Tuple[Any, Dict[str, float]]:
    """Train LightGBM model"""
    import lightgbm as lgb

    logger.info("Training LightGBM...")

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    params = {
        "objective": "regression",
        "metric": "mae",
        "max_depth": 5,
        "num_leaves": 31,
        "learning_rate": 0.01,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
    )

    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)

    logger.info(f"  LightGBM MAE: ${metrics['mae']:,.2f}")
    logger.info(f"  LightGBM MAPE: {metrics['mape']:.2f}%")
    logger.info(f"  LightGBM Accuracy: {metrics['accuracy']:.2f}%")

    return model, metrics


def train_random_forest(
    X_train: Any,
    y_train: Any,
    X_test: Any,
    y_test: Any
) -> Tuple[Any, Dict[str, float]]:
    """Train Random Forest model"""
    from sklearn.ensemble import RandomForestRegressor

    logger.info("Training Random Forest...")

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features=0.7,
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)

    logger.info(f"  RandomForest MAE: ${metrics['mae']:,.2f}")
    logger.info(f"  RandomForest MAPE: {metrics['mape']:.2f}%")
    logger.info(f"  RandomForest Accuracy: {metrics['accuracy']:.2f}%")

    return model, metrics


def train_autoarima(
    train_df: Any,
    test_df: Any,
) -> Tuple[Any, Dict[str, float]]:
    """Train AutoARIMA statistical model"""
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    logger.info("Training AutoARIMA...")

    # Prepare data in StatsForecast format
    sf_train = train_df[['index', 'y']].copy()
    sf_train.columns = ['ds', 'y']
    sf_train['unique_id'] = 0

    model = AutoARIMA(season_length=52)
    fcst = StatsForecast(models=[model], freq='W', n_jobs=-1)
    fcst.fit(sf_train)

    # Predict
    h = len(test_df)
    forecast = fcst.predict(h=h)

    # Get predictions
    if 'AutoARIMA' in forecast.columns:
        predictions = forecast['AutoARIMA'].values
    else:
        # Find the prediction column
        for col in forecast.columns:
            if col not in ['unique_id', 'ds']:
                predictions = forecast[col].values
                break
        else:
            raise ValueError("Could not find prediction column")

    actuals = test_df['y'].values

    # Calculate metrics
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))

    mask = actuals != 0
    mape = np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask])) * 100 if mask.any() else 0
    accuracy = max(0, 100 - mape)

    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "accuracy": float(accuracy),
        "r2": float(0),  # Not calculated for ARIMA
    }

    logger.info(f"  AutoARIMA MAE: ${metrics['mae']:,.2f}")
    logger.info(f"  AutoARIMA MAPE: {metrics['mape']:.2f}%")
    logger.info(f"  AutoARIMA Accuracy: {metrics['accuracy']:.2f}%")

    return fcst, metrics


def train_model_ensemble(
    category_key: str = 'total_sales',
    model_types: Optional[List[str]] = None,
    test_size: int = 52,
) -> Dict[str, Any]:
    """
    Train an ensemble of models for a category

    Args:
        category_key: Category to train
        model_types: List of model types (e.g., ['lgbm', 'rf', 'arima'])
        test_size: Test set size

    Returns:
        Dictionary with training results
    """
    if model_types is None:
        model_types = ['lgbm', 'rf']

    import pandas as pd

    results = {
        'category': category_key,
        'models_trained': [],
        'models_failed': [],
        'metrics': {},
    }

    try:
        # Load data
        df, feature_cols = load_multi_resolution_data(category_key)

        # Split data
        split_idx = len(df) - test_size
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        exclude_cols = ['y', 'index', 'year']
        X_train = train_df[[c for c in df.columns if c not in exclude_cols]]
        X_test = test_df[[c for c in df.columns if c not in exclude_cols]]
        y_train = train_df['y']
        y_test = test_df['y']

        # Train each model type
        for model_type in model_types:
            try:
                if model_type.lower() in ['lgbm', 'lightgbm']:
                    model, metrics = train_lgbm(X_train, y_train, X_test, y_test)
                    model_name = f"{category_key}_LGBM_model"

                elif model_type.lower() in ['rf', 'randomforest', 'random_forest']:
                    model, metrics = train_random_forest(X_train, y_train, X_test, y_test)
                    model_name = f"{category_key}_RandomForest_model"

                elif model_type.lower() in ['arima', 'autoarima']:
                    model, metrics = train_autoarima(train_df, test_df)
                    model_name = f"{category_key}_AutoARIMA_model"

                else:
                    logger.warning(f"Unknown model type: {model_type}")
                    continue

                # Save model
                model_path = BACKEND_MODELS_DIR / f"{model_name}.pkl"
                joblib.dump(model, model_path)
                logger.info(f"  Saved model to {model_path}")

                results['models_trained'].append(model_type)
                results['metrics'][model_type] = metrics

            except Exception as e:
                logger.error(f"Failed to train {model_type}: {e}")
                results['models_failed'].append(model_type)

    except Exception as e:
        logger.error(f"Failed to train {category_key}: {e}")
        import traceback
        traceback.print_exc()
        results['error'] = str(e)

    return results


def save_metrics(metrics: Dict[str, Any], model_version: str = "latest") -> str:
    """
    Save training metrics to JSON file

    Args:
        metrics: Dictionary of metrics
        model_version: Model version identifier

    Returns:
        Path to saved metrics file
    """
    metrics_data = {
        "timestamp": datetime.now().isoformat(),
        "model_version": model_version,
        "metrics": metrics,
    }

    metrics_path = MODELS_DIR / "latest_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)

    logger.info(f"Saved metrics to {metrics_path}")
    return str(metrics_path)


def create_ensemble_wrapper(
    models: List[Any],
    model_types: List[str],
    category: str = 'total_sales'
) -> Any:
    """
    Create a simple ensemble wrapper for multiple models

    Args:
        models: List of trained models
        model_types: List of model type names
        category: Category name

    Returns:
        Ensemble wrapper object
    """
    class EnsembleModel:
        def __init__(self, models, model_types):
            self.models = models
            self.model_types = model_types

        def predict(self, X):
            predictions = []
            for model in self.models:
                try:
                    pred = model.predict(X)
                    predictions.append(pred)
                except:
                    pass
            # Average predictions
            import numpy as np
            return np.mean(predictions, axis=0)

    ensemble = EnsembleModel(models, model_types)
    return ensemble


# ============================================================================
# Main Training Pipeline
# ============================================================================

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(
        description="RetailPRED Unified Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py                    # Train all models
  python train.py --category total   # Train total_sales category
  python train.py --model lgbm       # Train only LightGBM
  python train.py --quick            # Quick training (single category)
        """
    )

    parser.add_argument(
        '--category',
        choices=['total_sales', 'automobile_dealers', 'building_material_and_garden_equipment',
                 'clothing_and_clothing_accessories_stores', 'electronics_and_appliance_stores',
                 'food_and_beverage_stores', 'furniture_and_home_furnishings_stores',
                 'gasoline_stations', 'general_merchandise_stores',
                 'health_and_personal_care_stores', 'sporting_goods_hobby_and_musical_instrument_stores'],
        default='total_sales',
        help='Category to train (default: total_sales)'
    )

    parser.add_argument(
        '--model',
        choices=['lgbm', 'rf', 'arima', 'all'],
        default='all',
        help='Model type to train (default: all)'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick training mode'
    )

    parser.add_argument(
        '--test-size',
        type=int,
        default=52,
        help='Test set size in weeks (default: 52)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output model path (default: models/model_latest.pkl)'
    )

    args = parser.parse_args()

    # Ensure logs directory exists
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Print header
    logger.info("=" * 80)
    logger.info("RetailPRED Training Pipeline")
    logger.info("=" * 80)
    logger.info(f"Category: {args.category}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Test size: {args.test_size} weeks")
    logger.info(f"Database: {DATABASE_PATH}")
    logger.info(f"Models directory: {MODELS_DIR}")
    logger.info("=" * 80)

    # Check dependencies
    deps = check_dependencies()
    if not any([deps['lightgbm'], deps['sklearn'], deps['statsforecast']]):
        logger.error("No ML libraries available. Please install lightgbm, scikit-learn, or statsforecast.")
        return 1

    # Determine model types to train
    if args.model == 'all':
        model_types = []
        if deps['lightgbm']:
            model_types.append('lgbm')
        if deps['sklearn']:
            model_types.append('rf')
        if deps['statsforecast']:
            model_types.append('arima')
    else:
        model_types = [args.model]

    if not model_types:
        logger.error("No valid model types available for training")
        return 1

    logger.info(f"Training model types: {model_types}")

    # Train models
    start_time = datetime.now()

    try:
        results = train_model_ensemble(
            category_key=args.category,
            model_types=model_types,
            test_size=args.test_size,
        )

        duration = (datetime.now() - start_time).total_seconds()

        # Print results
        logger.info("")
        logger.info("=" * 80)
        logger.info("Training Results")
        logger.info("=" * 80)
        logger.info(f"Models trained: {results['models_trained']}")
        logger.info(f"Models failed: {results['models_failed']}")
        logger.info(f"Duration: {duration:.1f} seconds")

        # Print metrics
        for model_type, metrics in results['metrics'].items():
            logger.info("")
            logger.info(f"{model_type.upper()} Metrics:")
            logger.info(f"  MAE: ${metrics['mae']:,.2f}")
            logger.info(f"  RMSE: ${metrics['rmse']:,.2f}")
            logger.info(f"  MAPE: {metrics['mape']:.2f}%")
            logger.info(f"  Accuracy: {metrics['accuracy']:.2f}%")
            logger.info(f"  R²: {metrics['r2']:.4f}")

        # Save metrics
        metrics_data = {
            'category': args.category,
            'models': results['metrics'],
            'duration_seconds': duration,
            'timestamp': datetime.now().isoformat(),
        }

        # Create model version string
        model_version = f"{args.category}_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        save_metrics(metrics_data, model_version)

        # Save symlink to latest model
        if results['models_trained']:
            # Pick the best model (lowest MAE)
            best_model_type = min(results['metrics'].items(), key=lambda x: x[1]['mae'])[0]
            best_model_name = f"{args.category}_{best_model_type.upper()}_model.pkl"

            # Copy to models/latest
            src = BACKEND_MODELS_DIR / best_model_name
            dst = MODELS_DIR / "model_latest.pkl"

            if src.exists():
                import shutil
                shutil.copy(src, dst)
                logger.info(f"")
                logger.info(f"Best model: {best_model_type}")
                logger.info(f"Latest model saved to: {dst}")

        logger.info("")
        logger.info("=" * 80)
        logger.info("Training completed successfully!")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
