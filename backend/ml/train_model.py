"""
Model Training Script with SHAP Integration
Trains ML models and saves them with SHAP explainers
"""

import logging
import joblib
from typing import Dict, List, Any, Optional
from pathlib import Path
import time
import numpy as np
import pandas as pd
import sys

# Add project root to path for config import
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from config import BACKEND_MODELS_DIR
    MODELS_DIR = BACKEND_MODELS_DIR
except ImportError:
    MODELS_DIR = Path(__file__).parent / "models"

MODELS_DIR.mkdir(exist_ok=True)

# ML models
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# SHAP for explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

logger = logging.getLogger(__name__)

# Model directory
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate evaluation metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # MAPE (avoiding division by zero)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else 0

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "mape": float(mape),
    }


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    hyperparameters: Optional[Dict[str, Any]] = None,
) -> tuple:
    """
    Train XGBoost model with SHAP explainer

    Returns:
        Tuple of (model, explainer, metrics, feature_names)
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is not installed")

    # Default hyperparameters
    params = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    }
    if hyperparameters:
        params.update(hyperparameters)

    logger.info(f"Training XGBoost with params: {params}")

    # Train model
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred)
    logger.info(f"XGBoost Metrics: {metrics}")

    # Initialize SHAP explainer
    explainer = None
    if SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(model)
            logger.info("✓ SHAP TreeExplainer initialized")
        except Exception as e:
            logger.warning(f"Could not initialize SHAP explainer: {e}")

    return model, explainer, metrics, list(X_train.columns)


def train_random_forest_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    hyperparameters: Optional[Dict[str, Any]] = None,
) -> tuple:
    """
    Train Random Forest model with SHAP explainer

    Returns:
        Tuple of (model, explainer, metrics, feature_names)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is not installed")

    # Default hyperparameters (optimized to prevent overfitting)
    params = {
        "n_estimators": 200,              # More trees (but limited depth)
        "max_depth": 5,                   # SHALLOWER trees to prevent overfitting
        "min_samples_split": 20,          # More samples required to split
        "min_samples_leaf": 10,           # More samples required at leaf nodes
        "max_features": 0.7,              # Use only 70% of features per tree
        "bootstrap": True,                # Bootstrap sampling
        "oob_score": True,                # Out-of-bag scoring for validation
        "random_state": 42,
        "n_jobs": -1,
    }
    if hyperparameters:
        params.update(hyperparameters)

    logger.info(f"Training Random Forest with params: {params}")

    # Train model
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred)
    logger.info(f"Random Forest Metrics: {metrics}")

    # Initialize SHAP explainer
    explainer = None
    if SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(model)
            logger.info("✓ SHAP TreeExplainer initialized")
        except Exception as e:
            logger.warning(f"Could not initialize SHAP explainer: {e}")

    return model, explainer, metrics, list(X_train.columns)


def save_model_with_shap(
    model: Any,
    explainer: Any,
    feature_names: List[str],
    model_name: str,
    metrics: Dict[str, float],
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save trained model with SHAP explainer

    Args:
        model: Trained model
        explainer: SHAP explainer
        feature_names: List of feature names
        model_name: Name for the model
        metrics: Performance metrics
        metadata: Additional metadata

    Returns:
        Path to saved model
    """
    model_path = MODELS_DIR / f"{model_name}.pkl"
    explainer_path = MODELS_DIR / f"{model_name}_explainer.pkl"
    features_path = MODELS_DIR / f"{model_name}_features.pkl"
    metadata_path = MODELS_DIR / f"{model_name}_metadata.json"

    # Save model
    joblib.dump(model, model_path)
    logger.info(f"✓ Model saved to {model_path}")

    # Save explainer
    if explainer is not None:
        joblib.dump(explainer, explainer_path)
        logger.info(f"✓ SHAP explainer saved to {explainer_path}")

    # Save feature names
    joblib.dump(feature_names, features_path)
    logger.info(f"✓ Feature names saved to {features_path}")

    # Save metadata
    metadata_dict = {
        "model_name": model_name,
        "metrics": metrics,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if metadata:
        metadata_dict.update(metadata)

    import json
    with open(metadata_path, "w") as f:
        json.dump(metadata_dict, f, indent=2)
    logger.info(f"✓ Metadata saved to {metadata_path}")

    return str(model_path)


def train_single_model(
    model_type: str,
    model_name: str,
    force_retrain: bool = False,
    test_size: float = 0.2,
    hyperparameters: Optional[Dict[str, Any]] = None,
    use_real_data: bool = True,
    category: str = "total_retail_sales",
) -> Dict[str, Any]:
    """
    Train a single ML model with SHAP explainer

    Args:
        model_type: Type of model to train ("XGBoost" or "RandomForest")
        model_name: Name for saving the model
        force_retrain: Force retraining even if model exists
        test_size: Proportion of data for testing
        hyperparameters: Custom hyperparameters for the model
        use_real_data: If True, loads real retail data; otherwise uses synthetic data
        category: Retail category to train on

    Returns:
        Dictionary with training results including metrics
    """
    # Check if model already exists
    model_path = MODELS_DIR / f"{model_name}.pkl"

    if not force_retrain and model_path.exists():
        logger.info(f"Model {model_name} already exists, skipping...")
        # Load existing metrics
        metadata_path = MODELS_DIR / f"{model_name}_metadata.json"
        if metadata_path.exists():
            import json
            with open(metadata_path) as f:
                metadata = json.load(f)
                return {
                    "model_name": model_name,
                    "metrics": metadata["metrics"],
                    "status": "cached"
                }

    # Load training data
    if use_real_data:
        try:
            from ml.data_loader import load_real_data_for_training, get_category_display_name
            category_name = get_category_display_name(category)
            logger.info(f"Loading {category_name} data...")
            X_train, X_test, y_train, y_test, feature_names = load_real_data_for_training(
                test_size=test_size,
                category=category
            )
            logger.info(f"✓ Real data loaded: {len(X_train)} train samples, {len(X_test)} test samples")
        except Exception as e:
            logger.warning(f"Could not load real data: {e}")
            logger.warning("Falling back to synthetic data")
            use_real_data = False

    if not use_real_data:
        # Generate synthetic data as fallback
        logger.warning("Using synthetic data - integrate your actual dataset")
        X_train, X_test, y_train, y_test = train_test_split(
            *generate_synthetic_data(n_samples=1000),
            test_size=test_size,
            random_state=42
        )
        feature_names = list(X_train.columns)

    # Train model based on type
    if model_type == "XGBoost":
        model, explainer, metrics, _ = train_xgboost_model(
            X_train, y_train, X_test, y_test, hyperparameters
        )
    elif model_type == "RandomForest":
        model, explainer, metrics, _ = train_random_forest_model(
            X_train, y_train, X_test, y_test, hyperparameters
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Save model with SHAP explainer
    save_model_with_shap(
        model=model,
        explainer=explainer,
        feature_names=feature_names,
        model_name=model_name,
        metrics=metrics,
        metadata={"category": category} if category != "total_retail_sales" else None,
    )

    logger.info(f"✓ {model_type} model trained successfully")

    return {
        "model_name": model_name,
        "metrics": metrics,
        "status": "trained"
    }


def train_all_models(
    model_types: Optional[List[str]] = None,
    force_retrain: bool = False,
    test_size: float = 0.2,
    hyperparameters: Optional[Dict[str, Dict[str, Any]]] = None,
    use_real_data: bool = True,
) -> Dict[str, Any]:
    """
    Train all specified ML models with SHAP explainers

    Args:
        model_types: List of model types to train (e.g., ["XGBoost", "RandomForest"])
        force_retrain: Force retraining even if models exist
        test_size: Proportion of data for testing
        hyperparameters: Custom hyperparameters for each model type
        use_real_data: If True, loads real retail data; otherwise uses synthetic data

    Returns:
        Dictionary with training results including metrics

    Example:
        >>> result = train_all_models(
        ...     model_types=["XGBoost", "RandomForest"],
        ...     force_retrain=False,
        ...     test_size=0.2,
        ...     use_real_data=True
        ... )
    """
    logger.info(f"Starting training for models: {model_types}")

    if model_types is None:
        model_types = ["XGBoost", "RandomForest"]

    start_time = time.time()
    models_trained = []
    metrics_dict = {}

    # Train each model type
    for model_type in model_types:
        try:
            model_name = f"{model_type.lower()}_forecaster"

            result = train_single_model(
                model_type=model_type,
                model_name=model_name,
                force_retrain=force_retrain,
                test_size=test_size,
                hyperparameters=hyperparameters.get(model_type, {}) if hyperparameters else None,
                use_real_data=use_real_data,
            )

            models_trained.append(model_name)
            metrics_dict[model_name] = result["metrics"]

        except Exception as e:
            logger.error(f"Error training {model_type}: {e}")
            import traceback
            traceback.print_exc()
            continue

    training_time = time.time() - start_time

    result = {
        "status": "success",
        "models_trained": models_trained,
        "training_time_seconds": round(training_time, 2),
        "metrics": metrics_dict,
        "message": f"Successfully trained {len(models_trained)} model(s)",
    }

    logger.info(f"Training completed in {training_time:.2f}s")
    return result


def generate_synthetic_data(n_samples: int = 1000) -> tuple:
    """
    Generate synthetic training data for demonstration
    Replace this with your actual data loading logic

    Returns:
        Tuple of (X, y) where X is DataFrame of features, y is Series of targets
    """
    np.random.seed(42)

    # Generate features
    data = {
        "lag_1": np.random.uniform(1000, 2000, n_samples),
        "lag_2": np.random.uniform(900, 1900, n_samples),
        "moving_avg_7": np.random.uniform(950, 1950, n_samples),
        "moving_avg_30": np.random.uniform(900, 1900, n_samples),
        "month": np.random.randint(1, 13, n_samples),
        "day_of_week": np.random.randint(0, 7, n_samples),
        "promotion_flag": np.random.randint(0, 2, n_samples),
        "is_holiday": np.random.randint(0, 2, n_samples),
        "inventory_level": np.random.uniform(50, 200, n_samples),
        "competitor_price": np.random.uniform(0.8, 1.5, n_samples),
    }

    X = pd.DataFrame(data)

    # Generate target with some relationship to features
    y = (
        500 +
        0.5 * X["lag_1"] +
        0.3 * X["moving_avg_7"] +
        200 * X["promotion_flag"] -
        100 * X["is_holiday"] +
        0.1 * X["inventory_level"] * X["competitor_price"] +
        np.random.normal(0, 50, n_samples)
    )

    return X, y


if __name__ == "__main__":
    # Test training
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    print("=" * 60)
    print("Testing Model Training with SHAP")
    print("=" * 60)

    result = train_all_models(
        model_types=["XGBoost", "RandomForest"],
        force_retrain=True,
        test_size=0.2
    )

    print("\nTraining Results:")
    print(f"Status: {result['status']}")
    print(f"Models trained: {result['models_trained']}")
    print(f"Training time: {result['training_time_seconds']}s")
    print(f"\nMetrics:")
    for model, metrics in result['metrics'].items():
        print(f"\n{model}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    print("\n" + "=" * 60)
    print("Models and SHAP explainers saved to:", MODELS_DIR)
    print("=" * 60)
