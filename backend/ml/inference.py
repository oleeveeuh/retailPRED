"""
Inference Module
Loads trained models and generates predictions with SHAP explanations
"""

import logging
import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not installed. Install with: pip install shap")

logger = logging.getLogger(__name__)

# Model directory
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def get_model_name_for_category(
    category: str = "total_retail_sales",
    model_type: str = "xgboost"
) -> str:
    """
    Get model name for a retail category and model type

    Args:
        category: Retail category key (e.g., "total_retail_sales", "automobile_dealers")
        model_type: Model type ("xgboost" or "randomforest")

    Returns:
        Model name for use with ModelPredictor

    Example:
        >>> model_name = get_model_name_for_category("automobile_dealers", "xgboost")
        >>> print(model_name)  # "automobile_dealers_xgboost_forecaster"
    """
    return f"{category}_{model_type}_forecaster"


class ModelPredictor:
    """
    Model predictor with SHAP explanation support
    """

    def __init__(self, model_name: str = "xgboost_forecaster"):
        """
        Initialize predictor with trained model

        Args:
            model_name: Name of the model file (without .pkl extension)
        """
        self.model_name = model_name
        self.model = None
        self.explainer = None
        self.feature_names = None
        self.model_type = None

        self._load_model()

    def _load_model(self):
        """Load trained model and explainer from disk"""
        model_path = MODELS_DIR / f"{self.model_name}.pkl"

        if not model_path.exists():
            logger.warning(f"Model file not found: {model_path}")
            logger.info("Place your trained model in backend/ml/models/ directory")
            return False

        try:
            # Load model
            self.model = joblib.load(model_path)
            logger.info(f"✓ Loaded model: {self.model_name}")

            # Try to load feature names if available
            feature_names_path = MODELS_DIR / f"{self.model_name}_features.pkl"
            if feature_names_path.exists():
                self.feature_names = joblib.load(feature_names_path)
                logger.info(f"✓ Loaded feature names: {len(self.feature_names)} features")

            # Determine model type
            if hasattr(self.model, 'feature_importances_'):
                self.model_type = 'tree'
            elif hasattr(self.model, 'coef_'):
                self.model_type = 'linear'
            else:
                self.model_type = 'ensemble'

            # Initialize SHAP explainer if available
            if SHAP_AVAILABLE and self.model is not None:
                try:
                    if self.model_type == 'tree':
                        self.explainer = shap.TreeExplainer(self.model)
                    elif self.model_type == 'linear':
                        self.explainer = shap.LinearExplainer(self.model)
                    else:
                        self.explainer = shap.Explainer(self.model)
                    logger.info(f"✓ Initialized SHAP explainer ({self.model_type})")
                except Exception as e:
                    logger.warning(f"Could not initialize SHAP explainer: {e}")

            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def predict(
        self,
        features: Dict[str, Any],
        return_shap: bool = True
    ) -> Tuple[float, Optional[Dict[str, float]], Dict[str, Any]]:
        """
        Generate prediction with optional SHAP values

        Args:
            features: Dictionary of feature names to values
            return_shap: Whether to calculate SHAP values

        Returns:
            Tuple of (prediction, shap_values, metadata)
        """
        if self.model is None:
            # Return mock prediction if model not loaded
            return self._mock_prediction(features)

        try:
            # Convert features to DataFrame with correct feature order
            if self.feature_names is not None:
                # Ensure all required features are present
                feature_array = []
                for fname in self.feature_names:
                    feature_array.append(features.get(fname, 0))
                X = pd.DataFrame([feature_array], columns=self.feature_names)
            else:
                # Use feature dict as-is
                X = pd.DataFrame([features])

            # Generate prediction
            prediction = float(self.model.predict(X)[0])

            # Calculate SHAP values
            shap_values = None
            if return_shap and self.explainer is not None:
                try:
                    shap_array = self.explainer.shap_values(X)

                    # Handle different SHAP output formats
                    if isinstance(shap_array, list):
                        shap_array = shap_array[0]  # For multi-class

                    # Create feature->shap_value dict
                    shap_values = {}
                    for i, fname in enumerate(X.columns):
                        shap_values[fname] = float(shap_array[0][i])

                except Exception as e:
                    logger.warning(f"SHAP calculation failed: {e}")

            # Metadata
            metadata = {
                "model_used": self.model_name,
                "model_type": self.model_type,
                "feature_count": len(X.columns),
                "features_used": list(X.columns),
            }

            logger.info(f"✓ Prediction: {prediction:.2f}")

            return prediction, shap_values, metadata

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return self._mock_prediction(features)

    def _mock_prediction(self, features: Dict[str, Any]) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
        """
        Generate mock prediction when model not available
        """
        logger.warning("Using mock prediction - integrate actual model")

        # Simple mock based on features
        base_value = features.get("lag_1", features.get("sales_lag_1", 1000))
        promotion = features.get("promotion", features.get("promotion_flag", 0))
        holiday = features.get("holiday", features.get("is_holiday", 0))
        month = features.get("month", 1)
        moving_avg = features.get("moving_avg_7", features.get("sales_ma_7", base_value))

        # Mock formula
        prediction = base_value * (1 + 0.15 * promotion - 0.05 * holiday)
        prediction += (month / 12) * 100
        prediction += np.random.uniform(-50, 50)

        # Mock SHAP values
        shap_values = {
            "lag_1": 0.35 * base_value,
            "promotion": 0.25 * prediction if promotion else 0,
            "holiday": -0.15 * prediction if holiday else 0,
            "moving_avg_7": 0.15 * moving_avg,
            "month": 0.10 * (prediction / 12),
            "day_of_week": 0.05 * (prediction / 7),
            "inventory_level": 0.08 * features.get("inventory_level", 100),
            "competitor_price": -0.05 * features.get("competitor_price", 1.0),
        }

        metadata = {
            "model_used": self.model_name or "mock_model",
            "model_type": "mock",
            "warning": "Using mock prediction - integrate actual models in backend/ml/models/",
        }

        return prediction, shap_values, metadata


# Global predictor cache
_predictors: Dict[str, ModelPredictor] = {}


def get_predictor(model_name: str = "xgboost_forecaster") -> ModelPredictor:
    """
    Get or create predictor for model (cached)

    Args:
        model_name: Name of model

    Returns:
        ModelPredictor instance
    """
    if model_name not in _predictors:
        _predictors[model_name] = ModelPredictor(model_name)
    return _predictors[model_name]


def get_prediction(
    features: Dict[str, Any],
    model_name: Optional[str] = None,
    store_id: Optional[int] = None,
    product_id: Optional[int] = None,
) -> Tuple[float, Optional[Dict[str, float]], Dict[str, Any]]:
    """
    Generate prediction with SHAP values

    This is the main function called by the API

    Args:
        features: Dictionary of features for prediction
        model_name: Specific model to use (default: xgboost_forecaster)
        store_id: Store ID (for logging/metadata)
        product_id: Product ID (for logging/metadata)

    Returns:
        Tuple of:
        - predicted_value (float)
        - shap_values (dict of feature -> shap_value, or None)
        - metadata (dict with additional info)

    Example:
        >>> features = {
        ...     "lag_1": 1500,
        ...     "promotion": 1,
        ...     "month": 1,
        ...     "moving_avg_7": 1480
        ... }
        >>> pred, shap, meta = get_prediction(features)
        >>> print(f"Prediction: ${pred:.2f}")
        >>> print(f"Top feature: {max(shap.items(), key=lambda x: abs(x[1]))}")
    """
    logger.info(f"Generating prediction for store_id={store_id}, product_id={product_id}")

    # Get predictor
    predictor = get_predictor(model_name or "xgboost_forecaster")

    # Generate prediction
    prediction, shap_values, metadata = predictor.predict(features)

    # Add store/product info to metadata
    if store_id:
        metadata["store_id"] = store_id
    if product_id:
        metadata["product_id"] = product_id

    return prediction, shap_values, metadata


def get_available_models() -> List[str]:
    """
    List available trained models

    Returns:
        List of model names
    """
    models = []
    for path in MODELS_DIR.glob("*.pkl"):
        if not path.name.endswith("_features.pkl") and not path.name.endswith("_explainer.pkl"):
            models.append(path.stem)
    return models


def get_available_categories() -> List[str]:
    """
    List available retail categories with trained models

    Returns:
        List of category keys
    """
    from ml.data_loader import RETAIL_CATEGORIES
    return list(RETAIL_CATEGORIES.keys())


def get_category_display_name(category: str) -> str:
    """
    Get display name for a category

    Args:
        category: Category key

    Returns:
        Display name
    """
    from ml.data_loader import CATEGORY_DISPLAY_NAMES
    return CATEGORY_DISPLAY_NAMES.get(category, category)


def validate_features(features: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate features for prediction

    Args:
        features: Feature dictionary

    Returns:
        Tuple of (is_valid, missing_features_list)
    """
    # Required features (adjust based on your actual model)
    required_features = [
        "lag_1",
        "moving_avg_7",
    ]

    missing = []
    for feature in required_features:
        if feature not in features:
            # Check for alternative names
            alternatives = {
                "lag_1": ["sales_lag_1", "lag_sales", "previous_sales"],
                "moving_avg_7": ["sales_ma_7", "ma_7", "moving_average"],
            }
            found = False
            for alt in alternatives.get(feature, []):
                if alt in features:
                    found = True
                    break
            if not found:
                missing.append(feature)

    return len(missing) == 0, missing


if __name__ == "__main__":
    # Test prediction
    logging.basicConfig(level=logging.INFO)

    features = {
        "lag_1": 1500,
        "promotion": 1,
        "holiday": 0,
        "month": 1,
        "moving_avg_7": 1480,
        "day_of_week": 1,
        "inventory_level": 100,
        "competitor_price": 1.2,
    }

    print("=" * 60)
    print("Testing Inference Module")
    print("=" * 60)

    # Test validation
    is_valid, missing = validate_features(features)
    print(f"\nFeature validation: {'✓ Valid' if is_valid else '✗ Invalid'}")
    if missing:
        print(f"Missing features: {missing}")

    # Test prediction
    pred, shap, meta = get_prediction(
        features,
        model_name="xgboost_forecaster",
        store_id=1,
        product_id=101
    )

    print(f"\nPrediction: ${pred:.2f}")
    print(f"\nModel: {meta.get('model_used')}")
    print(f"Type: {meta.get('model_type')}")

    if shap:
        print(f"\nSHAP Values:")
        for feature, value in sorted(shap.items(), key=lambda x: abs(x[1]), reverse=True):
            print(f"  {feature:20s}: {value:+8.2f}")

    print(f"\nAvailable models: {get_available_models()}")
    print("=" * 60)
