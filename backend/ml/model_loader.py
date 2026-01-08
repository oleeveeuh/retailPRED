"""
ML Integration Layer
Wraps existing ML scripts and provides a clean interface for the API
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLModelIntegrator:
    """
    Integration layer for ML models
    Wraps existing train_model.py, inference.py, and data_fetcher.py
    """

    def __init__(self, ml_base_path: str = None):
        """
        Initialize the ML integrator

        Args:
            ml_base_path: Base path to ML scripts (default: backend/ml/)
        """
        if ml_base_path is None:
            ml_base_path = Path(__file__).parent

        self.ml_path = Path(ml_base_path)
        self.models_cache = {}

    def _import_script(self, script_name: str):
        """
        Dynamically import a script from the ml directory

        Args:
            script_name: Name of the script to import

        Returns:
            Imported module
        """
        script_path = self.ml_path / script_name
        if not script_path.exists():
            logger.warning(f"Script not found: {script_path}")
            return None

        # Add ml directory to Python path
        if str(self.ml_path) not in sys.path:
            sys.path.insert(0, str(self.ml_path))

        try:
            module = __import__(script_name.replace(".py", ""))
            return module
        except ImportError as e:
            logger.error(f"Failed to import {script_name}: {e}")
            return None

    def get_prediction(
        self,
        features_dict: Dict[str, Any],
        model_name: Optional[str] = None,
        store_id: Optional[int] = None,
        product_id: Optional[int] = None,
    ) -> Tuple[float, Optional[Dict[str, float]], Dict[str, Any]]:
        """
        Get prediction from the inference module

        Args:
            features_dict: Dictionary of features for prediction
            model_name: Specific model to use (optional)
            store_id: Store ID (optional)
            product_id: Product ID (optional)

        Returns:
            Tuple of (predicted_value, shap_values_dict, metadata)
        """
        try:
            # Try to import and use the actual inference module
            inference_module = self._import_script("inference.py")

            if inference_module and hasattr(inference_module, "get_prediction"):
                # Use the actual inference function
                result = inference_module.get_prediction(
                    features=features_dict,
                    model_name=model_name,
                    store_id=store_id,
                    product_id=product_id,
                )
                return result

            else:
                # Fallback: Mock implementation for testing
                logger.warning("Using mock prediction (inference.py not found)")
                return self._mock_prediction(features_dict)

        except Exception as e:
            logger.error(f"Error getting prediction: {e}")
            raise

    def _mock_prediction(
        self, features_dict: Dict[str, Any]
    ) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
        """
        Mock prediction for testing when actual models aren't available

        Args:
            features_dict: Input features

        Returns:
            Tuple of (prediction, shap_values, metadata)
        """
        import random

        # Generate a plausible prediction based on features
        base_value = features_dict.get("lag_1", 1000)
        promotion = features_dict.get("promotion", 0)
        holiday = features_dict.get("holiday", 0)

        # Simple mock prediction logic
        prediction = base_value * (1 + 0.15 * promotion - 0.05 * holiday)
        prediction += random.uniform(-50, 50)  # Add some randomness

        # Mock SHAP values
        shap_values = {
            "lag_1": 0.35 * base_value,
            "promotion": 0.25 * prediction if promotion else 0,
            "holiday": -0.15 * prediction if holiday else 0,
            "moving_avg_7": 0.15 * base_value,
            "month": 0.10 * prediction / 12,
        }

        metadata = {
            "model_used": "mock_xgboost",
            "model_version": "1.0",
            "prediction_method": "mock",
            "warning": "Using mock prediction - connect actual models",
        }

        return prediction, shap_values, metadata

    def train_all_models(
        self,
        model_types: Optional[List[str]] = None,
        force_retrain: bool = False,
        test_size: float = 0.2,
        hyperparameters: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Train all models using the training module

        Args:
            model_types: List of model types to train
            force_retrain: Force retraining even if models exist
            test_size: Proportion of data for testing
            hyperparameters: Custom hyperparameters for each model

        Returns:
            Dictionary with training results
        """
        try:
            train_module = self._import_script("train_model.py")

            if train_module and hasattr(train_module, "train_all_models"):
                # Use the actual training function
                result = train_module.train_all_models(
                    model_types=model_types,
                    force_retrain=force_retrain,
                    test_size=test_size,
                    hyperparameters=hyperparameters,
                )
                return result

            else:
                # Fallback: Mock training
                logger.warning("Using mock training (train_model.py not found)")
                return self._mock_training(model_types or ["XGBoost", "RandomForest"])

        except Exception as e:
            logger.error(f"Error training models: {e}")
            raise

    def _mock_training(self, model_types: List[str]) -> Dict[str, Any]:
        """
        Mock training for testing

        Args:
            model_types: List of model types

        Returns:
            Mock training results
        """
        import time
        import random

        start_time = time.time()
        models_trained = []

        metrics = {}
        for model_type in model_types:
            models_trained.append(f"{model_type.lower()}_v1")
            metrics[f"{model_type.lower()}_v1"] = {
                "rmse": round(random.uniform(0.5, 1.0), 2),
                "mae": round(random.uniform(0.3, 0.7), 2),
                "r2": round(random.uniform(0.85, 0.95), 2),
                "mape": round(random.uniform(3.0, 8.0), 2),
                "training_samples": 10593,
            }

        training_time = time.time() - start_time

        return {
            "status": "success",
            "models_trained": models_trained,
            "training_time_seconds": round(training_time, 2),
            "metrics": metrics,
            "message": f"Trained {len(models_trained)} models (mock mode)",
        }

    def fetch_latest_data(self) -> Dict[str, Any]:
        """
        Fetch latest data using the data fetcher module

        Returns:
            Dictionary with fetch results
        """
        try:
            fetcher_module = self._import_script("data_fetcher.py")

            if fetcher_module and hasattr(fetcher_module, "fetch_latest_data"):
                # Use the actual data fetcher
                result = fetcher_module.fetch_latest_data()
                return result

            else:
                # Fallback: Mock data fetch
                logger.warning("Using mock data fetch (data_fetcher.py not found)")
                return self._mock_data_fetch()

        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise

    def _mock_data_fetch(self) -> Dict[str, Any]:
        """
        Mock data fetch for testing

        Returns:
            Mock fetch results
        """
        import random

        return {
            "status": "success",
            "message": "Data refreshed successfully (mock mode)",
            "records_updated": random.randint(100, 2000),
            "new_categories": random.randint(0, 5),
            "last_fetch_time": datetime.now().isoformat(),
            "sources_updated": ["FRED", "MRTS", "MockSource"],
        }

    def load_model(self, model_name: str):
        """
        Load a trained model from disk

        Args:
            model_name: Name of the model to load

        Returns:
            Loaded model object
        """
        if model_name in self.models_cache:
            return self.models_cache[model_name]

        # Try to load from disk
        model_path = self.ml_path / "models" / f"{model_name}.pkl"

        if model_path.exists():
            import joblib

            model = joblib.load(model_path)
            self.models_cache[model_name] = model
            return model
        else:
            logger.warning(f"Model file not found: {model_path}")
            return None

    def get_available_models(self) -> List[str]:
        """
        Get list of available trained models

        Returns:
            List of model names
        """
        models_dir = self.ml_path / "models"

        if not models_dir.exists():
            return []

        model_files = list(models_dir.glob("*.pkl"))
        return [f.stem for f in model_files]


# Singleton instance for easy access
_ml_integrator: Optional[MLModelIntegrator] = None


def get_ml_integrator() -> MLModelIntegrator:
    """
    Get or create the ML integrator singleton

    Returns:
        MLModelIntegrator instance
    """
    global _ml_integrator
    if _ml_integrator is None:
        _ml_integrator = MLModelIntegrator()
    return _ml_integrator
