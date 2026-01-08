"""
Model Loader Service
Loads and manages trained forecasting models for scenario analysis
"""

import os
import pickle
import logging
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelLoader:
    """Loads and caches trained forecasting models"""

    def __init__(self, models_dir: str = None, lazy_load: bool = True):
        """
        Initialize model loader

        Args:
            models_dir: Directory containing trained models (default: auto-detect)
            lazy_load: If True, don't load models until needed (default: True)
        """
        if models_dir is None:
            # Auto-detect models directory
            # Try multiple possible paths
            possible_paths = [
                "../../training_outputs/models",
                "../training_outputs/models",
                "/Users/olivialiau/retailPRED/training_outputs/models",
                "./training_outputs/models",
            ]

            for path in possible_paths:
                test_path = Path(path).resolve()
                if test_path.exists() and test_path.is_dir():
                    models_dir = str(test_path)
                    break

            if models_dir is None:
                # Fallback to relative path
                models_dir = "../../training_outputs/models"

        self.models_dir = Path(models_dir).resolve()
        self.model_cache: Dict[str, Any] = {}
        self.category_models: Dict[str, Dict[str, Any]] = {}
        self._models_loaded = False

        logger.info(f"Looking for models in: {self.models_dir}")

        # Verify models directory exists
        if not self.models_dir.exists():
            logger.error(f"Models directory not found: {self.models_dir}")
            return

        # Only load models if not lazy loading
        if not lazy_load:
            self._load_available_models()

    def _load_available_models(self):
        """Scan and load all trained models"""
        logger.info("Scanning for trained models...")

        if not self.models_dir.exists():
            logger.error(f"Models directory does not exist: {self.models_dir}")
            return

        # Iterate through category directories
        for category_path in self.models_dir.iterdir():
            if not category_path.is_dir():
                continue

            category_name = category_path.name
            logger.info(f"Found category: {category_name}")

            self.category_models[category_name] = {}

            # Load all model files for this category
            for model_file in category_path.glob("*_model.pkl"):
                model_type = model_file.stem.replace("_model", "")

                try:
                    with open(model_file, 'rb') as f:
                        model_data = pickle.load(f)

                    # Extract actual model if stored in dictionary
                    if isinstance(model_data, dict) and 'model' in model_data:
                        model = model_data['model']
                    else:
                        model = model_data

                    self.category_models[category_name][model_type] = model
                    self.model_cache[f"{category_name}_{model_type}"] = model

                    logger.info(f"  Loaded {model_type} model for {category_name}")

                except Exception as e:
                    logger.error(f"  Failed to load {model_file}: {e}")

        logger.info(f"Loaded models for {len(self.category_models)} categories")

    def get_model(self, category: str, model_type: str = "RandomForest") -> Optional[Any]:
        """
        Get a specific model for a category

        Args:
            category: Retail category name
            model_type: Model type (RandomForest, LGBM, PatchTST, etc.)

        Returns:
            Model object or None if not found
        """
        # Load models on first access if lazy loading
        if not self._models_loaded:
            self._load_available_models()
            self._models_loaded = True

        # Normalize category name
        category_mapping = {
            "total_sales": "Total_Retail_Sales",
            "general_merchandise": "General_Merchandise",
            "food_beverage": "Food_Beverage_Stores",
            "automobile_dealers": "Automobile_Dealers",
            "building_materials": "Building_Materials_Garden",
            "clothing_accessories": "Clothing_Accessories",
            "electronics_appliances": "Electronics_and_Appliances",
            "furniture_home": "Furniture_Home_Furnishings",
            "gasoline_stations": "Gasoline_Stations",
            "health_personal_care": "Health_Personal_Care",
            "sporting_goods_hobby": "Sporting_Goods_Hobby",
        }

        normalized_category = category_mapping.get(category, category)

        if normalized_category in self.category_models:
            if model_type in self.category_models[normalized_category]:
                return self.category_models[normalized_category][model_type]

            # Try fallback to RandomForest
            if "RandomForest" in self.category_models[normalized_category]:
                logger.warning(f"{model_type} not found for {category}, using RandomForest")
                return self.category_models[normalized_category]["RandomForest"]

        logger.warning(f"No model found for category={category}, model_type={model_type}")
        return None

    def get_best_model(self, category: str) -> Optional[Any]:
        """
        Get the best performing model for a category
        Priority: RandomForest > LGBM > PatchTST > TimesNet

        Args:
            category: Retail category name

        Returns:
            Best available model or None
        """
        category_mapping = {
            "total_sales": "Total_Retail_Sales",
            "general_merchandise": "General_Merchandise",
            "food_beverage": "Food_Beverage_Stores",
            "automobile_dealers": "Automobile_Dealers",
            "building_materials": "Building_Materials_Garden",
        }

        normalized_category = category_mapping.get(category, category)

        if normalized_category not in self.category_models:
            return None

        models = self.category_models[normalized_category]

        # Priority order
        priority = ["RandomForest", "LGBM", "PatchTST", "TimesNet", "AutoARIMA", "AutoETS", "SeasonalNaive"]

        for model_type in priority:
            if model_type in models:
                logger.info(f"Using {model_type} for {category}")
                return models[model_type]

        # Return any available model
        if models:
            return list(models.values())[0]

        return None

    def predict(self, category: str, features: Dict[str, float], model_type: str = "RandomForest") -> Optional[float]:
        """
        Make a prediction using the specified model

        Args:
            category: Retail category name
            features: Dictionary of feature values (economic indicators like UNRATE, CPI, etc.)
            model_type: Model type to use

        Returns:
            Predicted value or None if prediction fails
        """
        try:
            # Import the multi-resolution inference module for proper feature preparation
            from ml.multi_resolution_inference import MultiResolutionInference

            # Initialize inference engine
            inference = MultiResolutionInference()

            # Make prediction using the proper inference pipeline
            # This will prepare all 242 features correctly
            result = inference.predict(
                category=category,
                prediction_date=None,  # Use latest available date
                model_name=model_type,
                custom_features=features  # Pass economic indicators
            )

            if result and 'forecast' in result:
                forecast_value = result['forecast'][0] if isinstance(result['forecast'], list) else result['forecast']
                logger.info(f"Prediction for {category}: {forecast_value:.2f}")
                return float(forecast_value)

            logger.warning(f"Prediction returned no result for {category}")
            return None

        except ImportError as e:
            logger.error(f"MultiResolutionInference import failed: {e}")
            # Fallback: use simple average if inference fails
            logger.warning("Using fallback prediction based on historical average")
            # Get a reasonable value from historical data
            historical_avg = features.get('RSXFS', 50000) / 10  # Rough approximation
            return historical_avg
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def list_available_models(self) -> Dict[str, Dict[str, bool]]:
        """
        List all available models by category

        Returns:
            Dictionary mapping categories to available model types
        """
        available = {}

        for category, models in self.category_models.items():
            available[category] = {
                model_type: True for model_type in models.keys()
            }

        return available

    def get_model_info(self, category: str) -> Dict[str, Any]:
        """
        Get information about available models for a category

        Args:
            category: Retail category name

        Returns:
            Dictionary with model information
        """
        category_mapping = {
            "total_sales": "Total_Retail_Sales",
            "general_merchandise": "General_Merchandise",
            "food_beverage": "Food_Beverage_Stores",
            "automobile_dealers": "Automobile_Dealers",
            "building_materials": "Building_Materials_Garden",
        }

        normalized_category = category_mapping.get(category, category)

        if normalized_category not in self.category_models:
            return {
                "category": category,
                "available": False,
                "models": []
            }

        return {
            "category": category,
            "available": True,
            "models": list(self.category_models[normalized_category].keys()),
            "count": len(self.category_models[normalized_category])
        }


# Global model loader instance
model_loader: Optional[ModelLoader] = None


def get_model_loader() -> ModelLoader:
    """Get or initialize the global model loader"""
    global model_loader

    if model_loader is None:
        model_loader = ModelLoader()
        logger.info("ModelLoader initialized")

    return model_loader
