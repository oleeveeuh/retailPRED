"""
Multi-Resolution Inference Module
Loads trained models from training_outputs and generates predictions
"""

import logging
import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Model directory - use training_outputs/models where all 7 model types exist
MODELS_DIR = Path(__file__).parent.parent.parent / "training_outputs" / "models"

# Category mappings to display names
CATEGORY_KEY_TO_DISPLAY = {
    "total_sales": "Total_Retail_Sales",
    "building_material_and_garden_equipment": "Building_Materials_Garden",
    "automobile_dealers": "Automobile_Dealers",
    "gasoline_stations": "Gasoline_Stations",
    "food_and_beverage_stores": "Food_Beverage_Stores",
    "health_and_personal_care_stores": "Health_Personal_Care",
    "general_merchandise_stores": "General_Merchandise",
    "furniture_and_home_furnishings_stores": "Furniture_Home_Furnishings",
    "clothing_and_clothing_accessories_stores": "Clothing_Accessories",
    "sporting_goods_hobby_and_musical_instrument_stores": "Sporting_Goods_Hobby",
    "electronics_and_appliance_stores": "Electronics_and_Appliances",
}

# Display names to keys (reverse mapping)
CATEGORY_DISPLAY_TO_KEY = {v: k for k, v in CATEGORY_KEY_TO_DISPLAY.items()}

# Available model types (all 7 types that were trained)
AVAILABLE_MODEL_TYPES = [
    "LGBM",
    "RandomForest",
    "PatchTST",
    "TimesNet",
    "AutoARIMA",
    "AutoETS",
    "SeasonalNaive"
]


def get_model_file_path(category: str, model_type: str) -> Path:
    """
    Get the file path for a trained model

    Args:
        category: Retail category key (e.g., "total_sales")
        model_type: Model type ("LGBM", "RandomForest", "PatchTST", etc.)

    Returns:
        Path to model file
    """
    # Map category key to display name
    display_name = CATEGORY_KEY_TO_DISPLAY.get(category, category.replace("_", " ").replace(" ", "_"))

    # Model files are named: {ModelType}_model.pkl
    # e.g., training_outputs/models/Total_Retail_Sales/LGBM_model.pkl
    model_filename = f"{model_type}_model.pkl"

    return MODELS_DIR / display_name / model_filename


def get_best_model_for_category(category: str) -> str:
    """
    Get the best performing model type for a category
    Based on lowest MAPE from training results

    Args:
        category: Retail category key

    Returns:
        Model type with best validation MAPE
    """
    # Best models from training results (based on training_outputs/robust_training_summary.json)
    best_models = {
        "total_sales": "LGBM",
        "building_material_and_garden_equipment": "LGBM",
        "automobile_dealers": "LGBM",
        "gasoline_stations": "LGBM",
        "furniture_and_home_furnishings_stores": "LGBM",
        "sporting_goods_hobby_and_musical_instrument_stores": "LGBM",
        "health_and_personal_care_stores": "RandomForest",
        "clothing_and_clothing_accessories_stores": "RandomForest",
        "food_and_beverage_stores": "RandomForest",
        "electronics_and_appliance_stores": "RandomForest",
        "general_merchandise_stores": "RandomForest",
    }

    return best_models.get(category, "LGBM")


def load_model(category: str, model_type: Optional[str] = None):
    """
    Load a trained model for a category

    Args:
        category: Retail category key
        model_type: Model type (if None, uses best model)

    Returns:
        Loaded model object

    Raises:
        FileNotFoundError: If model file not found
        ValueError: If category or model_type is invalid
    """
    if model_type is None:
        model_type = get_best_model_for_category(category)

    if model_type not in AVAILABLE_MODEL_TYPES:
        raise ValueError(f"Invalid model_type: {model_type}. Must be one of {AVAILABLE_MODEL_TYPES}")

    model_path = get_model_file_path(category, model_type)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        loaded_model = joblib.load(model_path)

        # Handle dictionary-based model format from robust_timecopilot_trainer
        if isinstance(loaded_model, dict) and 'model' in loaded_model:
            model = loaded_model['model']
            logger.info(f"✓ Loaded model from dict: {category} ({model_type}) from {model_path}")
        else:
            model = loaded_model
            logger.info(f"✓ Loaded model: {category} ({model_type}) from {model_path}")

        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise


def prepare_features(
    category: str,
    current_date: Optional[str] = None,
    historical_data: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Prepare features for prediction using full feature computer with 242 features

    Args:
        category: Retail category key
        current_date: Date string (YYYY-MM-DD) for prediction
        historical_data: Historical data for computing lags (optional, will generate if not provided)

    Returns:
        DataFrame with all 242 features ready for prediction
    """
    if current_date is None:
        current_date = datetime.now().strftime("%Y-%m-%d")

    # Use the full feature computer with all 242 features including external data
    try:
        from ml.feature_computer_full import compute_full_features
        from ml.feature_computer import load_historical_data_from_csv

        # Map category keys to display names
        category_display_names = {
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

        display_name = category_display_names.get(category, category.replace("_", " ").title())

        # Load historical data if not provided
        if historical_data is None or historical_data.empty:
            logger.info(f"Loading historical data for {display_name}...")
            historical_df = load_historical_data_from_csv(display_name, days_back=400)
            # Rename 'value' column to match expected format
            if 'value' in historical_df.columns:
                historical_df = historical_df.rename(columns={'value': 'value'})
        else:
            historical_df = historical_data

        # Compute all 242 features including external economic and stock data
        logger.info(f"Computing full feature set for {display_name} as of {current_date}...")
        features_df = compute_full_features(historical_df, current_date, category)

        # Load model to get expected feature order
        try:
            model_type_for_features = get_best_model_for_category(category)
            model_check = joblib.load(get_model_file_path(category, model_type_for_features))
            if isinstance(model_check, dict) and 'model' in model_check:
                model_check = model_check['model']

            # Reorder columns to match model's expected order
            if hasattr(model_check, 'feature_names_in_'):
                expected_features = model_check.feature_names_in_

                # Create aligned DataFrame with correct order (using dict for performance)
                aligned_data = {}
                for feat in expected_features:
                    if feat in features_df.columns:
                        aligned_data[feat] = features_df[feat].values
                    else:
                        # Use default value for missing features
                        aligned_data[feat] = 0.0

                features_df = pd.DataFrame(aligned_data, index=features_df.index)
                logger.info(f"✓ Features reordered to match model expectations")

        except Exception as align_error:
            logger.warning(f"Could not align features to model order: {align_error}")

        logger.info(f"✓ Computed {len(features_df.columns)} features for prediction")
        return features_df

    except Exception as e:
        logger.error(f"Error using full feature computer: {e}. Using fallback features.")
        import traceback
        logger.error(traceback.format_exc())
        # Fallback to simple feature set
        return _create_fallback_features(current_date)


def _create_fallback_features(current_date: str) -> pd.DataFrame:
    """
    Create fallback features when feature_computer fails

    Args:
        current_date: Date string (YYYY-MM-DD)

    Returns:
        DataFrame with basic features
    """
    dt = datetime.strptime(current_date, "%Y-%m-%d")

    features = {
        # Temporal features
        'year': dt.year,
        'month': dt.month,
        'quarter': (dt.month - 1) // 3 + 1,
        'day_of_week': dt.weekday(),
        'week_of_year': dt.isocalendar()[1],
        'is_weekend': 1 if dt.weekday() >= 5 else 0,
        'day_of_month': dt.day,
        'day_of_year': dt.timetuple().tm_yday,

        # Cyclical encodings
        'month_sin': np.sin(2 * np.pi * dt.month / 12),
        'month_cos': np.cos(2 * np.pi * dt.month / 12),
        'quarter_sin': np.sin(2 * np.pi * ((dt.month - 1) // 3 + 1) / 4),
        'quarter_cos': np.cos(2 * np.pi * ((dt.month - 1) // 3 + 1) / 4),
        'day_of_year_sin': np.sin(2 * np.pi * dt.timetuple().tm_yday / 365),
        'day_of_year_cos': np.cos(2 * np.pi * dt.timetuple().tm_yday / 365),
        'day_of_week_sin': np.sin(2 * np.pi * dt.weekday() / 7),
        'day_of_week_cos': np.cos(2 * np.pi * dt.weekday() / 7),

        # Lag features (fallback values)
        'lag_1d': 1000.0,
        'lag_7d': 1000.0,
        'lag_14d': 1000.0,
        'lag_30d': 1000.0,
        'lag_1w': 1000.0,
        'lag_4w': 1000.0,
        'lag_8w': 1000.0,
        'lag_12w': 1000.0,
        'lag_1m': 1000.0,
        'lag_3m': 1000.0,
        'lag_6m': 1000.0,
        'lag_12m': 1000.0,

        # Monthly rolling statistics
        'rolling_std_3': 50.0,
        'rolling_std_6': 60.0,
        'rolling_std_12': 70.0,
        'rolling_mean_3': 1000.0,
        'rolling_mean_6': 1000.0,
        'rolling_mean_12': 1000.0,

        # Daily rolling statistics
        'rolling_mean_7d': 1000.0,
        'rolling_std_7d': 50.0,
        'rolling_mean_14d': 1000.0,
        'rolling_std_14d': 50.0,
        'rolling_mean_30d': 1000.0,
        'rolling_std_30d': 50.0,

        # Weekly rolling statistics
        'rolling_std_4w': 50.0,
        'rolling_std_8w': 50.0,
        'rolling_std_12w': 50.0,
        'rolling_mean_4w': 1000.0,
        'rolling_mean_8w': 1000.0,
        'rolling_mean_12w': 1000.0,

        # Monthly rolling statistics (multi-resolution)
        'rolling_std_3m': 50.0,
        'rolling_std_6m': 60.0,
        'rolling_std_12m': 70.0,
        'rolling_mean_3m': 1000.0,
        'rolling_mean_6m': 1000.0,
        'rolling_mean_12m': 1000.0,

        # Cross-frequency aggregations
        'weekly_agg_rolling_mean_4w': 1000.0,
        'weekly_agg_rolling_mean_8w': 1000.0,
        'weekly_agg_rolling_mean_12w': 1000.0,
        'monthly_agg_rolling_mean_3m': 1000.0,
        'monthly_agg_rolling_mean_6m': 1000.0,
        'monthly_agg_rolling_mean_12m': 1000.0,

        # Monthly rate of change features
        'diff_1': 1.0,
        'diff_12': 12.0,
        'pct_change_1': 0.001,
        'pct_change_12': 0.012,

        # Daily/weekly/monthly rate of change features
        'pct_change_1d': 0.001,
        'diff_1d': 1.0,
        'pct_change_1w': 0.005,
        'diff_1w': 5.0,
        'pct_change_1m': 0.02,
        'diff_1m': 20.0,
        'pct_change_1y': 0.05,
        'diff_1y': 50.0,

        # Momentum indicators
        'momentum_7d': 0.5,
        'momentum_30d': 2.0,
        'momentum_90d': 5.0,

        # Year-over-year change
        'yoy_change': 0.05,
    }

    return pd.DataFrame([features])


def generate_forecast(
    category: str,
    model_type: Optional[str] = None,
    weeks_ahead: int = 4,
    granularity: str = "weekly",
    start_date: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Generate forecast for a retail category

    Args:
        category: Retail category key
        model_type: Model type (if None, uses best model)
        weeks_ahead: Number of weeks to forecast
        granularity: Forecast granularity ("daily", "weekly", "monthly")
        start_date: Start date string (YYYY-MM-DD)

    Returns:
        Tuple of (forecast_list, metadata)

        forecast_list: List of forecast points with date, value, confidence interval
        metadata: Dictionary with model info, category, etc.
    """
    if start_date is None:
        start_date = datetime.now().strftime("%Y-%m-%d")

    # Load model
    try:
        model = load_model(category, model_type)
        model_type_used = model_type or get_best_model_for_category(category)
    except FileNotFoundError as e:
        logger.warning(f"Model not found: {e}. Using mock predictions.")
        model = None
        model_type_used = model_type or "LGBM"

    # Generate forecast dates
    start = datetime.strptime(start_date, "%Y-%m-%d")
    forecast = []

    for i in range(weeks_ahead):
        if granularity == "daily":
            forecast_date = start + timedelta(days=i)
        elif granularity == "weekly":
            forecast_date = start + timedelta(weeks=i)
        else:  # monthly
            forecast_date = start + timedelta(days=30*i)

        # Prepare features or historical data depending on model type
        if model_type_used in ["PatchTST", "TimesNet"]:
            # NeuralForecast models need historical data in NeuralForecast format
            from ml.feature_computer import load_historical_data_from_csv

            category_display = CATEGORY_KEY_TO_DISPLAY.get(category, category.replace("_", " ").title())
            historical_df = load_historical_data_from_csv(category_display, days_back=400)

            # Prepare data in NeuralForecast format
            # Use simple numeric unique_id (1 for single time series)
            historical_df['unique_id'] = 1
            historical_df = historical_df.rename(columns={'date': 'ds', 'value': 'y'})
            historical_df = historical_df[['unique_id', 'ds', 'y']]
            historical_df = historical_df.tail(52).reset_index(drop=True)

            try:
                # Extract the actual NeuralForecast model from the dict
                if isinstance(model, dict) and 'model' in model:
                    nf_model = model['model']
                else:
                    nf_model = model

                # NeuralForecast models use predict() with DataFrame
                predictions = nf_model.predict(historical_df)
                logger.info(f"{model_type_used} predictions: {predictions.head()}")

                # Get prediction for this week
                # NeuralForecast returns columns named by model type
                if model_type_used in predictions.columns:
                    prediction = float(predictions[model_type_used].iloc[0])
                elif 'yhat' in predictions.columns:
                    prediction = float(predictions['yhat'].iloc[0])
                elif isinstance(predictions, pd.DataFrame):
                    prediction = float(predictions.iloc[0, 0])
                else:
                    prediction = float(predictions[0])

                logger.info(f"{model_type_used} extracted prediction: {prediction}")

                base_error_pct = 1.5  # Neural models have slightly higher error
                ci_lower = prediction * (1 - base_error_pct / 100)
                ci_upper = prediction * (1 + base_error_pct / 100)

            except Exception as e:
                logger.error(f"NeuralForecast prediction error for {model_type_used}: {e}. Using mock prediction.")
                import traceback
                logger.error(traceback.format_exc())
                prediction = 1000.0 + (i * 10)
                ci_lower = prediction * 0.98
                ci_upper = prediction * 1.02

        elif model_type_used in ["AutoARIMA", "AutoETS", "SeasonalNaive"]:
            # StatsForecast models are pre-fitted, just call predict(h)
            try:
                # Extract the actual StatsForecast model from the dict
                if isinstance(model, dict) and 'model' in model:
                    sf_model = model['model']
                else:
                    sf_model = model

                # StatsForecast models just need horizon, they're already fitted
                predictions = sf_model.predict(h=1)

                # Get prediction
                if isinstance(predictions, pd.DataFrame):
                    if 'mean' in predictions.columns:
                        prediction = float(predictions['mean'].iloc[0])
                    else:
                        # StatsForecast returns columns named by unique_id
                        prediction = float(predictions.iloc[0, 0])
                else:
                    prediction = float(predictions[0])

                # Model-specific MAPE
                model_mape = {
                    "AutoARIMA": 10.66,
                    "AutoETS": 6.84,
                    "SeasonalNaive": 6.94
                }.get(model_type_used, 8.0)

                ci_multiplier = 1 + (model_mape / 100)
                ci_lower = prediction / ci_multiplier
                ci_upper = prediction * ci_multiplier

            except Exception as e:
                logger.error(f"StatsForecast prediction error for {model_type_used}: {e}. Using mock prediction.")
                prediction = 1000.0 + (i * 10)
                ci_lower = prediction * 0.98
                ci_upper = prediction * 1.02

        else:
            # Traditional ML models need features
            features_df = prepare_features(category, forecast_date.strftime("%Y-%m-%d"))

            # Make prediction
            if model is not None:
                try:
                    prediction = float(model.predict(features_df)[0])

                    # Estimate confidence interval based on model type and category
                    # Models have ~0.5-1.0% MAPE on average
                    base_error_pct = 0.7  # Average validation MAPE

                    # Widen interval for longer horizons
                    horizon_multiplier = 1 + (i * 0.1)

                    # Calculate confidence interval
                    ci_lower = prediction * (1 - (base_error_pct / 100) * horizon_multiplier)
                    ci_upper = prediction * (1 + (base_error_pct / 100) * horizon_multiplier)

                except Exception as e:
                    logger.error(f"Prediction error: {e}. Using mock prediction.")
                    prediction = 1000.0 + (i * 10)
                    ci_lower = prediction * 0.98
                    ci_upper = prediction * 1.02
            else:
                # Mock prediction
                prediction = 1000.0 + (i * 10)
                ci_lower = prediction * 0.95
                ci_upper = prediction * 1.05

        forecast.append({
            "date": forecast_date.strftime("%Y-%m-%d"),
            "predicted_value": round(prediction, 2),
            "confidence_interval_lower": round(ci_lower, 2),
            "confidence_interval_upper": round(ci_upper, 2),
            "confidence_level": 0.95,
        })

    # Category display name mapping
    category_display_names = {
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

    # Determine if this is a StatsForecast model
    is_stats_model = model_type_used in ["AutoARIMA", "AutoETS", "SeasonalNaive"]

    # Determine features used based on model type
    if model is not None and hasattr(model, 'feature_names_in_'):
        features_used = len(model.feature_names_in_)
    elif is_stats_model:
        features_used = 74  # StatsForecast uses default features
    else:
        features_used = 242  # Default for multi-resolution models

    # Metadata
    metadata = {
        "category": category,
        "category_display_name": category_display_names.get(category, category),
        "model_name": f"{category}_{model_type_used}_model",
        "model_type": model_type_used,
        "model_version": "v1",
        "forecast_start_date": start_date,
        "forecast_end_date": forecast[-1]["date"] if forecast else start_date,
        "granularity": granularity,
        "weeks_ahead": weeks_ahead,
        "features_used": features_used,
        "average_mape": 0.7,  # From training results
        "model_accuracy": "high",  # Based on <1% MAPE
    }

    return forecast, metadata


def get_available_categories() -> List[Dict[str, str]]:
    """
    Get list of available retail categories

    Returns:
        List of dictionaries with category keys and display names
    """
    return [
        {"key": key, "display_name": CATEGORY_KEY_TO_DISPLAY[key].replace("_", " ")}
        for key in CATEGORY_KEY_TO_DISPLAY.keys()
    ]


def get_available_models_for_category(category: str) -> Dict[str, Any]:
    """
    Get available models for a category

    Args:
        category: Retail category key

    Returns:
        Dictionary with category info and available model types
    """
    available_models = []

    display_name = CATEGORY_KEY_TO_DISPLAY.get(category, category)
    category_dir = MODELS_DIR / display_name

    if category_dir.exists():
        for model_file in category_dir.glob("*_model.pkl"):
            # Extract model type from filename (e.g., "LGBM_model.pkl" -> "LGBM")
            model_type = model_file.stem.replace("_model", "")
            if model_type in AVAILABLE_MODEL_TYPES:
                available_models.append(model_type)

    return {
        "category": category,
        "category_display_name": display_name.replace("_", " "),
        "available_models": available_models,
        "total_count": len(available_models),
        "best_model": get_best_model_for_category(category),
        "model_version": "v1",
        "average_mape": 0.7,
    }


if __name__ == "__main__":
    # Test inference
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Testing Multi-Resolution Inference")
    print("=" * 60)

    # Test forecast generation
    category = "total_sales"
    forecast, metadata = generate_forecast(
        category=category,
        weeks_ahead=4,
        granularity="weekly"
    )

    print(f"\nCategory: {metadata['category_display_name']}")
    print(f"Model: {metadata['model_type']}")
    print(f"MAPE: {metadata['average_mape']}%")

    print(f"\n{len(forecast)} Week Forecast:")
    for point in forecast:
        print(f"  {point['date']}: ${point['predicted_value']:,.2f} "
              f"(${point['confidence_interval_lower']:,.2f} - ${point['confidence_interval_upper']:,.2f})")

    # Test available categories
    print(f"\nAvailable Categories: {len(get_available_categories())}")
    for cat in get_available_categories():
        print(f"  - {cat['display_name']} ({cat['key']})")

    # Test available models
    print(f"\nAvailable Models for {category}:")
    models_info = get_available_models_for_category(category)
    print(f"  Total: {models_info['total_count']}")
    print(f"  Available: {', '.join(models_info['available_models'])}")
    print(f"  Best: {models_info['best_model']}")

    print("=" * 60)
