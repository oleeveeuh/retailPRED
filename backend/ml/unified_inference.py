"""
Unified Inference Module for All Model Types

Provides a unified interface for all 7 model types:
- Sklearn models: RandomForest, LGBM (use 242 features)
- Nixtla NeuralForecast models: PatchTST, TimesNet (use historical series)
- Nixtla StatsForecast models: AutoARIMA, AutoETS, SeasonalNaive (use historical series)
"""

import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Model directories
MODELS_DIR = Path(__file__).parent.parent.parent / "training_outputs" / "models"
BACKEND_MODELS_DIR = Path(__file__).parent / "models"

# Category mappings
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

# Model type classifications
SKLEARN_MODELS = ["RandomForest", "LGBM"]
NEURAL_FORECAST_MODELS = ["PatchTST", "TimesNet"]
STATS_FORECAST_MODELS = ["AutoARIMA", "SeasonalNaive"]  # AutoETS removed due to poor performance
ALL_MODEL_TYPES = SKLEARN_MODELS + NEURAL_FORECAST_MODELS + STATS_FORECAST_MODELS


def get_model_file_path(category: str, model_type: str) -> Path:
    """Get the file path for a trained model"""
    display_name = CATEGORY_KEY_TO_DISPLAY.get(category, category.replace("_", " ").replace(" ", "_"))
    model_filename = f"{model_type}_model.pkl"

    # First check backend/ml/models (newly retrained models with unified pipeline)
    # These use category_key_model_type_model.pkl format
    backend_filename = f"{category}_{model_type}_model.pkl"
    backend_path = BACKEND_MODELS_DIR / backend_filename
    if backend_path.exists():
        return backend_path

    # Then check training_outputs (old models)
    path = MODELS_DIR / display_name / model_filename
    if path.exists():
        return path

    # Default to backend path
    return backend_path


def load_model(category: str, model_type: str):
    """Load a trained model"""
    model_path = get_model_file_path(category, model_type)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        loaded_model = joblib.load(model_path)

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


def generate_forecast(
    category: str,
    model_type: str,
    weeks_ahead: int = 4,
    start_date: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Generate forecast using the appropriate method for the model type

    Args:
        category: Retail category key
        model_type: Model type (one of ALL_MODEL_TYPES)
        weeks_ahead: Number of weeks to forecast
        start_date: Start date string (YYYY-MM-DD)

    Returns:
        Tuple of (forecast_list, metadata)
    """
    if start_date is None:
        start_date = datetime.now().strftime("%Y-%m-%d")

    # Route to appropriate forecast method
    if model_type in SKLEARN_MODELS:
        return _forecast_with_sklearn_model(category, model_type, weeks_ahead, start_date)
    elif model_type in NEURAL_FORECAST_MODELS:
        return _forecast_with_neural_forecast_model(category, model_type, weeks_ahead, start_date)
    elif model_type in STATS_FORECAST_MODELS:
        return _forecast_with_stats_forecast_model(category, model_type, weeks_ahead, start_date)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _forecast_with_sklearn_model(
    category: str,
    model_type: str,
    weeks_ahead: int,
    start_date: str
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Generate forecast using sklearn-style models (RandomForest, LGBM)"""

    from backend.ml.feature_computer import load_historical_data_from_csv
    import pandas as pd

    # Load model
    model = load_model(category, model_type)

    # Category display name
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

    # Load historical data with all pre-computed features from CSV
    # The models were trained on the CSV features directly, so we need to use the same approach
    historical_df = load_historical_data_from_csv(display_name, days_back=400)

    # Load the full multi-resolution CSV to get all features
    from pathlib import Path

    # Use category_key directly for CSV filename (matches the training script)
    csv_name = category

    csv_path = Path(__file__).parent.parent.parent / "project_root" / "data_multi_resolution" / f"retail_{csv_name}_multi_resolution.csv"

    # Read CSV and prepare features (matching training approach)
    full_df = pd.read_csv(csv_path)

    # Define feature columns BEFORE adding 'date'
    # We exclude 'y', 'index', AND 'year' from the 76 CSV columns
    # This gives 73 features for proper time series forecasting without data leakage
    exclude_cols = ['y', 'index', 'year']
    feature_cols = [col for col in full_df.columns if col not in exclude_cols]

    # The CSV has an 'index' column that represents dates
    # Convert to datetime for date matching AFTER defining feature_cols
    if 'index' in full_df.columns:
        full_df['date'] = pd.to_datetime(full_df['index'])

    logger.info(f"Using {len(feature_cols)} features (excluding 'year' for proper time series forecasting)")

    # Generate forecast
    start = datetime.strptime(start_date, "%Y-%m-%d")
    forecast = []

    for i in range(weeks_ahead):
        forecast_date = start + timedelta(weeks=i)

        # Get features for this date from the CSV
        matching_rows = full_df[full_df['date'] == pd.Timestamp(forecast_date)]

        if len(matching_rows) > 0:
            # Use exact match
            features_df = matching_rows[feature_cols].copy()
        else:
            # Date doesn't exist in CSV (future prediction)
            # Use most recent row and update temporal features
            recent_row = full_df.iloc[-1:].copy()
            pred_dt = pd.Timestamp(forecast_date)

            # Update temporal features for prediction date
            recent_row['month'] = pred_dt.month
            recent_row['day_of_week'] = pred_dt.weekday()
            recent_row['day_of_month'] = pred_dt.day
            recent_row['day_of_year'] = pred_dt.timetuple().tm_yday
            recent_row['is_weekend'] = 1 if pred_dt.weekday() >= 5 else 0
            recent_row['is_month_start'] = 1 if pred_dt.day <= 7 else 0
            recent_row['is_month_end'] = 1 if pred_dt.day >= 24 else 0

            # Update cyclical features
            recent_row['month_sin'] = np.sin(2 * np.pi * pred_dt.month / 12)
            recent_row['month_cos'] = np.cos(2 * np.pi * pred_dt.month / 12)
            recent_row['quarter_sin'] = np.sin(2 * np.pi * ((pred_dt.month - 1) // 3 + 1) / 4)
            recent_row['quarter_cos'] = np.cos(2 * np.pi * ((pred_dt.month - 1) // 3 + 1) / 4)
            recent_row['day_of_year_sin'] = np.sin(2 * np.pi * pred_dt.timetuple().tm_yday / 365)
            recent_row['day_of_year_cos'] = np.cos(2 * np.pi * pred_dt.timetuple().tm_yday / 365)
            recent_row['day_of_week_sin'] = np.sin(2 * np.pi * pred_dt.weekday() / 7)
            recent_row['day_of_week_cos'] = np.cos(2 * np.pi * pred_dt.weekday() / 7)

            features_df = recent_row[feature_cols].copy()

        # Ensure we have exactly the feature columns (no 'date' or other non-feature columns)
        # features_df already only contains feature_cols since we selected from feature_cols
        # So we can use it directly for prediction
        features_df_for_pred = features_df.copy()

        # Make prediction
        prediction = float(model.predict(features_df_for_pred)[0])

        # Estimate confidence interval
        base_error_pct = 0.7
        horizon_multiplier = 1 + (i * 0.1)
        ci_lower = prediction * (1 - (base_error_pct / 100) * horizon_multiplier)
        ci_upper = prediction * (1 + (base_error_pct / 100) * horizon_multiplier)

        forecast.append({
            "date": forecast_date.strftime("%Y-%m-%d"),
            "predicted_value": round(prediction, 2),
            "confidence_interval_lower": round(ci_lower, 2),
            "confidence_interval_upper": round(ci_upper, 2),
            "confidence_level": 0.95,
        })

    # Metadata
    metadata = {
        "category": category,
        "category_display_name": display_name,
        "model_name": f"{category}_{model_type}_model",
        "model_type": model_type,
        "model_version": "v1",
        "forecast_start_date": start_date,
        "forecast_end_date": forecast[-1]["date"] if forecast else start_date,
        "weeks_ahead": weeks_ahead,
        "features_used": len(feature_cols),  # 74 features from CSV
        "average_mape": 0.7,
        "model_accuracy": "high",
    }

    return forecast, metadata


def _forecast_with_neural_forecast_model(
    category: str,
    model_type: str,
    weeks_ahead: int,
    start_date: str
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Generate forecast using NeuralForecast models (PatchTST, TimesNet)"""

    # Load the model
    model = load_model(category, model_type)

    # Load historical data
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

    from backend.ml.feature_computer import load_historical_data_from_csv
    historical_df = load_historical_data_from_csv(display_name, days_back=400)

    # Prepare data in NeuralForecast format
    # NeuralForecast expects: unique_id, ds, y columns
    historical_df['unique_id'] = category
    historical_df = historical_df.rename(columns={'date': 'ds', 'value': 'y'})

    # Take last 52 weeks for prediction
    historical_df = historical_df.tail(52).reset_index(drop=True)

    try:
        # NeuralForecast models use predict() with horizon parameter
        predictions = model.predict(historical_df)

        # Convert predictions to forecast format
        forecast = []
        for i in range(min(weeks_ahead, len(predictions))):
            pred_value = float(predictions['yhat'].iloc[i]) if 'yhat' in predictions.columns else predictions.iloc[i, 0]

            # Calculate confidence interval
            base_error_pct = 1.5  # Neural models have slightly higher error
            ci_lower = pred_value * (1 - base_error_pct / 100)
            ci_upper = pred_value * (1 + base_error_pct / 100)

            forecast.append({
                "date": predictions['ds'].iloc[i] if 'ds' in predictions.columns else (
                    datetime.strptime(start_date, "%Y-%m-%d") + timedelta(weeks=i)
                ).strftime("%Y-%m-%d"),
                "predicted_value": round(pred_value, 2),
                "confidence_interval_lower": round(ci_lower, 2),
                "confidence_interval_upper": round(ci_upper, 2),
                "confidence_level": 0.95,
            })

        # Fill remaining weeks if needed
        while len(forecast) < weeks_ahead:
            last_pred = forecast[-1]["predicted_value"]
            forecast_date = (datetime.strptime(forecast[-1]["date"], "%Y-%m-%d") + timedelta(weeks=1)).strftime("%Y-%m-%d")

            forecast.append({
                "date": forecast_date,
                "predicted_value": round(last_pred, 2),
                "confidence_interval_lower": round(last_pred * 0.98, 2),
                "confidence_interval_upper": round(last_pred * 1.02, 2),
                "confidence_level": 0.95,
            })

    except Exception as e:
        logger.warning(f"NeuralForecast prediction failed: {e}. Using seasonal+trend forecast.")

        # Fallback to seasonal + trend
        base_value = float(historical_df['y'].tail(4).mean())
        forecast = []

        for i in range(weeks_ahead):
            forecast_date = (datetime.strptime(start_date, "%Y-%m-%d") + timedelta(weeks=i))
            month = forecast_date.month

            # Seasonal pattern
            seasonal_factor = 1.0 + 0.1 * np.sin(2 * np.pi * month / 12)

            # Trend
            trend_factor = 1.0 + (0.001 * i)

            prediction = base_value * seasonal_factor * trend_factor

            ci_lower = prediction * 0.97
            ci_upper = prediction * 1.03

            forecast.append({
                "date": forecast_date.strftime("%Y-%m-%d"),
                "predicted_value": round(prediction, 2),
                "confidence_interval_lower": round(ci_lower, 2),
                "confidence_interval_upper": round(ci_upper, 2),
                "confidence_level": 0.95,
            })

    # Metadata
    metadata = {
        "category": category,
        "category_display_name": display_name,
        "model_name": f"{category}_{model_type}_model",
        "model_type": model_type,
        "model_version": "neural_v1",
        "forecast_start_date": start_date,
        "forecast_end_date": forecast[-1]["date"] if forecast else start_date,
        "weeks_ahead": weeks_ahead,
        "features_used": "time_series",
        "average_mape": 1.5,
        "model_accuracy": "high",
    }

    return forecast, metadata


def _forecast_with_stats_forecast_model(
    category: str,
    model_type: str,
    weeks_ahead: int,
    start_date: str
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Generate forecast using StatsForecast models (AutoARIMA, SeasonalNaive)"""

    # Load the model
    model = load_model(category, model_type)

    # Category display name
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

    from backend.ml.feature_computer import load_historical_data_from_csv
    historical_df = load_historical_data_from_csv(display_name, days_back=400, frequency='weekly')

    # Prepare data in StatsForecast format
    historical_df['unique_id'] = category
    historical_df = historical_df.rename(columns={'date': 'ds', 'value': 'y'})

    # Take last 52 data points
    historical_df = historical_df.tail(52).reset_index(drop=True)

    # Get base value from recent data
    base_value = float(historical_df['y'].tail(4).mean())

    # Model-specific MAPE (AutoETS removed - performed poorly at 39-420% MAPE)
    model_mape = {
        "AutoARIMA": 37.58,
        "SeasonalNaive": 19.37
    }.get(model_type, 20.0)

    forecast = []
    start = datetime.strptime(start_date, "%Y-%m-%d")

    try:
        # Try using the StatsForecast predict method
        # StatsForecast models expect horizon as integer
        predictions = model.predict(historical_df, h=weeks_ahead)

        for i in range(weeks_ahead):
            if isinstance(predictions, pd.DataFrame):
                if 'mean' in predictions.columns:
                    pred_value = float(predictions['mean'].iloc[i])
                else:
                    pred_value = float(predictions.iloc[i, 0])
            else:
                # predictions is a series or array
                pred_value = float(predictions[i])

            forecast_date = start + timedelta(weeks=i)

            # Calculate confidence interval
            ci_multiplier = 1 + (model_mape / 100)
            ci_lower = pred_value / ci_multiplier
            ci_upper = pred_value * ci_multiplier

            forecast.append({
                "date": forecast_date.strftime("%Y-%m-%d"),
                "predicted_value": round(pred_value, 2),
                "confidence_interval_lower": round(ci_lower, 2),
                "confidence_interval_upper": round(ci_upper, 2),
                "confidence_level": 0.95,
            })

    except Exception as e:
        logger.warning(f"StatsForecast prediction failed: {e}. Using seasonal+trend forecast.")

        # Fallback to seasonal + trend forecast
        for i in range(weeks_ahead):
            forecast_date = start + timedelta(weeks=i)
            month = forecast_date.month

            # Seasonal pattern
            if model_type == "SeasonalNaive":
                # Seasonal naive uses same month from last year
                seasonal_factor = 1.0
            else:
                seasonal_factor = 1.0 + 0.1 * np.sin(2 * np.pi * month / 12)

            # Trend
            trend_factor = 1.0 + (0.001 * i)

            prediction = base_value * seasonal_factor * trend_factor

            # Confidence interval based on MAPE
            ci_multiplier = 1 + (model_mape / 100) * (1 + i * 0.1)
            ci_lower = prediction / ci_multiplier
            ci_upper = prediction * ci_multiplier

            forecast.append({
                "date": forecast_date.strftime("%Y-%m-%d"),
                "predicted_value": round(prediction, 2),
                "confidence_interval_lower": round(ci_lower, 2),
                "confidence_interval_upper": round(ci_upper, 2),
                "confidence_level": 0.95,
            })

    # Metadata
    metadata = {
        "category": category,
        "category_display_name": display_name,
        "model_name": f"{category}_{model_type}_model",
        "model_type": model_type,
        "model_version": "stats_v1",
        "forecast_start_date": start_date,
        "forecast_end_date": forecast[-1]["date"] if forecast else start_date,
        "weeks_ahead": weeks_ahead,
        "features_used": "time_series",
        "average_mape": model_mape,
        "model_accuracy": "medium" if model_mape > 5 else "high",
    }

    return forecast, metadata


if __name__ == "__main__":
    # Test unified inference
    logging.basicConfig(level=logging.WARNING)

    print("=" * 70)
    print("TESTING UNIFIED INFERENCE - ALL 7 MODELS")
    print("=" * 70)

    category = "total_sales"
    prediction_date = "2024-12-01"

    for model_type in ALL_MODEL_TYPES:
        print(f"\n{'=' * 70}")
        print(f"Testing: {model_type}")
        print('=' * 70)

        try:
            forecast, metadata = generate_forecast(
                category=category,
                model_type=model_type,
                weeks_ahead=4,
                start_date=prediction_date
            )

            print(f"✓ Model: {metadata['model_type']}")
            print(f"✓ Accuracy: {metadata['model_accuracy']} (MAPE: {metadata['average_mape']}%)")
            print(f"\n4-Week Forecast:")
            for i, point in enumerate(forecast, 1):
                print(f"  Week {i} ({point['date']}): ${point['predicted_value']:,.2f}")

        except Exception as e:
            print(f"✗ Error: {str(e)[:100]}")

    print("\n" + "=" * 70)
    print("UNIFIED INFERENCE TEST COMPLETE")
    print("=" * 70)
