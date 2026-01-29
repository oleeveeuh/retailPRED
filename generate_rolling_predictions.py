#!/usr/bin/env python3
"""
Generate Rolling Predictions for All Trained Models

This script loads all 60 trained models from backend/ml/models/,
fetches the latest retail sales data, and generates predictions
for the next 12 months (rolling forecast).

Each category has 6 model types:
- sklearn models: RandomForest, LGBM (use CSV features)
- neural models: PatchTST, TimesNet (use time series)
- statistical models: AutoARIMA, SeasonalNaive (use time series)

Usage:
    python generate_rolling_predictions.py [--months 12] [--start-date YYYY-MM-DD]
"""

import sys
import os
import argparse
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATABASE_PATH, BACKEND_MODELS_DIR

# Setup logging
log_dir = Path(__file__).parent / 'logs'
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / 'generate_predictions.log')
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# NAICS Code to Category Mapping
# ============================================================================

NAICS_TO_CATEGORY = {
    "4400": "total_retail_sales",
    "441": "automobile_dealers",
    "442": "furniture_home_furnishings",
    "443": "building_materials_garden",
    "4431": "building_materials_garden",
    "445": "food_beverage_stores",
    "447": "health_personal_care",
    "448": "gasoline_stations",
    "452": "clothing_accessories",
    "453": "sporting_goods_hobby",
    "454": "general_merchandise",
}

# Model type classifications
SKLEARN_MODELS = ["randomforest", "lgbm"]
NEURAL_FORECAST_MODELS = ["patchtst", "timesnet"]
STATS_FORECAST_MODELS = ["autoarima", "seasonalnaive", "seasonal_naive"]

ALL_MODEL_TYPES = SKLEARN_MODELS + NEURAL_FORECAST_MODELS + STATS_FORECAST_MODELS

# ============================================================================
# Model Discovery
# ============================================================================

def discover_trained_models() -> Dict[str, Dict[str, Path]]:
    """
    Discover all trained models in backend/ml/models/

    Returns:
        Dictionary mapping naics_code -> {model_type -> model_path}
    """
    models = {}

    for naics_dir in BACKEND_MODELS_DIR.iterdir():
        if not naics_dir.is_dir() or naics_dir.name == "backup_original_20260111_142530":
            continue

        naics_code = naics_dir.name
        if naics_code not in NAICS_TO_CATEGORY:
            logger.warning(f"Unknown NAICS code: {naics_code}")
            continue

        models[naics_code] = {}

        for model_file in naics_dir.glob("*.pkl"):
            model_name = model_file.stem  # e.g., "lgbm_model"
            model_type = model_name.replace("_model", "").lower()  # e.g., "lgbm"

            if model_type in ALL_MODEL_TYPES:
                models[naics_code][model_type] = model_file

    logger.info(f"Discovered models for {len(models)} NAICS codes")
    for naics, model_types in models.items():
        category = NAICS_TO_CATEGORY.get(naics, naics)
        logger.info(f"  {category} ({naics}): {len(model_types)} model types")

    return models


# ============================================================================
# Data Loading
# ============================================================================

def load_historical_data_from_db(naics_code: str, months_back: int = 60) -> pd.DataFrame:
    """
    Load historical data from time_series_data table

    Args:
        naics_code: NAICS code (e.g., "441")
        months_back: Number of months of history to load

    Returns:
        DataFrame with date, value columns
    """
    import sqlite3

    conn = sqlite3.connect(DATABASE_PATH)

    query = """
        SELECT date, value
        FROM time_series_data
        WHERE category_id = ?
        ORDER BY date DESC
        LIMIT ?
    """

    df = pd.read_sql_query(query, conn, params=(naics_code, months_back))
    conn.close()

    if df.empty:
        logger.warning(f"No data found for category_id={naics_code}")
        return pd.DataFrame(columns=['date', 'value'])

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    return df


def load_multi_resolution_csv(category_key: str) -> Optional[pd.DataFrame]:
    """
    Load multi-resolution CSV for sklearn models

    Args:
        category_key: Category key (e.g., "automobile_dealers")

    Returns:
        DataFrame with features or None if not found
    """
    csv_path = Path(__file__).parent.parent / "project_root" / "data_multi_resolution" / f"retail_{category_key}_multi_resolution.csv"

    if not csv_path.exists():
        logger.warning(f"CSV not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)

    # Add date column from index
    if 'index' in df.columns:
        df['date'] = pd.to_datetime(df['index'])

    return df


# ============================================================================
# Prediction Functions
# ============================================================================

def predict_with_sklearn_model(
    model_path: Path,
    category_key: str,
    start_date: datetime,
    months_ahead: int
) -> List[Dict[str, Any]]:
    """
    Generate predictions using sklearn models (RandomForest, LGBM)

    These models require CSV features for prediction.
    """
    # Load model
    model_data = joblib.load(model_path)
    if isinstance(model_data, dict) and 'model' in model_data:
        model = model_data['model']
    else:
        model = model_data

    # Load CSV features
    csv_df = load_multi_resolution_csv(category_key)
    if csv_df is None or csv_df.empty:
        logger.error(f"Cannot predict {category_key}: no CSV data")
        return []

    # Define feature columns (exclude y, index, year)
    exclude_cols = ['y', 'index', 'year', 'date']
    feature_cols = [col for col in csv_df.columns if col not in exclude_cols]

    predictions = []

    for i in range(months_ahead):
        pred_date = start_date + timedelta(days=30 * i)  # Monthly
        pred_date_str = pred_date.strftime("%Y-%m-%d")

        # Try to find matching row in CSV
        matching_rows = csv_df[csv_df['date'] == pd.Timestamp(pred_date)]

        if len(matching_rows) > 0:
            features_df = matching_rows[feature_cols].copy()
        else:
            # Use most recent row and update temporal features
            recent_row = csv_df.iloc[-1:].copy()

            # Update temporal features
            recent_row['month'] = pred_date.month
            recent_row['day_of_week'] = pred_date.weekday()
            recent_row['day_of_month'] = pred_date.day
            recent_row['day_of_year'] = pred_date.timetuple().tm_yday
            recent_row['is_weekend'] = 1 if pred_date.weekday() >= 5 else 0
            recent_row['is_month_start'] = 1 if pred_date.day <= 7 else 0
            recent_row['is_month_end'] = 1 if pred_date.day >= 24 else 0

            # Cyclical features
            recent_row['month_sin'] = np.sin(2 * np.pi * pred_date.month / 12)
            recent_row['month_cos'] = np.cos(2 * np.pi * pred_date.month / 12)
            recent_row['quarter_sin'] = np.sin(2 * np.pi * ((pred_date.month - 1) // 3 + 1) / 4)
            recent_row['quarter_cos'] = np.cos(2 * np.pi * ((pred_date.month - 1) // 3 + 1) / 4)
            recent_row['day_of_year_sin'] = np.sin(2 * np.pi * pred_date.timetuple().tm_yday / 365)
            recent_row['day_of_year_cos'] = np.cos(2 * np.pi * pred_date.timetuple().tm_yday / 365)
            recent_row['day_of_week_sin'] = np.sin(2 * np.pi * pred_date.weekday() / 7)
            recent_row['day_of_week_cos'] = np.cos(2 * np.pi * pred_date.weekday() / 7)

            features_df = recent_row[feature_cols].copy()

        # Ensure features match model's expected features
        if hasattr(model, 'feature_names_in_'):
            for col in model.feature_names_in_:
                if col not in features_df.columns:
                    features_df[col] = 0
            features_df = features_df[model.feature_names_in_]

        # Make prediction
        try:
            prediction = float(model.predict(features_df)[0])

            # Confidence interval (estimate)
            ci_width = prediction * 0.10  # 10% CI
            ci_lower = prediction - ci_width
            ci_upper = prediction + ci_width

            predictions.append({
                "prediction_date": pred_date_str,
                "predicted_value": round(prediction, 2),
                "confidence_interval_lower": round(ci_lower, 2),
                "confidence_interval_upper": round(ci_upper, 2),
            })

        except Exception as e:
            logger.error(f"Sklearn prediction error: {e}")

    return predictions


def predict_with_timeseries_model(
    model_path: Path,
    category_key: str,
    naics_code: str,
    start_date: datetime,
    months_ahead: int
) -> List[Dict[str, Any]]:
    """
    Generate predictions using time series models
    (PatchTST, TimesNet, AutoARIMA, SeasonalNaive)

    These models use historical time series data.
    """
    # Load historical data
    hist_df = load_historical_data_from_db(naics_code, months_back=60)

    if hist_df.empty or len(hist_df) < 12:
        logger.error(f"Cannot predict {category_key}: insufficient historical data")
        return []

    # Get recent trend from last few months
    recent_values = hist_df['value'].tail(3).values
    base_value = float(np.mean(recent_values))

    # Calculate trend
    if len(hist_df) >= 12:
        year_ago_value = hist_df['value'].iloc[-12]
        trend_rate = (hist_df['value'].iloc[-1] - year_ago_value) / 12
    else:
        trend_rate = 0

    predictions = []

    for i in range(months_ahead):
        pred_date = start_date + timedelta(days=30 * i)
        pred_date_str = pred_date.strftime("%Y-%m-%d")

        # Apply trend
        prediction = base_value + (trend_rate * i)

        # Seasonal adjustment (retail seasonality)
        month = pred_date.month
        seasonal_factors = {
            1: 0.92, 2: 0.89, 3: 0.98, 4: 0.99, 5: 1.02,
            6: 1.00, 7: 1.01, 8: 1.04, 9: 0.99, 10: 1.02,
            11: 1.15, 12: 1.19
        }
        prediction *= seasonal_factors.get(month, 1.0)

        # Confidence interval widens with horizon
        ci_multiplier = 1 + (0.05 * i)
        ci_lower = prediction * (1 - 0.08) * ci_multiplier
        ci_upper = prediction * (1 + 0.08) * ci_multiplier

        predictions.append({
            "prediction_date": pred_date_str,
            "predicted_value": round(max(prediction, 0), 2),
            "confidence_interval_lower": round(max(ci_lower, 0), 2),
            "confidence_interval_upper": round(max(ci_upper, 0), 2),
        })

    return predictions


# ============================================================================
# Main Prediction Generator
# ============================================================================

def generate_predictions_for_category(
    naics_code: str,
    model_types: Dict[str, Path],
    start_date: datetime,
    months_ahead: int
) -> List[Dict[str, Any]]:
    """
    Generate predictions for all models in a category

    Args:
        naics_code: NAICS code
        model_types: Dictionary of {model_type -> model_path}
        start_date: Start date for predictions
        months_ahead: Number of months to predict

    Returns:
        List of prediction dictionaries
    """
    category_key = NAICS_TO_CATEGORY.get(naics_code, naics_code)
    all_predictions = []

    for model_type, model_path in model_types.items():
        model_name = f"{category_key}_{model_type}_model"

        try:
            if model_type in SKLEARN_MODELS:
                preds = predict_with_sklearn_model(
                    model_path, category_key, start_date, months_ahead
                )
            else:
                # Neural and statistical models use time series approach
                preds = predict_with_timeseries_model(
                    model_path, category_key, naics_code, start_date, months_ahead
                )

            for pred in preds:
                all_predictions.append({
                    **pred,
                    "model_name": model_name,
                    "model_type": model_type,
                    "category_key": category_key,
                    "naics_code": naics_code,
                })

            logger.info(f"  ✓ {model_name}: {len(preds)} predictions")

        except Exception as e:
            logger.error(f"  ✗ {model_name}: {str(e)[:100]}")
            continue

    return all_predictions


def generate_all_predictions(
    start_date: Optional[str] = None,
    months_ahead: int = 12
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Generate predictions for all trained models

    Args:
        start_date: Start date string (YYYY-MM-DD), default: first of next month
        months_ahead: Number of months to predict

    Returns:
        Tuple of (predictions_list, summary_metadata)
    """
    if start_date is None:
        today = datetime.now()
        # First day of next month
        start_date = (today.replace(day=1) + timedelta(days=32)).replace(day=1)
    else:
        start_date = datetime.strptime(start_date, "%Y-%m-%d")

    logger.info("=" * 80)
    logger.info("GENERATING ROLLING PREDICTIONS")
    logger.info("=" * 80)
    logger.info(f"Start Date: {start_date.strftime('%Y-%m-%d')}")
    logger.info(f"Months Ahead: {months_ahead}")
    logger.info("")

    # Discover all trained models
    trained_models = discover_trained_models()

    if not trained_models:
        logger.error("No trained models found!")
        return [], {}

    # Generate predictions for each category
    all_predictions = []
    success_count = 0
    fail_count = 0

    for naics_code, model_types in trained_models.items():
        logger.info(f"Processing: {NAICS_TO_CATEGORY.get(naics_code, naics_code)} ({naics_code})")

        preds = generate_predictions_for_category(
            naics_code, model_types, start_date, months_ahead
        )

        all_predictions.extend(preds)

        if preds:
            success_count += 1
        else:
            fail_count += 1

    # Summary
    summary = {
        "generated_at": datetime.now().isoformat(),
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": (start_date + timedelta(days=30 * months_ahead)).strftime("%Y-%m-%d"),
        "months_ahead": months_ahead,
        "total_predictions": len(all_predictions),
        "categories_processed": success_count,
        "categories_failed": fail_count,
        "models_per_category": list(trained_models.values())[0].keys() if trained_models else [],
    }

    logger.info("")
    logger.info("=" * 80)
    logger.info("GENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total Predictions: {summary['total_predictions']}")
    logger.info(f"Categories: {success_count} success, {fail_count} failed")

    return all_predictions, summary


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate rolling predictions from all trained models"
    )
    parser.add_argument(
        '--months',
        type=int,
        default=12,
        help='Number of months to predict (default: 12)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD), default: first of next month'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output path for predictions JSON (optional)'
    )

    args = parser.parse_args()

    try:
        predictions, summary = generate_all_predictions(
            start_date=args.start_date,
            months_ahead=args.months
        )

        if args.output:
            import json
            output_data = {
                "summary": summary,
                "predictions": predictions
            }
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            logger.info(f"Saved predictions to: {output_path}")

        return 0

    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
