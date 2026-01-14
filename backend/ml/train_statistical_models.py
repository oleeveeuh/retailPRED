#!/usr/bin/env python3
"""
Train Statistical Models (SeasonalNaive, AutoARIMA, AutoETS)

Trains statistical models with proper weekly frequency handling and CSV loading.
These models use historical time series data without feature engineering.
"""

import sys
from pathlib import Path
import logging
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive, AutoARIMA, AutoETS
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Category mappings
CATEGORIES = {
    'automobile_dealers': 'Automobile_Dealers',
    'building_material_and_garden_equipment': 'Building_Materials_Garden',
    'clothing_and_clothing_accessories_stores': 'Clothing_Accessories',
    'electronics_and_appliance_stores': 'Electronics_and_Appliances',
    'food_and_beverage_stores': 'Food_Beverage_Stores',
    'furniture_and_home_furnishings_stores': 'Furniture_Home_Furnishings',
    'gasoline_stations': 'Gasoline_Stations',
    'general_merchandise_stores': 'General_Merchandise',
    'health_and_personal_care_stores': 'Health_Personal_Care',
    'sporting_goods_hobby_and_musical_instrument_stores': 'Sporting_Goods_Hobby',
    'total_sales': 'Total_Retail_Sales',
}

# Statistical model types
MODEL_TYPES = ['SeasonalNaive', 'AutoARIMA', 'AutoETS']

# Data directory
DATA_DIR = Path(__file__).parent.parent.parent / "project_root" / "data_multi_resolution"
MODELS_DIR = Path(__file__).parent.parent.parent / "training_outputs" / "models"
MODELS_DIR.mkdir(exist_ok=True, parents=True)


def load_weekly_data(category_key: str) -> pd.DataFrame:
    """
    Load time series data from CSV and sample to weekly frequency

    Args:
        category_key: Category key (e.g., 'automobile_dealers')

    Returns:
        DataFrame with weekly sampled data
    """
    csv_path = DATA_DIR / f"retail_{category_key}_multi_resolution.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)

    # Convert index to datetime
    df['date'] = pd.to_datetime(df['index'])

    # Sort by date
    df = df.sort_values('date')

    # Sample to weekly frequency (take every 7th day starting from most recent)
    # This matches the inference code's approach
    weekly_df = df.iloc[::-1].iloc[::7].iloc[::-1].reset_index(drop=True)

    logger.info(f"Loaded {len(weekly_df)} weekly records from {len(df)} daily records")
    logger.info(f"  Date range: {weekly_df['date'].min()} to {weekly_df['date'].max()}")
    logger.info(f"  Value range: ${weekly_df['y'].min():,.2f} to ${weekly_df['y'].max():,.2f}")

    return weekly_df


def prepare_statsforecast_format(df: pd.DataFrame):
    """
    Convert DataFrame to StatsForecast format

    Args:
        df: DataFrame with 'date' and 'y' columns

    Returns:
        DataFrame in StatsForecast format (unique_id, ds, y)
    """
    sf_df = df[['date', 'y']].copy()
    sf_df.columns = ['ds', 'y']
    sf_df['unique_id'] = 0  # Single time series

    return sf_df


def train_statsmodel(model_type: str, train_df: pd.DataFrame, test_size: int = 52):
    """
    Train a statistical model

    Args:
        model_type: 'SeasonalNaive', 'AutoARIMA', or 'AutoETS'
        train_df: Training data
        test_size: Number of weeks to hold out for testing

    Returns:
        Trained model and metrics
    """
    logger.info(f"Training {model_type}...")

    # Split train/test
    train_data = train_df.iloc[:-test_size]
    test_data = train_df.iloc[-test_size:]

    # Prepare StatsForecast format
    train_sf = prepare_statsforecast_format(train_data)

    # Initialize model
    if model_type == 'SeasonalNaive':
        model = SeasonalNaive(season_length=52)
    elif model_type == 'AutoARIMA':
        model = AutoARIMA(season_length=52)
    elif model_type == 'AutoETS':
        model = AutoETS(season_length=52)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Create and fit StatsForecast
    fcst = StatsForecast(
        models=[model],
        freq='W',  # Weekly frequency
        n_jobs=-1
    )

    fcst.fit(train_sf)

    # Make predictions for test period
    forecast = fcst.predict(h=test_size)

    # Extract predictions
    if model_type in forecast.columns:
        predictions = forecast[model_type].values
    else:
        # Try alternative column names
        for col in forecast.columns:
            if col != 'unique_id' and col != 'ds':
                predictions = forecast[col].values
                break
        else:
            raise ValueError(f"Could not find predictions in forecast columns: {forecast.columns}")

    # Calculate metrics
    actuals = test_data['y'].values
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))

    # Calculate MAPE (handle zeros)
    mask = actuals != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask])) * 100
    else:
        mape = np.nan

    logger.info(f"  Train MAE: ${mae:,.2f}")
    logger.info(f"  Test MAE: ${mae:,.2f}")
    logger.info(f"  Test RMSE: ${rmse:,.2f}")
    logger.info(f"  Test MAPE: {mape:.2f}%" if not np.isnan(mape) else "  Test MAPE: N/A")

    return fcst, {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
    }


def train_category(category_key: str, category_display: str):
    """Train all statistical models for a category"""
    logger.info(f"\n{'='*80}")
    logger.info(f"Training: {category_display}")
    logger.info(f"{'='*80}\n")

    results = {}

    try:
        # Load data
        df = load_weekly_data(category_key)

        # Train each model type
        for model_type in MODEL_TYPES:
            try:
                model, metrics = train_statsmodel(model_type, df)

                # Save model
                model_filename = f"{model_type}_model.pkl"
                category_dir = MODELS_DIR / category_display
                category_dir.mkdir(exist_ok=True, parents=True)
                model_path = category_dir / model_filename

                model_data = {
                    'model': model,
                    'model_name': model_filename,
                    'is_trained': True,
                    'training_time': datetime.now().isoformat(),
                    'metrics': metrics,
                }

                joblib.dump(model_data, model_path)
                logger.info(f"  ✅ Saved: {category_display}/{model_filename}")

                results[model_type] = {
                    'status': 'success',
                    'metrics': metrics,
                }

            except Exception as e:
                logger.error(f"  ❌ {model_type}: {e}")
                import traceback
                traceback.print_exc()
                results[model_type] = {'status': 'failed', 'error': str(e)}

        return results

    except Exception as e:
        logger.error(f"Failed to train {category_display}: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'failed', 'error': str(e)}


def main():
    """Main training function"""
    logger.info("="*80)
    logger.info("TRAINING STATISTICAL MODELS (SeasonalNaive, AutoARIMA, AutoETS)")
    logger.info("="*80)
    logger.info(f"Categories: {len(CATEGORIES)}")
    logger.info(f"Model types: {MODEL_TYPES}")
    logger.info("="*80)

    all_results = {}
    start_time = datetime.now()

    for category_key, category_display in CATEGORIES.items():
        results = train_category(category_key, category_display)
        all_results[category_key] = results

    # Save summary
    duration = (datetime.now() - start_time).total_seconds()

    summary = {
        'training_date': datetime.now().isoformat(),
        'duration_seconds': duration,
        'model_types': MODEL_TYPES,
        'categories_trained': len(CATEGORIES),
        'results': all_results,
    }

    summary_path = Path(__file__).parent.parent.parent / "training_outputs" / "statistical_models_training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"Duration: {duration:.1f} seconds")
    logger.info(f"Summary: {summary_path}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
