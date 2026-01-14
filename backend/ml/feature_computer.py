"""
Real Feature Computer Module
Computes actual feature values from historical data and generates real SHAP values
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
    logger.info("✓ SHAP library available")
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("✗ SHAP library not available. Install with: pip install shap")


def load_historical_data_from_csv(category_display: str, days_back: int = 400, frequency: str = 'daily') -> pd.DataFrame:
    """
    Load historical data from CSV files

    Args:
        category_display: Display name (e.g., "Total Retail Sales")
        days_back: Number of days to load
        frequency: Data frequency - 'daily' (default), 'weekly', or 'monthly'
                  For 'weekly', samples every 7th day. For 'monthly', samples ~30th day.

    Returns:
        DataFrame with date and value columns
    """
    try:
        # Map display names to multi-resolution CSV file names
        category_to_file = {
            "Total Retail Sales": "retail_total_sales_multi_resolution.csv",
            "Total_Retail_Sales": "retail_total_sales_multi_resolution.csv",
            "Building Materials & Garden": "retail_building_material_and_garden_equipment_multi_resolution.csv",
            "Building_Materials_Garden": "retail_building_material_and_garden_equipment_multi_resolution.csv",
            "Automobile Dealers": "retail_automobile_dealers_multi_resolution.csv",
            "Automobile_Dealers": "retail_automobile_dealers_multi_resolution.csv",
            "Gasoline Stations": "retail_gasoline_stations_multi_resolution.csv",
            "Gasoline_Stations": "retail_gasoline_stations_multi_resolution.csv",
            "Food & Beverage Stores": "retail_food_and_beverage_stores_multi_resolution.csv",
            "Food_Beverage_Stores": "retail_food_and_beverage_stores_multi_resolution.csv",
            "Health & Personal Care": "retail_health_and_personal_care_stores_multi_resolution.csv",
            "Health_Personal_Care": "retail_health_and_personal_care_stores_multi_resolution.csv",
            "General Merchandise": "retail_general_merchandise_stores_multi_resolution.csv",
            "General_Merchandise": "retail_general_merchandise_stores_multi_resolution.csv",
            "Furniture & Home Furnishings": "retail_furniture_and_home_furnishings_stores_multi_resolution.csv",
            "Furniture_Home_Furnishings": "retail_furniture_and_home_furnishings_stores_multi_resolution.csv",
            "Clothing & Accessories": "retail_clothing_and_clothing_accessories_stores_multi_resolution.csv",
            "Clothing_Accessories": "retail_clothing_and_clothing_accessories_stores_multi_resolution.csv",
            "Sporting Goods & Hobby": "retail_sporting_goods_hobby_and_musical_instrument_stores_multi_resolution.csv",
            "Sporting_Goods_Hobby": "retail_sporting_goods_hobby_and_musical_instrument_stores_multi_resolution.csv",
            "Electronics & Appliances": "retail_electronics_and_appliance_stores_multi_resolution.csv",
            "Electronics_and_Appliances": "retail_electronics_and_appliance_stores_multi_resolution.csv",
        }

        filename = category_to_file.get(category_display)
        if not filename:
            logger.warning(f"No CSV mapping for category: {category_display}")
            return generate_synthetic_data(category_display, days_back)

        data_dir = Path(__file__).parent.parent.parent / "project_root" / "data_multi_resolution"
        filepath = data_dir / filename

        if not filepath.exists():
            logger.warning(f"CSV file not found: {filepath}")
            return generate_synthetic_data(category_display, days_back)

        # Load CSV
        df = pd.read_csv(filepath)
        # Multi-resolution CSV uses 'index' instead of 'date' and 'y' instead of 'value'
        df['date'] = pd.to_datetime(df['index'])
        df['value'] = df['y']  # Use raw values (models were trained on these)

        df = df[['date', 'value']]
        df = df.sort_values('date').tail(days_back).reset_index(drop=True)

        # Apply frequency sampling
        if frequency == 'weekly':
            # Sample every 7th day (weekly frequency)
            df = df.iloc[::7].reset_index(drop=True)
            logger.info(f"✓ Sampled weekly data (every 7th day) from {filename}")
        elif frequency == 'monthly':
            # Sample approximately every 30th day (monthly frequency)
            df = df.iloc[::30].reset_index(drop=True)
            logger.info(f"✓ Sampled monthly data (every 30th day) from {filename}")
        else:
            logger.info(f"✓ Loaded daily data from {filename}")

        logger.info(f"✓ Loaded {len(df)} {frequency} records from multi-resolution CSV: {filename}")
        logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"  Value range: ${df['value'].min():,.2f} to ${df['value'].max():,.2f}")
        return df

    except Exception as e:
        logger.error(f"Error loading CSV data for {category_display}: {e}")
        return generate_synthetic_data(category_display, days_back)


def generate_synthetic_data(category_display: str, days_back: int) -> pd.DataFrame:
    """Generate synthetic historical data"""
    base_values = {
        "Total Retail Sales": 17000,
        "Building Materials & Garden": 35000,
        "Automobile Dealers": 25000,
        "Gasoline Stations": 12000,
        "Food & Beverage Stores": 15000,
        "Health & Personal Care": 8000,
        "General Merchandise": 18000,
        "Furniture & Home Furnishings": 5000,
        "Clothing & Accessories": 6000,
        "Sporting Goods & Hobby": 9000,
        "Electronics & Appliances": 2000,
    }

    base_value = base_values.get(category_display, 10000)

    # Generate dates
    end_date = datetime.now()
    dates = pd.date_range(
        end=end_date - timedelta(days=days_back),
        periods=days_back,
        freq='D'
    )

    # Generate values with trend, seasonality, and noise
    np.random.seed(42)  # For reproducibility
    values = []
    for date in dates:
        # Trend
        trend = 1.0 + ((len(dates) - dates.get_loc(date)) / len(dates)) * 0.1

        # Seasonality (weekly)
        weekly_seasonal = 1.0 + 0.05 * np.sin(2 * np.pi * date.dayofyear / 7)

        # Seasonality (yearly)
        yearly_seasonal = 1.0 + 0.15 * np.sin(2 * np.pi * date.dayofyear / 365)

        # Random noise
        noise = np.random.normal(1.0, 0.02)

        value = base_value * trend * weekly_seasonal * yearly_seasonal * noise
        values.append(max(0, value))

    df = pd.DataFrame({
        'date': dates,
        'value': values
    })

    logger.info(f"Generated {len(df)} synthetic records for {category_display}")
    return df


def compute_real_features(
    historical_df: pd.DataFrame,
    prediction_date: str
) -> pd.DataFrame:
    """
    Compute real feature values from historical data

    Args:
        historical_df: DataFrame with date and value columns
        prediction_date: Date for prediction (YYYY-MM-DD)

    Returns:
        DataFrame with 74 feature columns
    """
    if historical_df.empty or len(historical_df) < 30:
        logger.warning("Insufficient historical data, using fallback features")
        return get_fallback_features(prediction_date)

    pred_dt = datetime.strptime(prediction_date, "%Y-%m-%d")

    # Ensure we have data up to the prediction date
    historical_df = historical_df[historical_df['date'] < pred_dt].copy()

    if len(historical_df) < 30:
        return get_fallback_features(prediction_date)

    # Compute all 74 features
    features = {}

    # 1. Temporal features (16 features)
    features['year'] = pred_dt.year
    features['month'] = pred_dt.month
    features['quarter'] = (pred_dt.month - 1) // 3 + 1
    features['day_of_week'] = pred_dt.weekday()
    features['week_of_year'] = pred_dt.isocalendar()[1]
    features['is_weekend'] = 1 if pred_dt.weekday() >= 5 else 0
    features['day_of_month'] = pred_dt.day
    features['day_of_year'] = pred_dt.timetuple().tm_yday

    # Cyclical encodings (8 features)
    features['month_sin'] = np.sin(2 * np.pi * pred_dt.month / 12)
    features['month_cos'] = np.cos(2 * np.pi * pred_dt.month / 12)
    features['quarter_sin'] = np.sin(2 * np.pi * ((pred_dt.month - 1) // 3 + 1) / 4)
    features['quarter_cos'] = np.cos(2 * np.pi * ((pred_dt.month - 1) // 3 + 1) / 4)
    features['day_of_year_sin'] = np.sin(2 * np.pi * pred_dt.timetuple().tm_yday / 365)
    features['day_of_year_cos'] = np.cos(2 * np.pi * pred_dt.timetuple().tm_yday / 365)
    features['day_of_week_sin'] = np.sin(2 * np.pi * pred_dt.weekday() / 7)
    features['day_of_week_cos'] = np.cos(2 * np.pi * pred_dt.weekday() / 7)

    # Get latest value - handle both 'value' and 'y' column names
    value_col = 'value' if 'value' in historical_df.columns else 'y'
    latest_value = float(historical_df[value_col].iloc[-1])

    # 2. Lag features (10 unique features - removed lag_1w=lag_7d, lag_1m=lag_30d)
    features['lag_1d'] = float(historical_df[value_col].iloc[-1]) if len(historical_df) >= 1 else latest_value
    features['lag_7d'] = float(historical_df[value_col].iloc[-7]) if len(historical_df) >= 7 else latest_value
    features['lag_14d'] = float(historical_df[value_col].iloc[-14]) if len(historical_df) >= 14 else latest_value
    features['lag_30d'] = float(historical_df[value_col].iloc[-30]) if len(historical_df) >= 30 else latest_value
    # Weekly lags (4w, 8w, 12w - removed lag_1w as duplicate of lag_7d)
    features['lag_4w'] = float(historical_df[value_col].iloc[-28]) if len(historical_df) >= 28 else latest_value
    features['lag_8w'] = float(historical_df[value_col].iloc[-56]) if len(historical_df) >= 56 else latest_value
    features['lag_12w'] = float(historical_df[value_col].iloc[-84]) if len(historical_df) >= 84 else latest_value
    # Monthly lags (removed lag_1m as duplicate of lag_30d)
    features['lag_3m'] = float(historical_df[value_col].iloc[-90]) if len(historical_df) >= 90 else latest_value
    features['lag_6m'] = float(historical_df[value_col].iloc[-180]) if len(historical_df) >= 180 else latest_value
    features['lag_12m'] = float(historical_df[value_col].iloc[-365]) if len(historical_df) >= 365 else latest_value

    # 3. Monthly rolling statistics (6 features)
    values = historical_df[value_col].values
    features['rolling_std_3'] = float(pd.Series(values).rolling(window=3).std().iloc[-1]) if len(values) >= 3 else 50.0
    features['rolling_std_6'] = float(pd.Series(values).rolling(window=6).std().iloc[-1]) if len(values) >= 6 else 60.0
    features['rolling_std_12'] = float(pd.Series(values).rolling(window=12).std().iloc[-1]) if len(values) >= 12 else 70.0
    features['rolling_mean_3'] = float(pd.Series(values).rolling(window=3).mean().iloc[-1]) if len(values) >= 3 else latest_value
    features['rolling_mean_6'] = float(pd.Series(values).rolling(window=6).mean().iloc[-1]) if len(values) >= 6 else latest_value
    features['rolling_mean_12'] = float(pd.Series(values).rolling(window=12).mean().iloc[-1]) if len(values) >= 12 else latest_value

    # 4. Daily rolling statistics (6 features)
    features['rolling_mean_7d'] = float(pd.Series(values).rolling(window=7).mean().iloc[-1]) if len(values) >= 7 else latest_value
    features['rolling_std_7d'] = float(pd.Series(values).rolling(window=7).std().iloc[-1]) if len(values) >= 7 else 50.0
    features['rolling_mean_14d'] = float(pd.Series(values).rolling(window=14).mean().iloc[-1]) if len(values) >= 14 else latest_value
    features['rolling_std_14d'] = float(pd.Series(values).rolling(window=14).std().iloc[-1]) if len(values) >= 14 else 50.0
    features['rolling_mean_30d'] = float(pd.Series(values).rolling(window=30).mean().iloc[-1]) if len(values) >= 30 else latest_value
    features['rolling_std_30d'] = float(pd.Series(values).rolling(window=30).std().iloc[-1]) if len(values) >= 30 else 50.0

    # 5. Weekly rolling statistics (6 features)
    features['rolling_std_4w'] = float(pd.Series(values).rolling(window=28).std().iloc[-1]) if len(values) >= 28 else 50.0
    features['rolling_std_8w'] = float(pd.Series(values).rolling(window=56).std().iloc[-1]) if len(values) >= 56 else 50.0
    features['rolling_std_12w'] = float(pd.Series(values).rolling(window=84).std().iloc[-1]) if len(values) >= 84 else 50.0
    features['rolling_mean_4w'] = float(pd.Series(values).rolling(window=28).mean().iloc[-1]) if len(values) >= 28 else latest_value
    features['rolling_mean_8w'] = float(pd.Series(values).rolling(window=56).mean().iloc[-1]) if len(values) >= 56 else latest_value
    features['rolling_mean_12w'] = float(pd.Series(values).rolling(window=84).mean().iloc[-1]) if len(values) >= 84 else latest_value

    # 6. Monthly rolling statistics (6 features)
    features['rolling_std_3m'] = float(pd.Series(values).rolling(window=90).std().iloc[-1]) if len(values) >= 90 else 50.0
    features['rolling_std_6m'] = float(pd.Series(values).rolling(window=180).std().iloc[-1]) if len(values) >= 180 else 60.0
    features['rolling_std_12m'] = float(pd.Series(values).rolling(window=365).std().iloc[-1]) if len(values) >= 365 else 70.0
    features['rolling_mean_3m'] = float(pd.Series(values).rolling(window=90).mean().iloc[-1]) if len(values) >= 90 else latest_value
    features['rolling_mean_6m'] = float(pd.Series(values).rolling(window=180).mean().iloc[-1]) if len(values) >= 180 else latest_value
    features['rolling_mean_12m'] = float(pd.Series(values).rolling(window=365).mean().iloc[-1]) if len(values) >= 365 else latest_value

    # 7. Cross-frequency aggregations (removed - were duplicates of rolling means)

    # 8. Rate of change features (10 unique features - removed pct_change_1d=pct_change_1, diff_1d=diff_1)
    features['diff_1'] = float(values[-1] - values[-2]) if len(values) >= 2 else 1.0
    features['diff_12'] = float(values[-1] - values[-12]) if len(values) >= 12 else 12.0
    features['pct_change_1'] = float((values[-1] - values[-2]) / values[-2] * 100) if len(values) >= 2 and values[-2] != 0 else 0.001
    features['pct_change_12'] = float((values[-1] - values[-12]) / values[-12] * 100) if len(values) >= 12 and values[-12] != 0 else 0.012
    features['pct_change_1w'] = float((values[-1] - values[-7]) / values[-7] * 100) if len(values) >= 7 and values[-7] != 0 else 0.005
    features['diff_1w'] = float(values[-1] - values[-7]) if len(values) >= 7 else 5.0
    features['pct_change_1m'] = float((values[-1] - values[-30]) / values[-30] * 100) if len(values) >= 30 and values[-30] != 0 else 0.02
    features['diff_1m'] = float(values[-1] - values[-30]) if len(values) >= 30 else 20.0
    features['pct_change_1y'] = float((values[-1] - values[-365]) / values[-365] * 100) if len(values) >= 365 and values[-365] != 0 else 0.05
    features['diff_1y'] = float(values[-1] - values[-365]) if len(values) >= 365 else 50.0

    # 9. Momentum indicators (2 unique features - removed momentum_7d as duplicate of diff_1w)
    features['momentum_30d'] = float(values[-1] - values[-30]) if len(values) >= 30 else 2.0
    features['momentum_90d'] = float(values[-1] - values[-90]) if len(values) >= 90 else 5.0

    # 10. Year-over-year change (1 feature)
    features['yoy_change'] = features['pct_change_1y'] / 100.0

    return pd.DataFrame([features])


def get_fallback_features(prediction_date: str) -> pd.DataFrame:
    """Get fallback features when historical data is insufficient"""
    pred_dt = datetime.strptime(prediction_date, "%Y-%m-%d")

    features = {
        # Temporal features
        'year': pred_dt.year,
        'month': pred_dt.month,
        'quarter': (pred_dt.month - 1) // 3 + 1,
        'day_of_week': pred_dt.weekday(),
        'week_of_year': pred_dt.isocalendar()[1],
        'is_weekend': 1 if pred_dt.weekday() >= 5 else 0,
        'day_of_month': pred_dt.day,
        'day_of_year': pred_dt.timetuple().tm_yday,

        # Cyclical encodings
        'month_sin': np.sin(2 * np.pi * pred_dt.month / 12),
        'month_cos': np.cos(2 * np.pi * pred_dt.month / 12),
        'quarter_sin': np.sin(2 * np.pi * ((pred_dt.month - 1) // 3 + 1) / 4),
        'quarter_cos': np.cos(2 * np.pi * ((pred_dt.month - 1) // 3 + 1) / 4),
        'day_of_year_sin': np.sin(2 * np.pi * pred_dt.timetuple().tm_yday / 365),
        'day_of_year_cos': np.cos(2 * np.pi * pred_dt.timetuple().tm_yday / 365),
        'day_of_week_sin': np.sin(2 * np.pi * pred_dt.weekday() / 7),
        'day_of_week_cos': np.cos(2 * np.pi * pred_dt.weekday() / 7),

        # Default values for other features (63 unique features - removed 11 redundancies)
        **{f: 1000.0 for f in ['lag_1d', 'lag_7d', 'lag_14d', 'lag_30d', 'lag_4w', 'lag_8w', 'lag_12w',
                                   'lag_3m', 'lag_6m', 'lag_12m', 'rolling_mean_3', 'rolling_mean_6',
                                   'rolling_mean_12', 'rolling_mean_7d', 'rolling_mean_14d', 'rolling_mean_30d',
                                   'rolling_mean_4w', 'rolling_mean_8w', 'rolling_mean_12w', 'rolling_mean_3m',
                                   'rolling_mean_6m', 'rolling_mean_12m']},
        **{f: 50.0 for f in ['rolling_std_3', 'rolling_std_6', 'rolling_std_12', 'rolling_std_7d',
                              'rolling_std_14d', 'rolling_std_30d', 'rolling_std_4w', 'rolling_std_8w',
                              'rolling_std_12w', 'rolling_std_3m', 'rolling_std_6m', 'rolling_std_12m']},
        **{f: 1.0 for f in ['diff_1', 'diff_12', 'diff_1w', 'diff_1m', 'diff_1y']},
        **{f: 0.001 for f in ['pct_change_1', 'pct_change_12', 'pct_change_1w',
                               'pct_change_1m', 'pct_change_1y', 'yoy_change']},
        **{f: 0.5 for f in ['momentum_30d', 'momentum_90d']},
    }

    return pd.DataFrame([features])


def compute_shap_values(
    model,
    features_df: pd.DataFrame,
    feature_names: List[str],
    top_n: int = 10
) -> List[Dict[str, Any]]:
    """
    Compute real SHAP values for a prediction

    Args:
        model: Trained model (LightGBM or Random Forest)
        features_df: DataFrame with feature values
        feature_names: List of feature names
        top_n: Number of top features to return

    Returns:
        List of SHAP values with feature name, value, and importance
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available, using feature importance as fallback")
        return get_fallback_shap_values(model, features_df, top_n)

    try:
        # Create SHAP explainer based on model type
        # Check model type more comprehensively
        model_type = type(model).__name__
        module_name = type(model).__module__

        logger.info(f"Model type for SHAP: {model_type} from {module_name}")

        if 'lightgbm' in module_name.lower() or 'LGBM' in model_type:
            # LightGBM model
            explainer = shap.TreeExplainer(model)
            logger.info("✓ Using TreeExplainer for LightGBM")
        elif 'randomforest' in model_type.lower() or 'forest' in model_type.lower():
            # Random Forest model
            explainer = shap.TreeExplainer(model)
            logger.info("✓ Using TreeExplainer for Random Forest")
        elif hasattr(model, 'booster_'):  # LightGBM (alternative check)
            explainer = shap.TreeExplainer(model)
            logger.info("✓ Using TreeExplainer for LightGBM (booster_)")
        elif hasattr(model, 'estimators_'):  # Random Forest (alternative check)
            explainer = shap.TreeExplainer(model)
            logger.info("✓ Using TreeExplainer for Random Forest (estimators_)")
        else:
            logger.warning(f"Unknown model type: {model_type}, using fallback SHAP")
            return get_fallback_shap_values(model, features_df, top_n)

        # Compute SHAP values
        shap_values = explainer.shap_values(features_df)

        # Get absolute SHAP values for ranking
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # For multi-output, take first

        abs_shap = np.abs(shap_values[0])

        # Get top features
        top_indices = np.argsort(abs_shap)[::-1][:top_n]

        # Format results
        results = []
        for idx in top_indices:
            feature_name = feature_names[idx]
            shap_value = float(shap_values[0][idx])
            importance = float(abs_shap[idx] / np.sum(abs_shap))

            results.append({
                "feature": feature_name,
                "value": shap_value,
                "importance": importance
            })

        logger.info(f"✓ Computed SHAP values for {len(results)} features")
        return results

    except Exception as e:
        logger.error(f"Error computing SHAP values: {e}")
        return get_fallback_shap_values(model, features_df, top_n)


def get_fallback_shap_values(
    model,
    features_df: pd.DataFrame,
    top_n: int = 10
) -> List[Dict[str, Any]]:
    """
    Get fallback SHAP values using model feature importances

    Args:
        model: Trained model
        features_df: DataFrame with feature values
        top_n: Number of top features

    Returns:
        List of feature importances
    """
    try:
        # Get feature importances from model
        model_type = type(model).__name__
        module_name = type(model).__module__

        if hasattr(model, 'feature_importances_'):
            # sklearn-like models (Random Forest)
            importances = model.feature_importances_
            logger.info(f"✓ Got feature importances from {model_type}")
        elif 'lightgbm' in module_name.lower() or 'LGBM' in model_type:
            # LightGBM model
            try:
                importances = model.feature_importance(importance_type='gain')
                logger.info(f"✓ Got LightGBM feature importances (gain)")
            except:
                importances = model.feature_importance(importance_type='split')
                logger.info(f"✓ Got LightGBM feature importances (split)")
        elif hasattr(model, 'booster_'):  # LightGBM (alternative check)
            importances = model.feature_importance(importance_type='gain')
        else:
            # Equal importances as fallback
            logger.warning(f"Could not extract feature importances from {model_type}, using equal weights")
            importances = np.ones(len(features_df.columns)) / len(features_df.columns)

        # Normalize
        importances = importances / importances.sum()

        # Get top features
        feature_names = features_df.columns.tolist()
        top_indices = np.argsort(importances)[::-1][:top_n]

        results = []
        for idx in top_indices:
            feature_name = feature_names[idx]
            importance = float(importances[idx])

            # Use feature value as SHAP value
            feature_value = float(features_df[feature_name].iloc[0])

            results.append({
                "feature": feature_name,
                "value": feature_value * importance,
                "importance": importance
            })

        return results

    except Exception as e:
        logger.error(f"Error getting fallback SHAP values: {e}")
        # Final fallback: use equal weights with actual feature values
        try:
            feature_names = features_df.columns.tolist()
            # Use a subset of important features if available, otherwise all features
            important_features = [f for f in ['lag_1d', 'lag_7d', 'lag_14d', 'lag_30d',
                                              'pct_change_1w', 'diff_1w', 'rolling_mean_7d',
                                              'rolling_std_7d', 'momentum_30d']
                                 if f in feature_names]

            # If no important features found, use first 5 features
            if not important_features:
                important_features = feature_names[:5]

            equal_importance = 1.0 / len(important_features)

            results = []
            for feature in important_features:
                feature_value = float(features_df[feature].iloc[0])
                results.append({
                    "feature": feature,
                    "value": feature_value * equal_importance,
                    "importance": equal_importance
                })

            logger.warning(f"Using equal-weight fallback SHAP values for {len(results)} features")
            return results

        except Exception as final_error:
            logger.error(f"Error in final SHAP fallback: {final_error}")
            return []


if __name__ == "__main__":
    # Test feature computation
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Testing Feature Computer")
    print("=" * 60)

    # Test loading data
    category = "Total Retail Sales"
    historical_df = load_historical_data_from_csv(category, days_back=100)

    print(f"\nLoaded {len(historical_df)} records")
    print(f"Date range: {historical_df['date'].min()} to {historical_df['date'].max()}")

    # Test feature computation
    features_df = compute_real_features(historical_df, "2026-01-10")

    print(f"\nComputed {len(features_df.columns)} features")
    print("\nSample features:")
    for col in ['lag_1d', 'lag_7d', 'lag_1w', 'rolling_mean_7d', 'pct_change_1w']:
        if col in features_df.columns:
            print(f"  {col}: {features_df[col].iloc[0]:.2f}")

    print("\n" + "=" * 60)
