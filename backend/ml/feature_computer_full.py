"""
Full Feature Computer for 242-Feature Model

Generates all 242 features required by the trained RandomForest models:
- 64 stock features (8 tickers × 8 features each)
- 50+ economic indicator features
- 128 temporal/time series features
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Import external data loader
from ml.external_data_loader import get_external_data_loader


def compute_full_features(
    historical_df: pd.DataFrame,
    prediction_date: str,
    category_key: str = "total_sales"
) -> pd.DataFrame:
    """
    Compute all 242 features required by the trained models

    Args:
        historical_df: Historical retail sales data with 'date' and 'value' columns
        prediction_date: Date string (YYYY-MM-DD) for prediction
        category_key: Category key (e.g., 'total_sales')

    Returns:
        DataFrame with all 242 features
    """
    logger.info(f"Computing all 242 features for {category_key} as of {prediction_date}")

    # Get external data loader
    external_loader = get_external_data_loader()

    # Initialize features dictionary
    features = {}

    # 1. TARGET VARIABLE FEATURES (5 features: y_lag_1, y_lag_3, y_lag_6, y_lag_12, y_pct_change)
    target_features = compute_target_features(historical_df, prediction_date)
    features.update(target_features)

    # 2. TEMPORAL FEATURES (37 features)
    temporal_features = compute_temporal_features(prediction_date)
    features.update(temporal_features)

    # 3. LAG FEATURES (5 features: lag_1, lag_2, lag_3, lag_6, lag_12)
    lag_features = compute_lag_features(historical_df)
    features.update(lag_features)

    # 4. MOVING AVERAGE FEATURES (12 features: ma_3, ma_6, ma_12 and their std/min/max)
    ma_features = compute_ma_features(historical_df)
    features.update(ma_features)

    # 5. MOMENTUM FEATURES (3 features: momentum_1, momentum_3, momentum_12)
    momentum_features = compute_momentum_features(historical_df)
    features.update(momentum_features)

    # 6. VOLATILITY FEATURES (2 features: volatility_3, volatility_12)
    volatility_features = compute_volatility_features(historical_df)
    features.update(volatility_features)

    # 7. YoY GROWTH FEATURES (3 features: yoy_growth, yoy_lag_1, yoy_lag_2)
    yoy_features = compute_yoy_features(historical_df)
    features.update(yoy_features)

    # 8. ECONOMIC INDICATOR FEATURES (50+ features)
    economic_features = external_loader.get_economic_feature_history(prediction_date)
    features.update(economic_features)

    # 9. CONSUMER SPENDING FEATURES (27 features)
    consumer_spending_features = external_loader.get_consumer_spending_features(prediction_date)
    features.update(consumer_spending_features)

    # 10. STOCK MARKET FEATURES (64 features: 8 tickers × 8 features)
    stock_features = external_loader.get_stock_features(
        prediction_date,
        tickers=['AAPL', 'AMZN', 'WMT', 'COST', 'SPY', 'QQQ', 'XLY', 'XRT']
    )
    features.update(stock_features)

    # 11. ADDITIONAL FEATURES
    features['unique_id'] = 0
    features['global_trend'] = len(historical_df)
    features['economic_uncertainty'] = 0.5
    features['seasonal_component'] = temporal_features.get('month_sin', 0) * 10
    features['trend_strength'] = 1.0

    # Convert to DataFrame
    features_df = pd.DataFrame([features])

    logger.info(f"✓ Generated {len(features_df.columns)} features")

    return features_df


def compute_target_features(historical_df: pd.DataFrame, prediction_date: str) -> Dict[str, float]:
    """Compute target variable features (y_lag_1, y_lag_3, y_lag_6, y_lag_12, y_pct_change)"""
    features = {}

    if 'value' not in historical_df.columns:
        # Use default values
        features['y_lag_1'] = 1000.0
        features['y_lag_3'] = 1000.0
        features['y_lag_6'] = 1000.0
        features['y_lag_12'] = 1000.0
        features['y_pct_change'] = 0.01
        return features

    series = historical_df['value'].dropna()

    if len(series) >= 1:
        features['y_lag_1'] = float(series.iloc[-1])
    else:
        features['y_lag_1'] = 1000.0

    if len(series) >= 3:
        features['y_lag_3'] = float(series.iloc[-3])
    else:
        features['y_lag_3'] = features['y_lag_1']

    if len(series) >= 6:
        features['y_lag_6'] = float(series.iloc[-6])
    else:
        features['y_lag_6'] = features['y_lag_1']

    if len(series) >= 12:
        features['y_lag_12'] = float(series.iloc[-12])
    else:
        features['y_lag_12'] = features['y_lag_1']

    if len(series) >= 2:
        features['y_pct_change'] = float(series.iloc[-1] / series.iloc[-2] - 1)
    else:
        features['y_pct_change'] = 0.01

    return features


def compute_temporal_features(prediction_date: str) -> Dict[str, float]:
    """Compute temporal/date features (37 features)"""
    dt = pd.to_datetime(prediction_date)
    features = {}

    # Basic temporal
    features['year'] = float(dt.year)
    features['month'] = float(dt.month)
    features['quarter'] = float((dt.month - 1) // 3 + 1)
    features['day_of_month'] = float(dt.day)
    features['day_of_year'] = float(dt.dayofyear)
    features['week_of_year'] = float(dt.isocalendar()[1])
    features['days_in_month'] = float(dt.days_in_month)

    # Cyclical encodings
    features['month_sin'] = np.sin(2 * np.pi * dt.month / 12)
    features['month_cos'] = np.cos(2 * np.pi * dt.month / 12)
    features['quarter_sin'] = np.sin(2 * np.pi * ((dt.month - 1) // 3 + 1) / 4)
    features['quarter_cos'] = np.cos(2 * np.pi * ((dt.month - 1) // 3 + 1) / 4)

    # Fourier terms
    for order in [1, 2, 3]:
        features[f'sin_yearly_{order}'] = np.sin(2 * np.pi * order * dt.dayofyear / 365.25)
        features[f'cos_yearly_{order}'] = np.cos(2 * np.pi * order * dt.dayofyear / 365.25)

    for order in [1, 2]:
        quarter = (dt.month - 1) // 3 + 1
        features[f'sin_quarterly_{order}'] = np.sin(2 * np.pi * order * quarter / 4)
        features[f'cos_quarterly_{order}'] = np.cos(2 * np.pi * order * quarter / 4)

    # Month dummies
    for m in range(1, 13):
        features[f'month_{m}'] = 1.0 if dt.month == m else 0.0

    # Seasonal indicators
    features['is_spring'] = 1.0 if 3 <= dt.month <= 5 else 0.0
    features['is_summer'] = 1.0 if 6 <= dt.month <= 8 else 0.0
    features['is_fall'] = 1.0 if 9 <= dt.month <= 11 else 0.0
    features['is_winter'] = 1.0 if dt.month == 12 or dt.month <= 2 else 0.0

    # Holiday/special period indicators
    features['is_holiday_season'] = 1.0 if dt.month in [11, 12] else 0.0
    features['is_black_friday_month'] = 1.0 if dt.month == 11 else 0.0
    features['is_christmas_month'] = 1.0 if dt.month == 12 else 0.0
    features['is_december'] = 1.0 if dt.month == 12 else 0.0
    features['is_back_to_school'] = 1.0 if dt.month in [7, 8] else 0.0
    features['is_summer_peak'] = 1.0 if dt.month in [6, 7, 8] else 0.0
    features['is_quarter_end'] = 1.0 if dt.month in [3, 6, 9, 12] else 0.0
    features['is_january'] = 1.0 if dt.month == 1 else 0.0
    features['is_year_end'] = 1.0 if dt.month == 12 else 0.0
    features['is_new_year'] = 1.0 if dt.month == 1 else 0.0
    features['is_payday_period'] = 1.0 if dt.day in [1, 15, 30, 31] else 0.0

    # Progress features
    features['year_progress'] = float(dt.dayofyear / 365.25)
    features['month_progress'] = float(dt.day / dt.days_in_month)
    features['quarter_progress'] = float(((dt.month - 1) % 3 + dt.day / dt.days_in_month) / 3)

    return features


def compute_lag_features(historical_df: pd.DataFrame) -> Dict[str, float]:
    """Compute lag features (lag_1, lag_2, lag_3, lag_6, lag_12)"""
    features = {}

    if 'value' not in historical_df.columns:
        return {
            'lag_1': 1000.0,
            'lag_2': 1000.0,
            'lag_3': 1000.0,
            'lag_6': 1000.0,
            'lag_12': 1000.0
        }

    series = historical_df['value'].dropna()

    for lag in [1, 2, 3, 6, 12]:
        if len(series) > lag:
            features[f'lag_{lag}'] = float(series.iloc[-lag])
        else:
            features[f'lag_{lag}'] = float(series.iloc[-1]) if len(series) > 0 else 1000.0

    return features


def compute_ma_features(historical_df: pd.DataFrame) -> Dict[str, float]:
    """Compute moving average features (ma_3, ma_6, ma_12 and std/min/max)"""
    features = {}

    if 'value' not in historical_df.columns:
        defaults = {
            'ma_3': 1000.0, 'ma_3_std': 50.0, 'ma_3_min': 900.0, 'ma_3_max': 1100.0,
            'ma_6': 1000.0, 'ma_6_std': 50.0, 'ma_6_min': 900.0, 'ma_6_max': 1100.0,
            'ma_12': 1000.0, 'ma_12_std': 50.0, 'ma_12_min': 900.0, 'ma_12_max': 1100.0
        }
        return defaults

    series = historical_df['value'].dropna()

    for window in [3, 6, 12]:
        if len(series) >= window:
            rolling = series.tail(window)
            features[f'ma_{window}'] = float(rolling.mean())
            features[f'ma_{window}_std'] = float(rolling.std())
            features[f'ma_{window}_min'] = float(rolling.min())
            features[f'ma_{window}_max'] = float(rolling.max())
        else:
            base_value = float(series.iloc[-1]) if len(series) > 0 else 1000.0
            features[f'ma_{window}'] = base_value
            features[f'ma_{window}_std'] = base_value * 0.05
            features[f'ma_{window}_min'] = base_value * 0.95
            features[f'ma_{window}_max'] = base_value * 1.05

    return features


def compute_momentum_features(historical_df: pd.DataFrame) -> Dict[str, float]:
    """Compute momentum features"""
    features = {}

    if 'value' not in historical_df.columns:
        return {'momentum_1': 0.0, 'momentum_3': 0.0, 'momentum_12': 0.0}

    series = historical_df['value'].dropna()

    if len(series) >= 2:
        features['momentum_1'] = float(series.iloc[-1] - series.iloc[-2])
    else:
        features['momentum_1'] = 0.0

    if len(series) >= 4:
        features['momentum_3'] = float(series.iloc[-1] - series.iloc[-4])
    else:
        features['momentum_3'] = features['momentum_1'] * 3

    if len(series) >= 13:
        features['momentum_12'] = float(series.iloc[-1] - series.iloc[-13])
    else:
        features['momentum_12'] = features['momentum_1'] * 12

    return features


def compute_volatility_features(historical_df: pd.DataFrame) -> Dict[str, float]:
    """Compute volatility features"""
    features = {}

    if 'value' not in historical_df.columns:
        return {'volatility_3': 50.0, 'volatility_12': 50.0}

    series = historical_df['value'].dropna()

    if len(series) >= 4:
        returns = series.tail(4).pct_change().dropna()
        features['volatility_3'] = float(returns.std()) if len(returns) > 0 else 50.0
    else:
        features['volatility_3'] = 50.0

    if len(series) >= 13:
        returns = series.tail(13).pct_change().dropna()
        features['volatility_12'] = float(returns.std()) if len(returns) > 0 else 50.0
    else:
        features['volatility_12'] = features['volatility_3']

    return features


def compute_yoy_features(historical_df: pd.DataFrame) -> Dict[str, float]:
    """Compute year-over-year growth features"""
    features = {}

    if 'value' not in historical_df.columns:
        return {'yoy_growth': 0.05, 'yoy_lag_1': 0.05, 'yoy_lag_2': 0.05}

    series = historical_df['value'].dropna()

    if len(series) >= 13:
        features['yoy_growth'] = float(series.iloc[-1] / series.iloc[-13] - 1)
    else:
        features['yoy_growth'] = 0.05

    if len(series) >= 14:
        features['yoy_lag_1'] = float(series.iloc[-2] / series.iloc[-14] - 1)
    else:
        features['yoy_lag_1'] = features['yoy_growth']

    if len(series) >= 15:
        features['yoy_lag_2'] = float(series.iloc[-3] / series.iloc[-15] - 1)
    else:
        features['yoy_lag_2'] = features['yoy_lag_1']

    return features


if __name__ == "__main__":
    # Test the feature computer
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Testing Full Feature Computer (242 features)")
    print("=" * 60)

    # Create synthetic historical data
    dates = pd.date_range(end=datetime.now(), periods=400, freq='D')
    historical_df = pd.DataFrame({
        'date': dates,
        'value': np.random.normal(10000, 500, 400) + np.arange(400) * 10
    })

    # Compute features
    features_df = compute_full_features(historical_df, "2024-12-01", "total_sales")

    print(f"\nGenerated {len(features_df.columns)} features")
    print(f"Feature columns: {list(features_df.columns)[:20]}")
    print(f"... and {len(features_df.columns) - 20} more")

    # Check for missing expected features
    expected_features = [
        'AAPL_monthly_return', 'AMZN_avg_volume', 'cpi', 'interest_rates',
        'unemployment', 'consumer_sentiment', 'month_sin', 'y_lag_1'
    ]

    print("\nChecking expected features:")
    for feat in expected_features:
        present = feat in features_df.columns
        print(f"  {feat}: {'✓' if present else '✗'}")
        if present:
            print(f"    Value: {features_df[feat].iloc[0]:.4f}")

    print("\n" + "=" * 60)
