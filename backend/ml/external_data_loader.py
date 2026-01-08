"""
External Data Loader Module

Loads economic indicators (FRED) and stock market data (Yahoo Finance)
for retail sales forecasting models.

This module loads pre-fetched data from project_root/data_raw/ directory.
To update the data, run the ETL scripts in project_root/etl/
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent / "project_root"
DATA_RAW_DIR = PROJECT_ROOT / "data_raw"


class ExternalDataLoader:
    """Load external economic and stock market data"""

    def __init__(self):
        self.fred_data = None
        self.yahoo_data = None
        self.consumer_spending_data = None

    def load_fred_data(self) -> pd.DataFrame:
        """
        Load FRED economic indicators data

        Returns:
            DataFrame with columns: date, cpi, interest_rates, unemployment,
                                  consumer_sentiment, money_supply, industrial_production
        """
        if self.fred_data is not None:
            return self.fred_data

        fred_file = DATA_RAW_DIR / "fred_monthly.csv"

        if not fred_file.exists():
            logger.warning(f"FRED data file not found: {fred_file}")
            logger.info("To fetch FRED data, run: python project_root/etl/fetch_fred.py")
            # Create empty DataFrame with correct columns
            self.fred_data = pd.DataFrame(columns=[
                'date', 'cpi', 'interest_rates', 'unemployment',
                'consumer_sentiment', 'money_supply', 'industrial_production'
            ])
            return self.fred_data

        try:
            self.fred_data = pd.read_csv(fred_file)
            self.fred_data['date'] = pd.to_datetime(self.fred_data['date'])
            self.fred_data = self.fred_data.sort_values('date').reset_index(drop=True)

            logger.info(f"✓ Loaded FRED data: {len(self.fred_data)} rows from {fred_file}")
            return self.fred_data

        except Exception as e:
            logger.error(f"Error loading FRED data: {e}")
            self.fred_data = pd.DataFrame(columns=[
                'date', 'cpi', 'interest_rates', 'unemployment',
                'consumer_sentiment', 'money_supply', 'industrial_production'
            ])
            return self.fred_data

    def load_yahoo_data(self) -> pd.DataFrame:
        """
        Load Yahoo Finance stock data

        Returns:
            DataFrame with columns: Date, symbol, Close, daily_return,
                                  rolling_vol_20d, Volume, monthly_return,
                                  monthly_volatility, monthly_avg_volume
        """
        if self.yahoo_data is not None:
            return self.yahoo_data

        yahoo_file = DATA_RAW_DIR / "yahoo_daily.csv"

        if not yahoo_file.exists():
            logger.warning(f"Yahoo Finance data file not found: {yahoo_file}")
            logger.info("To fetch Yahoo data, run: python project_root/etl/fetch_yahoo.py")
            # Create empty DataFrame with correct columns
            self.yahoo_data = pd.DataFrame(columns=[
                'Date', 'symbol', 'Close', 'daily_return',
                'rolling_vol_20d', 'Volume', 'monthly_return',
                'monthly_volatility', 'monthly_avg_volume'
            ])
            return self.yahoo_data

        try:
            self.yahoo_data = pd.read_csv(yahoo_file)
            self.yahoo_data['Date'] = pd.to_datetime(self.yahoo_data['Date'], utc=True)
            self.yahoo_data = self.yahoo_data.sort_values('Date').reset_index(drop=True)

            logger.info(f"✓ Loaded Yahoo Finance data: {len(self.yahoo_data)} rows from {yahoo_file}")
            return self.yahoo_data

        except Exception as e:
            logger.error(f"Error loading Yahoo Finance data: {e}")
            self.yahoo_data = pd.DataFrame(columns=[
                'Date', 'symbol', 'Close', 'daily_return',
                'rolling_vol_20d', 'Volume', 'monthly_return',
                'monthly_volatility', 'monthly_avg_volume'
            ])
            return self.yahoo_data

    def get_latest_economic_indicators(self, as_of_date: str) -> dict:
        """
        Get latest economic indicators as of a given date

        Args:
            as_of_date: Date string (YYYY-MM-DD)

        Returns:
            Dictionary with economic indicator values
        """
        fred_df = self.load_fred_data()

        if fred_df.empty:
            # Return default values
            return {
                'cpi': 300.0,
                'interest_rates': 5.0,
                'unemployment': 4.0,
                'consumer_sentiment': 70.0,
                'money_supply': 20000.0,
                'industrial_production': 105.0
            }

        as_of_dt = pd.to_datetime(as_of_date)

        # Get most recent data point on or before as_of_date
        available_data = fred_df[fred_df['date'] <= as_of_dt]

        if available_data.empty:
            # Use first available date
            latest = fred_df.iloc[0]
        else:
            latest = available_data.iloc[-1]

        return {
            'cpi': float(latest['cpi']) if pd.notna(latest['cpi']) else 300.0,
            'interest_rates': float(latest['interest_rates']) if pd.notna(latest['interest_rates']) else 5.0,
            'unemployment': float(latest['unemployment']) if pd.notna(latest['unemployment']) else 4.0,
            'consumer_sentiment': float(latest['consumer_sentiment']) if pd.notna(latest['consumer_sentiment']) else 70.0,
            'money_supply': float(latest['money_supply']) if pd.notna(latest['money_supply']) else 20000.0,
            'industrial_production': float(latest['industrial_production']) if pd.notna(latest['industrial_production']) else 105.0
        }

    def get_stock_features(self, as_of_date: str, tickers: list = None) -> dict:
        """
        Get stock market features for given tickers as of a date

        Args:
            as_of_date: Date string (YYYY-MM-DD)
            tickers: List of ticker symbols (default: all 8 tickers)

        Returns:
            Dictionary with stock features for each ticker
        """
        if tickers is None:
            tickers = ['AAPL', 'AMZN', 'WMT', 'COST', 'SPY', 'QQQ', 'XLY', 'XRT']

        yahoo_df = self.load_yahoo_data()

        features = {}

        for ticker in tickers:
            # Default values
            features[f'{ticker}_monthly_return'] = 0.01
            features[f'{ticker}_monthly_volatility'] = 0.02
            features[f'{ticker}_avg_volume'] = 50000000.0
            features[f'{ticker}_price_momentum'] = 0.0
            features[f'{ticker}_monthly_return_lag_1'] = 0.01
            features[f'{ticker}_monthly_return_lag_3'] = 0.01
            features[f'{ticker}_monthly_volatility_lag_1'] = 0.02
            features[f'{ticker}_monthly_volatility_lag_3'] = 0.02

            if yahoo_df.empty:
                continue

            as_of_dt = pd.to_datetime(as_of_date)

            # Get data for this ticker
            ticker_data = yahoo_df[yahoo_df['symbol'] == ticker].copy()
            ticker_data = ticker_data[ticker_data['Date'] <= as_of_dt]

            if ticker_data.empty:
                continue

            # Get most recent data point
            latest = ticker_data.iloc[-1]

            # Extract features
            if 'monthly_return' in latest and pd.notna(latest['monthly_return']):
                features[f'{ticker}_monthly_return'] = float(latest['monthly_return'])
            if 'monthly_volatility' in latest and pd.notna(latest['monthly_volatility']):
                features[f'{ticker}_monthly_volatility'] = float(latest['monthly_volatility'])
            if 'monthly_avg_volume' in latest and pd.notna(latest['monthly_avg_volume']):
                features[f'{ticker}_avg_volume'] = float(latest['monthly_avg_volume'])

            # Calculate price momentum (recent return)
            if 'Close' in latest and pd.notna(latest['Close']):
                if len(ticker_data) >= 20:
                    close_20d_ago = ticker_data.iloc[-20]['Close']
                    if pd.notna(close_20d_ago) and close_20d_ago > 0:
                        features[f'{ticker}_price_momentum'] = float(latest['Close'] / close_20d_ago - 1)

            # Lagged features
            if len(ticker_data) >= 2:
                lag1 = ticker_data.iloc[-2]
                if 'monthly_return' in lag1 and pd.notna(lag1['monthly_return']):
                    features[f'{ticker}_monthly_return_lag_1'] = float(lag1['monthly_return'])
                if 'monthly_volatility' in lag1 and pd.notna(lag1['monthly_volatility']):
                    features[f'{ticker}_monthly_volatility_lag_1'] = float(lag1['monthly_volatility'])

            if len(ticker_data) >= 4:
                lag3 = ticker_data.iloc[-4]
                if 'monthly_return' in lag3 and pd.notna(lag3['monthly_return']):
                    features[f'{ticker}_monthly_return_lag_3'] = float(lag3['monthly_return'])
                if 'monthly_volatility' in lag3 and pd.notna(lag3['monthly_volatility']):
                    features[f'{ticker}_monthly_volatility_lag_3'] = float(lag3['monthly_volatility'])

        return features

    def get_economic_feature_history(self, as_of_date: str) -> dict:
        """
        Get economic indicator history with lags, MA, std, pct_change

        Args:
            as_of_date: Date string (YYYY-MM-DD)

        Returns:
            Dictionary with all economic features including lags, MA, std, pct_change
        """
        fred_df = self.load_fred_data()

        if fred_df.empty:
            # Return default values
            features = {}
            for indicator in ['cpi', 'interest_rates', 'unemployment',
                             'consumer_sentiment', 'money_supply', 'industrial_production']:
                features[indicator] = 300.0 if indicator == 'cpi' else (
                    5.0 if indicator == 'interest_rates' else (
                        4.0 if indicator == 'unemployment' else (
                            70.0 if indicator == 'consumer_sentiment' else (
                                20000.0 if indicator == 'money_supply' else 105.0
                            )
                        )
                    )
                )
                # Add lags
                for lag in [1, 3, 6, 12]:
                    features[f'{indicator}_lag_{lag}'] = features[indicator]
                # Add MA
                for window in [3, 6, 12]:
                    features[f'{indicator}_ma_{window}'] = features[indicator]
                # Add std
                for window in [3, 6, 12]:
                    features[f'{indicator}_std_{window}'] = features[indicator] * 0.05
                # Add pct_change
                features[f'{indicator}_pct_change'] = 0.01
                features[f'{indicator}_pct_change_yoy'] = 0.03

            return features

        as_of_dt = pd.to_datetime(as_of_date)
        historical_data = fred_df[fred_df['date'] <= as_of_dt].copy()

        if historical_data.empty:
            historical_data = fred_df.head(1)

        features = {}

        for indicator in ['cpi', 'interest_rates', 'unemployment',
                         'consumer_sentiment', 'money_supply', 'industrial_production']:

            if indicator not in historical_data.columns:
                continue

            series = historical_data[indicator].dropna()

            if series.empty:
                continue

            latest_value = float(series.iloc[-1])
            features[indicator] = latest_value

            # Lags
            for lag in [1, 3, 6, 12]:
                if len(series) > lag:
                    features[f'{indicator}_lag_{lag}'] = float(series.iloc[-lag-1])
                else:
                    features[f'{indicator}_lag_{lag}'] = latest_value

            # Moving averages
            for window in [3, 6, 12]:
                if len(series) >= window:
                    features[f'{indicator}_ma_{window}'] = float(series.tail(window).mean())
                else:
                    features[f'{indicator}_ma_{window}'] = latest_value

            # Standard deviation
            for window in [3, 6, 12]:
                if len(series) >= window:
                    features[f'{indicator}_std_{window}'] = float(series.tail(window).std())
                else:
                    features[f'{indicator}_std_{window}'] = latest_value * 0.05

            # Percentage changes
            if len(series) >= 2:
                features[f'{indicator}_pct_change'] = float(series.iloc[-1] / series.iloc[-2] - 1)
            else:
                features[f'{indicator}_pct_change'] = 0.01

            if len(series) >= 13:
                features[f'{indicator}_pct_change_yoy'] = float(series.iloc[-1] / series.iloc[-13] - 1)
            else:
                features[f'{indicator}_pct_change_yoy'] = 0.03

        return features

    def get_consumer_spending_features(self, as_of_date: str) -> dict:
        """
        Get consumer spending features

        Args:
            as_of_date: Date string (YYYY-MM-DD)

        Returns:
            Dictionary with consumer spending features
        """
        # Consumer spending is derived from other indicators
        economic = self.get_economic_feature_history(as_of_date)

        features = {}

        # Base consumer spending (approximated from CPI and industrial production)
        base_spending = economic.get('cpi', 300.0) * economic.get('industrial_production', 105.0) / 300.0

        features['consumer_spending'] = base_spending * 100

        # Add lags, MA, std, pct_change (similar to other economic indicators)
        for lag in [1, 3, 6, 12]:
            features[f'consumer_spending_lag_{lag}'] = base_spending * 100

        for window in [3, 6, 12]:
            features[f'consumer_spending_ma_{window}'] = base_spending * 100
            features[f'consumer_spending_std_{window}'] = base_spending * 5

        features['consumer_spending_pct_change'] = 0.02
        features['consumer_spending_pct_change_yoy'] = 0.05
        features['consumer_spending_trend'] = 1.0

        return features


# Global singleton instance
_loader_instance = None


def get_external_data_loader() -> ExternalDataLoader:
    """Get global external data loader instance"""
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = ExternalDataLoader()
    return _loader_instance


if __name__ == "__main__":
    # Test the loader
    logging.basicConfig(level=logging.INFO)

    loader = ExternalDataLoader()

    print("=" * 60)
    print("Testing External Data Loader")
    print("=" * 60)

    # Test FRED data
    print("\nLoading FRED data...")
    fred = loader.load_fred_data()
    print(f"FRED rows: {len(fred)}")
    if not fred.empty:
        print(f"Date range: {fred['date'].min()} to {fred['date'].max()}")
        print(f"Columns: {list(fred.columns)}")

    # Test Yahoo data
    print("\nLoading Yahoo Finance data...")
    yahoo = loader.load_yahoo_data()
    print(f"Yahoo rows: {len(yahoo)}")
    if not yahoo.empty:
        print(f"Date range: {yahoo['Date'].min()} to {yahoo['Date'].max()}")
        print(f"Symbols: {yahoo['symbol'].unique().tolist()}")

    # Test getting latest indicators
    print("\nGetting latest economic indicators...")
    economic = loader.get_latest_economic_indicators("2024-12-01")
    for key, value in economic.items():
        print(f"  {key}: {value:.2f}")

    # Test getting stock features
    print("\nGetting stock features...")
    stock_features = loader.get_stock_features("2024-12-01", ['AAPL', 'AMZN'])
    for key, value in list(stock_features.items())[:8]:
        print(f"  {key}: {value:.4f}")

    print("\n" + "=" * 60)
