
"""
Yahoo Finance Data Fetcher

This module fetches financial market data from Yahoo Finance API
for time-series forecasting models.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import load_config

class YahooFinanceFetcher:
    def __init__(self):
        self.config = load_config()
        # Default tickers for retail time-series forecasting
        self.default_tickers = ['AAPL', 'AMZN', 'WMT', 'COST']

    def fetch_ticker_data(self, symbol: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetch historical ticker data for given symbol and period.

        Args:
            symbol: Stock ticker symbol
            period: Period to fetch (default: "5y" for 5 years)
            interval: Data interval (default: "1d" for daily)

        Returns:
            DataFrame with OHLCV data and technical features
        """
        try:
            # Fetch data using yfinance
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)

            if data.empty:
                print(f"No data found for ticker {symbol}")
                return pd.DataFrame()

            # Reset index to make Date a column
            data.reset_index(inplace=True)

            # Add symbol column
            data['symbol'] = symbol

            # Convert Date column to datetime (it's already datetime from yfinance)
            data['Date'] = pd.to_datetime(data['Date'], utc=True)

            # Data cleaning: remove any rows with missing critical price data
            initial_rows = len(data)
            data = data.dropna(subset=['Close', 'High', 'Low', 'Open'])
            cleaned_rows = len(data)
            if cleaned_rows < initial_rows:
                print(f"Removed {initial_rows - cleaned_rows} rows with missing price data for {symbol}")

            # Add TimeCopilot-Finance-Large required features
            data = self.add_financial_features(data, symbol)

            print(f"Successfully fetched {len(data)} days of data for {symbol}")
            return data

        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_multiple_tickers(self, symbols: list = None, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetch data for multiple tickers simultaneously.

        Args:
            symbols: List of ticker symbols
            period: Period to fetch (default: "5y")
            interval: Data interval (default: "1d")

        Returns:
            Combined DataFrame with all tickers
        """
        if symbols is None:
            symbols = self.default_tickers

        all_data = []

        for symbol in symbols:
            print(f"Fetching data for {symbol}...")
            ticker_data = self.fetch_ticker_data(symbol, period, interval)
            if not ticker_data.empty:
                all_data.append(ticker_data)

        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            print(f"Successfully fetched data for {len(symbols)} tickers")
            return combined_data
        else:
            print("No data fetched for any tickers")
            return pd.DataFrame()

    def fetch_market_indices(self) -> pd.DataFrame:
        """
        Fetch major market indices (S&P 500, NASDAQ, etc.).

        Returns:
            DataFrame with major index data
        """
        indices = ['^GSPC', '^IXIC', '^DJI', '^VIX']  # S&P 500, NASDAQ, Dow Jones, VIX
        return self.fetch_multiple_tickers(indices, period="5y", interval="1d")

    def fetch_commodities(self) -> pd.DataFrame:
        """
        Fetch commodity prices (gold, oil, etc.).

        Returns:
            DataFrame with commodity data
        """
        commodities = ['GC=F', 'CL=F', 'SI=F']  # Gold, Oil, Silver futures
        return self.fetch_multiple_tickers(commodities, period="5y", interval="1d")

    def add_financial_features(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Add financial features required for TimeCopilot-Finance-Large.

        Args:
            data: DataFrame with OHLCV data
            symbol: Ticker symbol for the data

        Returns:
            DataFrame with added financial features
        """
        # Calculate daily returns (percentage change)
        data['daily_return'] = data['Close'].pct_change()

        # Calculate log returns
        data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))

        # Calculate 20-day rolling volatility (standard deviation of daily returns)
        data['rolling_vol_20d'] = data['daily_return'].rolling(window=20, min_periods=1).std()

        # Calculate 10-day rolling mean of Close price
        data['rolling_mean_10d'] = data['Close'].rolling(window=10, min_periods=1).mean()

        # Additional useful features for TimeCopilot-Finance-Large
        # 5-day rolling mean
        data['rolling_mean_5d'] = data['Close'].rolling(window=5, min_periods=1).mean()

        # 20-day rolling mean
        data['rolling_mean_20d'] = data['Close'].rolling(window=20, min_periods=1).mean()

        # 50-day rolling mean
        data['rolling_mean_50d'] = data['Close'].rolling(window=50, min_periods=1).mean()

        # 10-day rolling volatility
        data['rolling_vol_10d'] = data['daily_return'].rolling(window=10, min_periods=1).std()

        # 5-day rolling volatility
        data['rolling_vol_5d'] = data['daily_return'].rolling(window=5, min_periods=1).std()

        # High-Low spread
        data['high_low_spread'] = (data['High'] - data['Low']) / data['Close']

        # Open-Close spread
        data['open_close_spread'] = (data['Close'] - data['Open']) / data['Open']

        # Volume change
        data['volume_change'] = data['Volume'].pct_change()

        # RSI (Relative Strength Index) - 14 period
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss
        data['rsi_14'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['macd'] = exp1 - exp2
        data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']

        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        data['bb_middle'] = data['Close'].rolling(window=bb_period, min_periods=1).mean()
        data['bb_upper'] = data['bb_middle'] + (data['Close'].rolling(window=bb_period, min_periods=1).std() * bb_std)
        data['bb_lower'] = data['bb_middle'] - (data['Close'].rolling(window=bb_period, min_periods=1).std() * bb_std)
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        # Handle division by zero in bb_position
        data['bb_position'] = (data['Close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower']).replace(0, np.nan)

        # 52-week high and low
        data['high_52w'] = data['High'].rolling(window=252, min_periods=1).max()
        data['low_52w'] = data['Low'].rolling(window=252, min_periods=1).min()
        # Handle division by zero in position_52w
        data['position_52w'] = (data['Close'] - data['low_52w']) / (data['high_52w'] - data['low_52w']).replace(0, np.nan)

        # Price momentum (various periods)
        for period in [1, 3, 5, 10, 20]:
            data[f'momentum_{period}d'] = data['Close'].pct_change(period)

        # Add metadata columns
        data['fetch_timestamp'] = datetime.now()
        data['data_source'] = 'Yahoo Finance'

        return data

    def save_to_csv(self, data: pd.DataFrame, filename: str) -> None:
        """
        Save fetched data to CSV file in data_raw directory.

        Args:
            data: DataFrame to save
            filename: Output filename
        """
        try:
            # Create data_raw directory if it doesn't exist
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data_raw')
            os.makedirs(data_dir, exist_ok=True)

            # Full path for output file
            output_path = os.path.join(data_dir, filename)

            # Save to CSV
            data.to_csv(output_path, index=False)
            print(f"Data saved to {output_path}")
            print(f"Saved {len(data)} rows with {len(data.columns)} columns")

            # Print basic statistics
            if not data.empty:
                print("\nData Summary:")
                print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
                print(f"Unique symbols: {data['symbol'].nunique()}")
                print(f"Columns: {list(data.columns)}")

        except Exception as e:
            print(f"Error saving data to CSV: {e}")

    def fetch_and_save_retail_stocks(self) -> str:
        """
        Fetch data for retail-related stocks and save to CSV.

        Returns:
            Path to saved CSV file
        """
        print("Fetching data for retail stocks: AAPL, AMZN, WMT, COST")

        # Fetch data for default retail tickers
        data = self.fetch_multiple_tickers(
            symbols=self.default_tickers,
            period="5y",
            interval="1d"
        )

        if not data.empty:
            # Save to CSV
            output_file = "yahoo_daily.csv"
            self.save_to_csv(data, output_file)

            # Return full path
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data_raw')
            return os.path.join(data_dir, output_file)
        else:
            print("No data fetched, nothing to save")
            return ""

    def validate_data_quality(self, data: pd.DataFrame) -> dict:
        """
        Validate data quality and generate quality report.

        Args:
            data: DataFrame to validate

        Returns:
            Dictionary with quality metrics
        """
        if data.empty:
            return {"error": "Empty DataFrame"}

        # Convert date column to datetime for proper calculations
        data['Date'] = pd.to_datetime(data['Date'], utc=True)

        quality_report = {
            "total_rows": len(data),
            "total_columns": len(data.columns),
            "date_range": {
                "start": data['Date'].min(),
                "end": data['Date'].max(),
                "days": (data['Date'].max() - data['Date'].min()).days
            },
            "symbols": data['symbol'].unique().tolist(),
            "missing_values": data.isnull().sum().to_dict(),
            "data_types": data.dtypes.to_dict(),
            "duplicates": data.duplicated().sum(),
            "price_stats": {}
        }

        # Price statistics for each symbol
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol]
            if not symbol_data.empty and 'Close' in symbol_data.columns:
                quality_report["price_stats"][symbol] = {
                    "min_close": symbol_data['Close'].min(),
                    "max_close": symbol_data['Close'].max(),
                    "mean_close": symbol_data['Close'].mean(),
                    "latest_close": symbol_data['Close'].iloc[-1]
                }

        return quality_report

def main():
    """Main execution function for Yahoo Finance data fetching"""
    try:
        # Initialize fetcher
        fetcher = YahooFinanceFetcher()

        # Fetch data for retail stocks
        output_path = fetcher.fetch_and_save_retail_stocks()

        if output_path:
            print(f"\n Successfully completed Yahoo Finance data fetch")
            print(f" Data saved to: {output_path}")

            # Load and validate the saved data
            df = pd.read_csv(output_path)
            quality_report = fetcher.validate_data_quality(df)

            print(f"\n Data Quality Report:")
            print(f"   Total rows: {quality_report['total_rows']:,}")
            print(f"   Total columns: {quality_report['total_columns']}")
            print(f"   Date range: {quality_report['date_range']['start']} to {quality_report['date_range']['end']}")
            print(f"   Symbols: {', '.join(quality_report['symbols'])}")
            print(f"   Duplicates: {quality_report['duplicates']}")

            # Check for missing values in critical columns (allowing some expected missing values)
            critical_cols = ['Date', 'symbol', 'Close', 'daily_return', 'rolling_vol_20d', 'rolling_mean_10d']
            missing_critical = {col: quality_report['missing_values'][col] for col in critical_cols if col in quality_report['missing_values'] and quality_report['missing_values'][col] > 0}

            if missing_critical:
                print(f"\n Missing values (expected for rolling indicators): {missing_critical}")

                # Check if missing values are within acceptable limits
                acceptable_missing = {
                    'Date': 0, 'symbol': 0, 'Close': 0,  # These should have no missing values
                    'daily_return': len(quality_report['symbols']),  # Expected: 1 per symbol (first day)
                    'rolling_mean_10d': len(quality_report['symbols']) * 9,  # Expected: 9 per symbol (first 9 days)
                    'rolling_vol_20d': len(quality_report['symbols']) * 19  # Expected: 19 per symbol (first 19 days)
                }

                all_acceptable = True
                for col, missing_count in missing_critical.items():
                    if col in acceptable_missing and missing_count > acceptable_missing[col]:
                        print(f" {col}: {missing_count} missing (expected ≤ {acceptable_missing[col]})")
                        all_acceptable = False
                    else:
                        print(f" {col}: {missing_count} missing (within acceptable limits)")

                if all_acceptable:
                    print(f" All missing values within expected limits for rolling calculations")
                else:
                    print(f"  Some missing values exceed expected limits")
            else:
                print(f" No missing values in critical columns")

        else:
            print(" Failed to fetch Yahoo Finance data")
            return 1

        return 0

    except Exception as e:
        print(f" Error in main execution: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
