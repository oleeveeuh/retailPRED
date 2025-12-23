
"""
FRED Data Fetcher

This module fetches economic data from Federal Reserve Economic Data (FRED) API
for time-series forecasting models.
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import load_config

# Import fredapi
try:
    from fredapi import Fred
except ImportError:
    print("fredapi not installed. Install with: pip install fredapi")
    sys.exit(1)

class FREDFetcher:
    def __init__(self):
        # Load environment variables from .env file first
        self._load_env_file()

        # Then load config
        self.config = load_config()

        # Prioritize environment variable over config for API key
        self.api_key = os.getenv('FRED_API_KEY') or self.config.get('data_sources.fred.api_key')

        print(f"Using FRED API key: {self.api_key[:4] if self.api_key else 'None'}...{self.api_key[-4:] if self.api_key else 'None'} (length: {len(self.api_key) if self.api_key else 0})")

        if not self.api_key:
            raise ValueError("FRED API key not found. Set FRED_API_KEY environment variable or add to config.yaml")

        self.fred = Fred(api_key=self.api_key)

    def _load_env_file(self):
        """Load environment variables from .env file"""
        env_file = os.path.join(os.path.dirname(__file__), '..', '.env')
        if os.path.exists(env_file):
            print(f"Loading environment from: {env_file}")
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip().strip('\"')
        else:
            print("No .env file found, relying on existing environment variables")

        # Required economic series with descriptions
        self.required_series = {
            'CPIAUCSL': {
                'name': 'Consumer Price Index for All Urban Consumers',
                'description': 'Consumer Price Index (CPI) - Inflation measure',
                'frequency': 'monthly',
                'units': 'index_1982_1984=100',
                'feature_name': 'cpi'
            },
            'FEDFUNDS': {
                'name': 'Federal Funds Effective Rate',
                'description': 'Federal Funds Rate - Interest rates',
                'frequency': 'monthly',
                'units': 'percent',
                'feature_name': 'interest_rates'
            },
            'UNRATE': {
                'name': 'Unemployment Rate',
                'description': 'Civilian Unemployment Rate',
                'frequency': 'monthly',
                'units': 'percent',
                'feature_name': 'unemployment'
            },
            'UMCSENT': {
                'name': 'University of Michigan Consumer Sentiment',
                'description': 'Consumer Sentiment Index',
                'frequency': 'monthly',
                'units': 'index',
                'feature_name': 'consumer_sentiment'
            },
            'M2SL': {
                'name': 'M2 Money Supply',
                'description': 'M2 Money Supply (seasonally adjusted)',
                'frequency': 'monthly',
                'units': 'billions_of_dollars',
                'feature_name': 'money_supply'
            },
            'INDPRO': {
                'name': 'Industrial Production Index',
                'description': 'Industrial Production Index',
                'frequency': 'monthly',
                'units': 'index_2017=100',
                'feature_name': 'industrial_production'
            },
            'PCE': {
                'name': 'Personal Consumption Expenditures',
                'description': 'Personal Consumption Expenditures',
                'frequency': 'monthly',
                'units': 'billions_of_dollars',
                'feature_name': 'consumer_spending'
            }
        }

        # Additional series for enhanced economic modeling
        self.additional_series = {
            'GDP': {'name': 'Gross Domestic Product', 'frequency': 'quarterly'},
            'GDPC1': {'name': 'Real GDP', 'frequency': 'quarterly'},
            'DGS10': {'name': '10-Year Treasury Rate', 'frequency': 'daily'},
            'DEXUSEU': {'name': 'US/EUR Exchange Rate', 'frequency': 'daily'},
            'DEXUSUK': {'name': 'US/UK Exchange Rate', 'frequency': 'daily'},
            'DEXJPUS': {'name': 'USD/JPY Exchange Rate', 'frequency': 'daily'},
            'HOUST': {'name': 'Housing Starts', 'frequency': 'monthly'},
            'ISRATIO': {'name': 'Inventory-Sales Ratio', 'frequency': 'monthly'},
            'TOTALSA': {'name': 'Total Retail Sales', 'frequency': 'monthly'},
            'RSAFS': {'name': 'Retail and Food Services Sales', 'frequency': 'monthly'},
            'CORESTICKM159SFRBATL': {'name': 'Core CPI', 'frequency': 'monthly'},
            'PAYEMS': {'name': 'All Employees: Total Nonfarm Payrolls', 'frequency': 'monthly'},
            'MANEMP': {'name': 'All Employees: Manufacturing', 'frequency': 'monthly'},
            'CES0600000007': {'name': 'Average Hourly Earnings', 'frequency': 'monthly'}
        }

    def fetch_series(self, series_id: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Fetch economic series data from FRED API.

        Args:
            series_id: FRED series ID
            start_date: Start date for data (default: 5 years ago)
            end_date: End date for data (default: today)

        Returns:
            DataFrame with series data and metadata
        """
        try:
            # Set default dates if not provided
            if not start_date:
                start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')

            # Fetch series data from FRED API
            series_data = self.fred.get_series(
                series_id=series_id,
                observation_start=start_date,
                observation_end=end_date
            )

            if series_data.empty:
                print(f"No data found for series {series_id}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(series_data, columns=['value'])
            df.index.name = 'date'
            df = df.reset_index()

            # Convert date to datetime
            df['date'] = pd.to_datetime(df['date'])

            # Add series metadata
            if series_id in self.required_series:
                metadata = self.required_series[series_id]
            else:
                metadata = self.additional_series.get(series_id, {'name': series_id, 'frequency': 'unknown'})

            df['series_id'] = series_id
            df['series_name'] = metadata['name']
            df['frequency'] = metadata.get('frequency', 'unknown')
            df['units'] = metadata.get('units', 'unknown')
            df['feature_name'] = metadata.get('feature_name', series_id.lower())

            print(f"Successfully fetched {len(df)} observations for {series_id} ({metadata['name']})")
            return df

        except Exception as e:
            print(f"Error fetching data for {series_id}: {e}")
            return pd.DataFrame()

    def fetch_multiple_series(self, series_ids: list = None, start_date: str = None,
                            end_date: str = None) -> pd.DataFrame:
        """
        Fetch multiple economic series simultaneously.

        Args:
            series_ids: List of FRED series IDs
            start_date: Start date for all series
            end_date: End date for all series

        Returns:
            Combined DataFrame with all series
        """
        if series_ids is None:
            series_ids = list(self.required_series.keys())

        all_data = []

        for series_id in series_ids:
            print(f"Fetching {series_id}...")
            series_data = self.fetch_series(series_id, start_date, end_date)
            if not series_data.empty:
                all_data.append(series_data)

        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            print(f"Successfully fetched data for {len(series_ids)} series")
            return combined_data
        else:
            print("No data fetched for any series")
            return pd.DataFrame()

    def fetch_gdp_data(self) -> pd.DataFrame:
        """
        Fetch GDP and GDP growth data.

        Returns:
            DataFrame with GDP data and growth rates
        """
        gdp_series = ['GDP', 'GDPC1', 'GDPPOT', 'NYGDPMKTPCDWLD']  # GDP, Real GDP, Potential GDP, World GDP
        return self.fetch_multiple_series(gdp_series)

    def fetch_inflation_data(self) -> pd.DataFrame:
        """
        Fetch CPI, PPI, and inflation indicators.

        Returns:
            DataFrame with inflation data
        """
        inflation_series = ['CPIAUCSL', 'CORESTICKM159SFRBATL', 'PCEPI', 'CPALTT01USM657N', 'DFEDTARU']  # CPI, Core CPI, PCE, All Items CPI, Inflation Expectations
        return self.fetch_multiple_series(inflation_series)

    def fetch_employment_data(self) -> pd.DataFrame:
        """
        Fetch employment and unemployment data.

        Returns:
            DataFrame with employment data
        """
        employment_series = ['UNRATE', 'PAYEMS', 'MANEMP', 'CE16OV', 'LNS14000024', 'LNS13000036']  # Unemployment Rate, Nonfarm Payrolls, Manufacturing, Labor Force Participation
        return self.fetch_multiple_series(employment_series)

    def fetch_interest_rates(self) -> pd.DataFrame:
        """
        Fetch Fed funds rate and treasury yields.

        Returns:
            DataFrame with interest rate data
        """
        rate_series = ['FEDFUNDS', 'DGS10', 'DGS30', 'DGS2', 'DTWEXBGS', 'TEDRATE']  # Fed Funds, 10Y, 30Y, 2Y Treasuries, Dollar Index, TED Spread
        return self.fetch_multiple_series(rate_series)

    def fetch_consumer_confidence(self) -> pd.DataFrame:
        """
        Fetch consumer confidence and sentiment data.

        Returns:
            DataFrame with consumer confidence data
        """
        confidence_series = ['UMCSENT', 'CCI', 'CONFCHNG', 'UMCSENT']  # Michigan Sentiment, Conference Board Confidence
        return self.fetch_multiple_series(confidence_series)

    def add_economic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add economic features required for forecasting models.

        Args:
            data: DataFrame with basic FRED data

        Returns:
            DataFrame with added economic features
        """
        # Sort by date and series_id for proper feature calculation
        data = data.sort_values(['series_id', 'date'])

        # Calculate period-over-period changes (inflation, growth rates)
        data['value_change'] = data.groupby('series_id')['value'].pct_change()
        data['value_diff'] = data.groupby('series_id')['value'].diff()

        # Calculate log changes for economic series
        data['log_value'] = np.log(data['value'])
        data['log_change'] = data.groupby('series_id')['log_value'].diff()

        # Calculate year-over-year changes (important for economic data)
        data['value_yoy_change'] = data.groupby('series_id')['value'].pct_change(12)  # 12 months YoY
        data['log_yoy_change'] = data.groupby('series_id')['log_value'].diff(12)

        # Calculate moving averages (for trend analysis)
        for window in [3, 6, 12]:  # 3-month, 6-month, 12-month moving averages
            data[f'ma_{window}m'] = data.groupby('series_id')['value'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            data[f'ma_{window}m_std'] = data.groupby('series_id')['value'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )

        # Calculate year-over-year moving averages
        for window in [12, 24]:  # 12-month, 24-month YoY moving averages
            data[f'ma_yoy_{window//12}y'] = data.groupby('series_id')['value'].transform(
                lambda x: x.rolling(window=window, min_periods=window//2).mean()
            )

        # Calculate volatility (standard deviation of changes)
        data['volatility_3m'] = data.groupby('series_id')['value_change'].transform(
            lambda x: x.rolling(window=3, min_periods=1).std()
        )
        data['volatility_12m'] = data.groupby('series_id')['value_change'].transform(
            lambda x: x.rolling(window=12, min_periods=1).std()
        )

        # Calculate z-scores (how many standard deviations from mean)
        for window in [12, 24]:  # 12-month, 24-month z-scores
            data[f'zscore_{window//12}y'] = data.groupby('series_id')['value'].transform(
                lambda x: (x - x.rolling(window=window, min_periods=window//2).mean()) /
                         x.rolling(window=window, min_periods=window//2).std()
            )

        # Calculate percentile ranks (relative positioning)
        for window in [12, 24, 60]:  # 1-year, 2-year, 5-year percentile ranks
            data[f'pct_rank_{window//12}y'] = data.groupby('series_id')['value'].transform(
                lambda x: x.rolling(window=window, min_periods=window//2).rank(pct=True)
            )

        # Add time-based features
        data['year'] = data['date'].dt.year
        data['quarter'] = data['date'].dt.quarter
        data['month'] = data['date'].dt.month
        data['day_of_year'] = data['date'].dt.dayofyear

        # Add cyclical features for seasonality
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        data['quarter_sin'] = np.sin(2 * np.pi * data['quarter'] / 4)
        data['quarter_cos'] = np.cos(2 * np.pi * data['quarter'] / 4)

        # Add economic recession indicators
        data['is_recession'] = self._add_recession_indicators(data)

        # Add leading economic indicators features
        data['leading_indicator_signal'] = self._calculate_leading_indicators(data)

        # Add metadata
        data['fetch_timestamp'] = datetime.now()
        data['data_source'] = 'FRED'

        return data

    def _add_recession_indicators(self, data: pd.DataFrame) -> pd.Series:
        """
        Add recession indicators based on common recession definitions.

        Args:
            data: DataFrame with economic data

        Returns:
            Boolean Series indicating recession periods
        """
        # Simple recession indicator based on multiple criteria
        # This is a simplified version - in practice you'd use NBER recession dates
        recession_indicators = pd.Series(False, index=data.index)

        # Look for unemployment rate spikes
        unemployment_mask = (data['series_id'] == 'UNRATE') & (data['value_change'] > 0.01)
        if unemployment_mask.any():
            unemployment_periods = data.loc[unemployment_mask, 'date']
            for date in unemployment_periods:
                recession_indicators |= (
                    (data['date'] >= date - pd.Timedelta(days=90)) &
                    (data['date'] <= date + pd.Timedelta(days=180))
                )

        return recession_indicators

    def _calculate_leading_indicators(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate leading economic indicator signals.

        Args:
            data: DataFrame with economic data

        Returns:
            Series with leading indicator values
        """
        # Simplified leading indicator based on yield curve and consumer sentiment
        leading_signal = pd.Series(0.0, index=data.index)

        # Consumer sentiment as leading indicator
        sentiment_mask = data['series_id'] == 'UMCSENT'
        if sentiment_mask.any():
            sentiment_values = data.loc[sentiment_mask, 'value'].rank(pct=True)
            leading_signal.loc[sentiment_mask] = sentiment_values

        return leading_signal

    def create_wide_format_dataset(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert long format data to wide format for modeling.

        Args:
            data: Long format DataFrame

        Returns:
            Wide format DataFrame with series as columns
        """
        # Create a copy to avoid SettingWithCopyWarning
        data_copy = data.copy()

        # Create wide format by pivoting (exclude series_id column from values)
        pivot_data = data_copy.pivot_table(
            index='date',
            columns='feature_name',
            values='value',
            aggfunc='first'
        ).reset_index()

        # Add technical features for each economic series
        for series_id in data_copy['series_id'].unique():
            series_mask = data_copy['series_id'] == series_id
            feature_name = data_copy.loc[series_mask, 'feature_name'].iloc[0]

            if feature_name in pivot_data.columns:
                # Add lagged values
                for lag in [1, 3, 6, 12]:
                    pivot_data[f'{feature_name}_lag_{lag}m'] = pivot_data[feature_name].shift(lag)

                # Add moving averages
                for window in [3, 6, 12]:
                    pivot_data[f'{feature_name}_ma_{window}m'] = pivot_data[feature_name].rolling(window=window).mean()

                # Add changes
                pivot_data[f'{feature_name}_change_1m'] = pivot_data[feature_name].pct_change(fill_method=None)
                pivot_data[f'{feature_name}_change_12m'] = pivot_data[feature_name].pct_change(12, fill_method=None)

        # Add time-based features
        pivot_data['year'] = pivot_data['date'].dt.year
        pivot_data['quarter'] = pivot_data['date'].dt.quarter
        pivot_data['month'] = pivot_data['date'].dt.month

        return pivot_data

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
                print(f"Date range: {data['date'].min()} to {data['date'].max()}")
                if 'series_id' in data.columns:
                    print(f"Unique series: {data['series_id'].nunique()}")
                    print(f"Series IDs: {', '.join(data['series_id'].unique())}")
                print(f"Columns: {list(data.columns)}")

        except Exception as e:
            print(f"Error saving data to CSV: {e}")
            # Debug info
            print(f"Debug: Data columns being saved: {list(data.columns)}")
            print(f"Debug: Data shape: {data.shape}")
            if 'series_id' in data.columns:
                print(f"Debug: Data has series_id column - long format")
            else:
                print(f"Debug: Data has feature_name column - wide format")

    def fetch_and_save_economic_data(self) -> str:
        """
        Fetch all required economic data and save to CSV.

        Returns:
            Path to saved CSV file
        """
        print("Fetching FRED economic data...")

        # Fetch required economic series
        print("\n=== Fetching Required Economic Series ===")
        required_data = self.fetch_multiple_series(
            series_ids=list(self.required_series.keys()),
            start_date=(datetime.now() - timedelta(days=10*365)).strftime('%Y-%m-%d')  # 10 years for better modeling
        )

        if not required_data.empty:
            # Add economic features
            print("\n=== Adding Economic Features ===")
            enhanced_data = self.add_economic_features(required_data)

            # Create wide format for easier modeling
            print("\n=== Creating Wide Format Dataset ===")
            pivot_data = self.create_wide_format_dataset(enhanced_data)

            # Save both long and wide format
            output_file_long = "fred_monthly_long.csv"
            output_file_wide = "fred_monthly.csv"

            self.save_to_csv(enhanced_data, output_file_long)
            # Save wide format data (handle potential errors gracefully)
            try:
                self.save_to_csv(pivot_data, output_file_wide)
            except Exception as e:
                print(f"Warning: Could not save wide format data: {e}")
                print("Proceeding with long format data only")

            # Return full path for wide format
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data_raw')
            return os.path.join(data_dir, output_file_wide)
        else:
            print("No economic data fetched, nothing to save")
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

        # Initialize quality report with safe values
        quality_report = {
            "total_rows": len(data),
            "total_columns": len(data.columns),
            "date_range": {
                "start": pd.to_datetime(data['date']).min(),
                "end": pd.to_datetime(data['date']).max(),
                "days": (pd.to_datetime(data['date']).max() - pd.to_datetime(data['date']).min()).days
            },
            "series": [],
            "feature_names": [],
            "missing_values": data.isnull().sum().to_dict(),
            "data_types": data.dtypes.to_dict(),
            "duplicates": data.duplicated().sum(),
            "series_stats": {}
        }

        # Statistics for each series
        # Handle both long format (with series_id) and wide format (with feature_name)
        series_col = None
        feature_col = None

        # Determine the data format
        if 'series_id' in data.columns:
            series_col = 'series_id'
            feature_col = 'feature_name'
            # Populate series and feature_names for long format
            quality_report["series"] = data['series_id'].unique().tolist()
            if 'feature_name' in data.columns:
                quality_report["feature_names"] = data['feature_name'].unique().tolist()

        elif any(col in data.columns for col in ['cpi', 'interest_rates', 'unemployment', 'consumer_sentiment',
                                              'money_supply', 'industrial_production', 'consumer_spending']):
            # Wide format data - use feature columns
            feature_columns = [col for col in ['cpi', 'interest_rates', 'unemployment', 'consumer_sentiment',
                                             'money_supply', 'industrial_production', 'consumer_spending']
                            if col in data.columns]

            # Populate feature_names for wide format
            quality_report["feature_names"] = feature_columns
            quality_report["series"] = [f"wide_format_{col}" for col in feature_columns]  # Create mock series IDs

            # Statistics for each feature column in wide format
            for feature_col in feature_columns:
                if feature_col in data.columns and not data[feature_col].isna().all():
                    feature_data = data[feature_col].dropna()
                    if not feature_data.empty:
                        quality_report["series_stats"][feature_col] = {
                            "observations": len(feature_data),
                            "min_value": feature_data.min(),
                            "max_value": feature_data.max(),
                            "mean_value": feature_data.mean(),
                            "latest_value": feature_data.iloc[-1],
                            "latest_date": data['date'].iloc[-1] if 'date' in data.columns else None,
                            "frequency": 'monthly'
                        }
        else:
            return quality_report

        # Only do long format processing if we have series_id column
        if 'series_id' in data.columns:
            # Get unique identifiers based on format
            if series_col:
                unique_ids = data[series_col].unique()
            else:
                unique_ids = []

            # Statistics for each unique identifier
            for identifier in unique_ids:
                if series_col:
                    series_data = data[data[series_col] == identifier]
                    value_col = 'value'
                else:
                    continue

                if not series_data.empty and value_col in series_data.columns:
                    quality_report["series_stats"][identifier] = {
                        "observations": len(series_data),
                        "min_value": series_data[value_col].min(),
                        "max_value": series_data[value_col].max(),
                        "mean_value": series_data[value_col].mean(),
                        "latest_value": series_data[value_col].iloc[-1],
                        "latest_date": series_data['date'].iloc[-1] if 'date' in series_data.columns else None,
                        "frequency": series_data['frequency'].iloc[0] if 'frequency' in series_data.columns else 'unknown'
                    }

        return quality_report

def main():
    """Main execution function for FRED data fetching"""
    try:
        # Initialize fetcher
        fetcher = FREDFetcher()

        # Fetch data for economic modeling
        output_path = fetcher.fetch_and_save_economic_data()

        if output_path:
            print(f"\n Successfully completed FRED data fetch")
            print(f" Data saved to: {output_path}")

            # Load and validate saved data
            df = pd.read_csv(output_path)
            # Convert date column
            df['date'] = pd.to_datetime(df['date'])
            try:
                quality_report = fetcher.validate_data_quality(df)
            except Exception as e:
                print(f"Warning: Could not validate data quality: {e}")
                print("Data fetch completed successfully, but validation failed.")
                return 0

            print(f"\n Data Quality Report:")
            print(f"   Total rows: {quality_report['total_rows']:,}")
            print(f"   Total columns: {quality_report['total_columns']}")
            print(f"   Date range: {quality_report['date_range']['start']} to {quality_report['date_range']['end']}")
            print(f"   Economic series: {', '.join(quality_report['series'])}")
            print(f"   Duplicates: {quality_report['duplicates']}")

            # Check for missing values in critical economic features
            critical_features = ['cpi', 'interest_rates', 'unemployment', 'consumer_sentiment',
                               'money_supply', 'industrial_production', 'consumer_spending']
            available_features = quality_report.get('feature_names', [])
            missing_features = [f for f in critical_features if f not in available_features]

            if missing_features:
                print(f"\n  Missing economic features: {missing_features}")
            else:
                print(f"\n All required economic features present")

            # Print series statistics
            print(f"\n Economic Series Summary:")
            for series_id, stats in quality_report['series_stats'].items():
                print(f"   {series_id}: {stats['observations']} observations, "
                      f"Latest: {stats['latest_value']:.2f} ({stats['latest_date']})")

        else:
            print(" Failed to fetch FRED data")
            return 1

        return 0

    except Exception as e:
        print(f" Error in main execution: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
