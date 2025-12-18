"""
Build Extended Dataset for Retail Prediction

This script merges FRED economic data, MRTS retail sales data, and Yahoo Finance
stock data to create a comprehensive multivariate dataset with ~50 features
for retail sales prediction.



"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fetch_fred import FREDFetcher
from fetch_mrts import MRTSFetcher
from fetch_yahoo import YahooFinanceFetcher


class DatasetBuilder:
    """Build extended dataset from multiple data sources"""

    def __init__(self):
        """Initialize the dataset builder"""
        self.fred_fetcher = FREDFetcher()
        self.mrts_fetcher = MRTSFetcher()
        self.yahoo_fetcher = YahooFinanceFetcher()

        # Target retail stocks for analysis
        self.target_tickers = ['AAPL', 'WMT', 'AMZN', 'COST']

        # MRTS target categories (NAICS codes)
        self.mrts_categories = {
            'retail_total_sales': '4400A',           # Total Retail Sales
            'clothing_sales': '452',                 # Clothing and Accessories
            'electronics_sales': '44X72',           # Electronics and Appliances
            'furniture_sales': '442',               # Furniture and Home Furnishings
            'sporting_goods_sales': '453',          # Sporting Goods, Hobby, Musical
            'food_beverage_sales': '445',           # Food and Beverage Stores
            'gas_station_sales': '448',             # Gasoline Stations
            'general_merchandise_sales': '454',     # General Merchandise Stores
            'nonstore_retail_sales': '456',         # Nonstore Retailers (E-commerce)
            'health_personal_care_sales': '447'     # Health and Personal Care Stores
        }

    def load_sources(self) -> Dict[str, pd.DataFrame]:
        """
        Load data from all three sources

        Returns:
            Dictionary with DataFrames from each source
        """
        print(" Loading data from three sources...")

        # Load FRED economic data
        print("   Loading FRED economic data...")
        fred_file = os.path.join('..', 'data_raw', 'fred_monthly.csv')
        if os.path.exists(fred_file):
            fred_data = pd.read_csv(fred_file)
            fred_data['date'] = pd.to_datetime(fred_data['date'])
            print(f"    Loaded FRED: {len(fred_data)} rows")
        else:
            print(f"    FRED data file not found at {fred_file}")
            # Try alternative path
            alt_fred_file = os.path.join('data_raw', 'fred_monthly.csv')
            if os.path.exists(alt_fred_file):
                fred_data = pd.read_csv(alt_fred_file)
                fred_data['date'] = pd.to_datetime(fred_data['date'])
                print(f"    Loaded FRED from alternate path: {len(fred_data)} rows")
            else:
                print("    FRED data file not found, fetching...")
                fred_data = self.fred_fetcher.fetch_and_save_economic_data()
                fred_data = pd.read_csv(fred_file)
                fred_data['date'] = pd.to_datetime(fred_data['date'])

        # Load MRTS retail sales data
        print("   Loading MRTS retail sales data...")
        mrts_file = os.path.join('..', 'data_raw', 'mrts_monthly.csv')
        if os.path.exists(mrts_file):
            mrts_data = pd.read_csv(mrts_file)
            mrts_data['date'] = pd.to_datetime(mrts_data['date'])
            print(f"    Loaded MRTS: {len(mrts_data)} rows")
        else:
            print(f"    MRTS data file not found at {mrts_file}")
            # Try alternative path
            alt_mrts_file = os.path.join('data_raw', 'mrts_monthly.csv')
            if os.path.exists(alt_mrts_file):
                mrts_data = pd.read_csv(alt_mrts_file)
                mrts_data['date'] = pd.to_datetime(mrts_data['date'])
                print(f"    Loaded MRTS from alternate path: {len(mrts_data)} rows")
            else:
                print("    MRTS data file not found, fetching...")
                self.mrts_fetcher.fetch_and_save_electronics_data()
                mrts_data = pd.read_csv(mrts_file)
                mrts_data['date'] = pd.to_datetime(mrts_data['date'])

        # Load Yahoo Finance stock data
        print("   Loading Yahoo Finance stock data...")
        yahoo_file = os.path.join('..', 'data_raw', 'yahoo_daily.csv')
        if os.path.exists(yahoo_file):
            yahoo_data = pd.read_csv(yahoo_file)
            yahoo_data['Date'] = pd.to_datetime(yahoo_data['Date'], utc=True)
            print(f"    Loaded Yahoo Finance: {len(yahoo_data)} rows")
        else:
            print(f"    Yahoo Finance data file not found at {yahoo_file}")
            # Try alternative path
            alt_yahoo_file = os.path.join('data_raw', 'yahoo_daily.csv')
            if os.path.exists(alt_yahoo_file):
                yahoo_data = pd.read_csv(alt_yahoo_file)
                yahoo_data['Date'] = pd.to_datetime(yahoo_data['Date'], utc=True)
                print(f"    Loaded Yahoo Finance from alternate path: {len(yahoo_data)} rows")
            else:
                print("    Yahoo Finance data file not found, fetching...")
                self.yahoo_fetcher.fetch_and_save_retail_stocks()
                yahoo_data = pd.read_csv(yahoo_file)
                yahoo_data['Date'] = pd.to_datetime(yahoo_data['Date'], utc=True)

        return {
            'fred': fred_data,
            'mrts': mrts_data,
            'yahoo': yahoo_data
        }

    def normalize_dates(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Normalize all dates to month-end

        Args:
            data_dict: Dictionary of DataFrames

        Returns:
            Dictionary with normalized date columns
        """
        print(" Normalizing dates to month-end...")

        # FRED data - already monthly, ensure month-end and remove timezone
        fred_data = data_dict['fred'].copy()
        fred_data['date'] = fred_data['date'] + pd.offsets.MonthEnd(0)
        fred_data['date'] = fred_data['date'].dt.tz_localize(None)

        # MRTS data - already monthly, ensure month-end and remove timezone
        mrts_data = data_dict['mrts'].copy()
        mrts_data['date'] = mrts_data['date'] + pd.offsets.MonthEnd(0)
        mrts_data['date'] = mrts_data['date'].dt.tz_localize(None)

        # Yahoo Finance data - aggregate daily to monthly
        yahoo_data = data_dict['yahoo'].copy()
        yahoo_data['month_end'] = yahoo_data['Date'] + pd.offsets.MonthEnd(0)

        # Aggregate Yahoo Finance data to monthly
        yahoo_monthly = []
        for ticker in self.target_tickers:
            ticker_data = yahoo_data[yahoo_data['symbol'] == ticker].copy()
            if not ticker_data.empty:
                # Group by month and aggregate
                monthly_agg = ticker_data.groupby('month_end').agg({
                    'symbol': 'first',
                    'Close': 'last',
                    'daily_return': 'mean',  # Average daily return
                    'rolling_vol_20d': 'mean',  # Average volatility
                    'Volume': 'mean'  # Average volume
                }).reset_index()

                # Calculate monthly metrics
                monthly_agg['monthly_return'] = monthly_agg['Close'].pct_change()
                monthly_agg['monthly_volatility'] = monthly_agg.groupby('month_end')['daily_return'].transform('std')
                monthly_agg['monthly_avg_volume'] = monthly_agg['Volume']

                # Rename columns for ticker
                for col in ['monthly_return', 'monthly_volatility', 'monthly_avg_volume']:
                    monthly_agg[f'{ticker}_{col}'] = monthly_agg[col]

                yahoo_monthly.append(monthly_agg[['month_end', 'symbol'] +
                                               [f'{ticker}_{col}' for col in ['monthly_return', 'monthly_volatility', 'monthly_avg_volume']]])

        # Combine all tickers
        if yahoo_monthly:
            yahoo_monthly = pd.concat(yahoo_monthly, ignore_index=True)
            yahoo_monthly = yahoo_monthly.rename(columns={'month_end': 'date'})
            # Remove timezone from date column
            yahoo_monthly['date'] = yahoo_monthly['date'].dt.tz_localize(None)
        else:
            yahoo_monthly = pd.DataFrame(columns=['date'])

        return {
            'fred': fred_data,
            'mrts': mrts_data,
            'yahoo': yahoo_monthly
        }

    def merge_sources(self, normalized_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge all data sources on date

        Args:
            normalized_data: Dictionary of normalized DataFrames

        Returns:
            Merged DataFrame
        """
        print(" Merging data sources...")

        # Start with MRTS data as base
        merged = normalized_data['mrts'][['date']].copy()
        merged = merged.drop_duplicates('date').sort_values('date')

        # Merge FRED data
        fred_data = normalized_data['fred'].copy()
        # Select key FRED columns
        fred_cols = ['date', 'cpi', 'interest_rates', 'unemployment', 'consumer_sentiment',
                    'industrial_production', 'consumer_spending', 'money_supply']
        available_fred_cols = [col for col in fred_cols if col in fred_data.columns]

        if available_fred_cols:
            merged = merged.merge(fred_data[available_fred_cols], on='date', how='left')
            print(f"    Merged FRED: {len(available_fred_cols)} columns")

        # Merge Yahoo Finance data (pivot ticker columns)
        yahoo_data = normalized_data['yahoo'].copy()
        if not yahoo_data.empty:
            # Pivot to have ticker columns as separate features
            yahoo_pivot = yahoo_data.pivot_table(
                index='date',
                values=[f'{ticker}_{col}' for ticker in self.target_tickers
                       for col in ['monthly_return', 'monthly_volatility', 'monthly_avg_volume']],
                aggfunc='first'
            ).reset_index()

            merged = merged.merge(yahoo_pivot, on='date', how='left')
            print(f"    Merged Yahoo Finance: {len(yahoo_pivot.columns)-1} ticker features")

        print(f"    Final merged dataset: {len(merged)} rows, {len(merged.columns)} columns")
        return merged

    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add lag features for specified columns

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with lag features added
        """
        print("  Adding lag features...")

        def add_lags(df: pd.DataFrame, col: str, lags: List[int]) -> pd.DataFrame:
            """Helper function to add lag features"""
            for lag in lags:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)
            return df

        # Market feature lags
        market_cols = [f'{ticker}_monthly_return' for ticker in self.target_tickers]
        for col in market_cols:
            if col in df.columns:
                df = add_lags(df, col, [1, 3])

        vol_cols = [f'{ticker}_monthly_volatility' for ticker in self.target_tickers]
        for col in vol_cols:
            if col in df.columns:
                df = add_lags(df, col, [1, 3])

        # Macroeconomic lags
        macro_lags = {
            'cpi': [1, 3, 6, 12],
            'unemployment': [1, 3, 6, 12],
            'interest_rates': [1, 3, 6, 12],
            'consumer_sentiment': [1, 3, 6]
        }

        for col, lags in macro_lags.items():
            if col in df.columns:
                df = add_lags(df, col, lags)

        # Retail lags
        retail_cols = ['retail_total_sales', 'nonstore_retail_sales']
        for col in retail_cols:
            if col in df.columns:
                df = add_lags(df, col, [1, 12])

        lag_count = len([col for col in df.columns if '_lag_' in col])
        print(f"    Added {lag_count} lag features")

        return df

    def add_seasonality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add seasonality and calendar features

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with seasonality features added
        """
        print(" Adding seasonality features...")

        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter

        # Holiday season indicator (Nov-Dec)
        df['is_holiday_season'] = (df['month'].isin([11, 12])).astype(int)

        # Back-to-school season (Aug-Sep)
        df['is_back_to_school'] = (df['month'].isin([8, 9])).astype(int)

        # Days in month
        df['days_in_month'] = df['date'].dt.days_in_month

        # Cyclical features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        seasonality_cols = ['year', 'month', 'quarter', 'is_holiday_season',
                          'is_back_to_school', 'days_in_month', 'month_sin', 'month_cos']
        print(f"    Added {len(seasonality_cols)} seasonality features")

        return df

    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with derived features added
        """
        print(" Adding derived features...")

        # Macroeconomic derived features
        if 'cpi' in df.columns and 'consumer_spending' in df.columns:
            df['real_retail_sales_fred'] = df['consumer_spending'] / (df['cpi'] / 100)

        if 'interest_rates' in df.columns and 'cpi' in df.columns:
            df['cpi_pct_change'] = df['cpi'].pct_change() * 100
            df['interest_rate_pressure'] = df['interest_rates'] - df['cpi_pct_change'].fillna(0)

        if 'consumer_sentiment' in df.columns and 'unemployment' in df.columns:
            df['consumer_health_index'] = df['consumer_sentiment'] / (1 + df['unemployment'])

        # Retail derived features
        retail_cols = [col for col in df.columns if any(cat in col for cat in self.mrts_categories.keys())]

        if 'retail_total_sales' in df.columns and 'nonstore_retail_sales' in df.columns:
            df['ecommerce_share'] = df['nonstore_retail_sales'] / df['retail_total_sales']

        if 'retail_total_sales' in df.columns and 'gas_station_sales' in df.columns:
            df['gas_share'] = df['gas_station_sales'] / df['retail_total_sales']

        derived_count = len([col for col in df.columns if col in [
            'real_retail_sales_fred', 'cpi_pct_change', 'interest_rate_pressure',
            'consumer_health_index', 'ecommerce_share', 'gas_share'
        ]])
        print(f"    Added {derived_count} derived features")

        return df

    def map_mrts_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map MRTS data to our target retail categories

        Args:
            df: DataFrame with MRTS data

        Returns:
            DataFrame with mapped retail categories
        """
        print(" Mapping MRTS retail categories...")

        # Load and process MRTS data by category
        mrts_file = os.path.join('..', 'data_raw', 'mrts_monthly.csv')
        if os.path.exists(mrts_file):
            mrts_data = pd.read_csv(mrts_file)
            mrts_data['date'] = pd.to_datetime(mrts_data['date'])
            mrts_data['date'] = mrts_data['date'] + pd.offsets.MonthEnd(0)
        else:
            # Try alternative path
            alt_mrts_file = os.path.join('data_raw', 'mrts_monthly.csv')
            if os.path.exists(alt_mrts_file):
                mrts_data = pd.read_csv(alt_mrts_file)
                mrts_data['date'] = pd.to_datetime(mrts_data['date'])
                mrts_data['date'] = mrts_data['date'] + pd.offsets.MonthEnd(0)
            else:
                print(f"     MRTS file not found at {mrts_file} or {alt_mrts_file}")
                return df  # Return unchanged if file not found

            # For this implementation, we'll use the electronics data as our primary retail metric
            # In a full implementation, you would fetch all categories
            electronics_data = mrts_data[mrts_data['category_code'] == '44X72'].copy()

            if not electronics_data.empty:
                # Map to our target variable name
                category_mapping = {
                    'value': 'retail_sales'
                }

                for old_col, new_col in category_mapping.items():
                    if old_col in electronics_data.columns:
                        df = df.merge(
                            electronics_data[['date', old_col]].rename(columns={old_col: new_col}),
                            on='date',
                            how='left'
                        )

                # Add percentage change
                if 'retail_sales' in df.columns:
                    df['retail_sales_pct_change'] = df['retail_sales'].pct_change() * 100

                print(f"    Mapped retail categories: {len([col for col in df.columns if 'retail' in col])} features")

        return df

    def build_final_dataset(self) -> pd.DataFrame:
        """
        Build the complete extended dataset

        Returns:
            Final DataFrame with all features
        """
        print(" Building extended retail prediction dataset...")
        print("=" * 60)

        # Step 1: Load data
        data_sources = self.load_sources()

        # Step 2: Normalize dates
        normalized_data = self.normalize_dates(data_sources)

        # Step 3: Merge sources
        merged_df = self.merge_sources(normalized_data)

        # Step 4: Map MRTS categories to target variables
        merged_df = self.map_mrts_categories(merged_df)

        # Step 5: Add derived features
        merged_df = self.add_derived_features(merged_df)

        # Step 6: Add lag features
        merged_df = self.add_lag_features(merged_df)

        # Step 7: Add seasonality features
        merged_df = self.add_seasonality_features(merged_df)

        # Step 8: Clean and prepare final dataset
        # Sort by date
        merged_df = merged_df.sort_values('date')

        # Fill missing values with forward fill
        numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            merged_df[numeric_cols] = merged_df[numeric_cols].fillna(method='ffill').fillna(method='bfill')

        # Remove rows with too many missing values - be more lenient
        if len(merged_df) > 0:
            max_missing_ratio = 0.7  # Allow up to 70% missing values per row (more lenient)
            missing_ratio = merged_df.isnull().sum(axis=1) / len(merged_df.columns)
            initial_rows = len(merged_df)
            merged_df = merged_df[missing_ratio <= max_missing_ratio]
            final_rows = len(merged_df)
            if final_rows < initial_rows:
                print(f"    Removed {initial_rows - final_rows} rows with too many missing values")

        # Additional check: if we still have no data, be even more lenient
        if len(merged_df) == 0:
            print("     No rows after strict filtering, trying more lenient approach...")
            # Go back to the data before the strict missing value filter
            # Re-process without the strict missing value filter
            return self._build_with_lenient_missing_handling()

        print("=" * 60)
        print(f" Final Dataset Summary:")
        print(f"   Rows: {len(merged_df)}")
        print(f"   Columns: {len(merged_df.columns)}")
        if len(merged_df) > 0:
            print(f"   Date range: {merged_df['date'].min()} to {merged_df['date'].max()}")
            print(f"   Missing values: {merged_df.isnull().sum().sum()}")
        else:
            print(f"     No data remaining after processing!")

        # Display column categories
        market_cols = [col for col in merged_df.columns if any(ticker in col for ticker in self.target_tickers)]
        macro_cols = [col for col in merged_df.columns if any(indicator in col for indicator in ['cpi', 'unemployment', 'interest', 'sentiment', 'production'])]
        retail_cols = [col for col in merged_df.columns if 'retail' in col or 'sales' in col]
        season_cols = [col for col in merged_df.columns if any(season in col for season in ['month', 'quarter', 'holiday', 'school'])]

        print(f"\n Feature Categories:")
        print(f"   Market Features: {len(market_cols)}")
        print(f"   Macro Features: {len(macro_cols)}")
        print(f"   Retail Features: {len(retail_cols)}")
        print(f"   Seasonal Features: {len(season_cols)}")
        print(f"   Lag Features: {len([col for col in merged_df.columns if '_lag_' in col])}")
        print(f"   Derived Features: {len([col for col in merged_df.columns if col in ['real_retail_sales_fred', 'cpi_pct_change', 'interest_rate_pressure', 'consumer_health_index', 'ecommerce_share', 'gas_share']])}")

        return merged_df

    def _build_with_lenient_missing_handling(self) -> pd.DataFrame:
        """
        Build dataset with more lenient missing value handling

        Returns:
            DataFrame with more lenient missing value handling
        """
        print("    Using lenient missing value handling...")

        # Load data sources again
        data_sources = self.load_sources()
        normalized_data = self.normalize_dates(data_sources)
        merged_df = self.merge_sources(normalized_data)
        merged_df = self.map_mrts_categories(merged_df)

        # Add features but be more careful with lags and derived features
        merged_df = self.add_derived_features(merged_df)

        # Only add lags for essential features to avoid too many missing values
        essential_lag_cols = ['cpi', 'unemployment', 'interest_rates']
        for col in essential_lag_cols:
            if col in merged_df.columns:
                for lag in [1, 3]:
                    merged_df[f"{col}_lag_{lag}"] = merged_df[col].shift(lag)

        # Add basic seasonality features
        merged_df['year'] = merged_df['date'].dt.year
        merged_df['month'] = merged_df['date'].dt.month
        merged_df['quarter'] = merged_df['date'].dt.quarter
        merged_df['is_holiday_season'] = (merged_df['month'].isin([11, 12])).astype(int)

        # Sort by date
        merged_df = merged_df.sort_values('date')

        # Fill missing values more aggressively
        numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
        merged_df[numeric_cols] = merged_df[numeric_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)

        print(f"    Lenient approach: {len(merged_df)} rows, {len(merged_df.columns)} columns")
        return merged_df

    def save_dataset(self, df: pd.DataFrame, filename: str = 'extended_dataset.csv') -> str:
        """
        Save the final dataset

        Args:
            df: Final DataFrame
            filename: Output filename

        Returns:
            Path to saved file
        """
        # Create data directory if it doesn't exist
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        os.makedirs(data_dir, exist_ok=True)

        # Save to CSV
        output_path = os.path.join(data_dir, filename)
        df.to_csv(output_path, index=False)

        print(f" Dataset saved to: {output_path}")
        return output_path


def main():
    """Main execution function"""
    try:
        # Initialize dataset builder
        builder = DatasetBuilder()

        # Build the extended dataset
        final_dataset = builder.build_final_dataset()

        # Save the dataset
        output_path = builder.save_dataset(final_dataset)

        if len(final_dataset) > 0:
            print(f"\n Successfully built extended dataset!")
            print(f" Output: {output_path}")
            print(f" Shape: {final_dataset.shape}")
            print(f" Period: {final_dataset['date'].min().strftime('%Y-%m-%d')} to {final_dataset['date'].max().strftime('%Y-%m-%d')}")
        else:
            print(f"\n  Dataset created but contains no rows")
            print(f" Output: {output_path}")
            print(f" Shape: {final_dataset.shape}")

        return 0

    except Exception as e:
        print(f" Error building dataset: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)