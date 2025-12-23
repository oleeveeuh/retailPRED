"""
Build Category-wise Multivariate Time Series Dataset

This script creates a separate dataset for each MRTS retail category where:
- Each MRTS category becomes its own unique_id (e.g., "4431US" for Electronics)
- All 50+ exogenous features (FRED macro, Yahoo Finance, derived features) are consistent across categories
- Final format matches: unique_id | ds | y | cpi | interest_rates | unemployment | ... features
- Automatically selects SM "yes" (seasonally adjusted) data from MRTS
- Proper date alignment with left joins on MRTS dates
- Lag features naturally produce early NaNs (kept)
- Each category dataset saved to data/processed/<category>.parquet



"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fetch_fred import FREDFetcher
from fetch_mrts import MRTSFetcher
from fetch_yahoo import YahooFinanceFetcher


class CategoryDatasetBuilder:
    """Build category-wise multivariate time series datasets"""

    def __init__(self):
        """Initialize the category dataset builder"""
        self.fred_fetcher = FREDFetcher()
        self.mrts_fetcher = MRTSFetcher()
        self.yahoo_fetcher = YahooFinanceFetcher()

        # MRTS categories to process (only seasonally adjusted SM data)
        self.mrts_categories = {
            '4431': {  # Electronics example ID from your request
                'series_id': 'MRTSSM44X72USS',
                'name': 'Electronics_and_Appliances',
                'description': 'Electronics and Appliance Stores'
            },
            '4400': {
                'series_id': 'MRTSSM44000USS',
                'name': 'Total_Retail_Sales',
                'description': 'Total Retail Sales'
            },
            '441': {
                'series_id': 'MRTSSM44131USS',
                'name': 'Automobile_Dealers',
                'description': 'Automobile Dealers'
            },
            '442': {
                'series_id': 'MRTSSM44272USS',
                'name': 'Furniture_Home_Furnishings',
                'description': 'Furniture and Home Furnishings'
            },
            '443': {
                'series_id': 'MRTSSM44313USS',
                'name': 'Building_Materials_Garden',
                'description': 'Building Material and Garden Equipment'
            },
            '445': {
                'series_id': 'MRTSSM44511USS',
                'name': 'Food_Beverage_Stores',
                'description': 'Food and Beverage Stores'
            },
            '447': {
                'series_id': 'MRTSSM44711USS',
                'name': 'Health_Personal_Care',
                'description': 'Health and Personal Care Stores'
            },
            '448': {
                'series_id': 'MRTSSM44811USS',
                'name': 'Gasoline_Stations',
                'description': 'Gasoline Stations'
            },
            '452': {
                'series_id': 'MRTSSM45221USS',
                'name': 'Clothing_Accessories',
                'description': 'Clothing and Clothing Accessories'
            },
            '453': {
                'series_id': 'MRTSSM45311USS',
                'name': 'Sporting_Goods_Hobby',
                'description': 'Sporting Goods, Hobby, Musical Instrument'
            },
            '454': {
                'series_id': 'MRTSSM45431USS',
                'name': 'General_Merchandise',
                'description': 'General Merchandise Stores'
            },
            '456': {
                'series_id': 'MRTSSM45611USS',
                'name': 'Nonstore_Retailers',
                'description': 'Nonstore Retailers (E-commerce)'
            }
        }

        # Yahoo Finance tickers for financial indicators
        self.yahoo_tickers = ['AAPL', 'AMZN', 'WMT', 'COST', 'XLY', 'XRT', 'SPY', 'QQQ', 'VIX']

        # Critical FRED series to use as lag features
        self.critical_fred_series = ['CPIAUCSL', 'FEDFUNDS', 'UNRATE', 'UMCSENT', 'M2SL', 'INDPRO', 'PCE']

    def load_fred_features(self, start_date: str = None) -> pd.DataFrame:
        """
        Load all FRED macro features and create wide format

        Args:
            start_date: Start date for data (default: 10 years ago)

        Returns:
            DataFrame with FRED features in wide format
        """
        print(" Loading FRED macro features...")

        if not start_date:
            start_date = (datetime.now() - timedelta(days=10*365)).strftime('%Y-%m-%d')

        # Fetch all FRED data
        fred_data = self.fred_fetcher.fetch_multiple_series(
            series_ids=self.critical_fred_series,
            start_date=start_date
        )

        if fred_data.empty:
            print(" No FRED data fetched")
            return pd.DataFrame()

        # Create wide format DataFrame
        fred_wide = self._create_fred_wide_format(fred_data)

        print(f" Loaded {len(fred_wide)} rows of FRED data with {len(fred_wide.columns)-1} features")
        return fred_wide

    def _create_fred_wide_format(self, fred_data: pd.DataFrame) -> pd.DataFrame:
        """Convert FRED long format to wide format with proper feature names"""

        # Create feature name mapping
        feature_mapping = {
            'CPIAUCSL': 'cpi',
            'FEDFUNDS': 'interest_rates',
            'UNRATE': 'unemployment',
            'UMCSENT': 'consumer_sentiment',
            'M2SL': 'money_supply',
            'INDPRO': 'industrial_production',
            'PCE': 'consumer_spending'
        }

        # Map feature names
        fred_data['feature_name'] = fred_data['series_id'].map(feature_mapping)

        # Pivot to wide format
        fred_wide = fred_data.pivot_table(
            index='date',
            columns='feature_name',
            values='value',
            aggfunc='first'
        ).reset_index()

        # Normalize to month-end and ensure timezone-naive
        fred_wide['date'] = pd.to_datetime(fred_wide['date']).dt.tz_localize(None)
        fred_wide['date'] = fred_wide['date'] + pd.offsets.MonthEnd(0)

        # Sort by date
        fred_wide = fred_wide.sort_values('date').reset_index(drop=True)

        return fred_wide

    def load_yahoo_features(self, start_date: str = None) -> pd.DataFrame:
        """
        Load Yahoo Finance data and aggregate to monthly features

        Args:
            start_date: Start date for data (default: 10 years ago)

        Returns:
            DataFrame with Yahoo Finance monthly features
        """
        print(" Loading Yahoo Finance features...")

        if not start_date:
            start_date = (datetime.now() - timedelta(days=10*365)).strftime('%Y-%m-%d')

        # Fetch data for all tickers
        yahoo_data = self.yahoo_fetcher.fetch_multiple_tickers(
            symbols=self.yahoo_tickers,
            period="10y",
            interval="1d"
        )

        if yahoo_data.empty:
            print(" No Yahoo Finance data fetched")
            return pd.DataFrame()

        # Aggregate to monthly features
        yahoo_monthly = self._aggregate_yahoo_to_monthly(yahoo_data)

        print(f" Loaded {len(yahoo_monthly)} rows of Yahoo Finance data with {len(yahoo_monthly.columns)-1} features")
        return yahoo_monthly

    def _aggregate_yahoo_to_monthly(self, yahoo_data: pd.DataFrame) -> pd.DataFrame:
        """Aggregate daily Yahoo Finance data to monthly features"""

        # Convert Date to datetime, make timezone-naive, and add month_end
        yahoo_data['Date'] = pd.to_datetime(yahoo_data['Date'], utc=True).dt.tz_localize(None)
        yahoo_data['month_end'] = yahoo_data['Date'] + pd.offsets.MonthEnd(0)

        # Group by month and symbol to calculate monthly features
        monthly_features = []

        for ticker in self.yahoo_tickers:
            ticker_data = yahoo_data[yahoo_data['symbol'] == ticker].copy()
            if ticker_data.empty:
                continue

            # Group by month and calculate aggregates
            monthly_agg = ticker_data.groupby('month_end').agg({
                'Close': 'last',  # Month-end close price
                'daily_return': 'mean',  # Average daily return
                'rolling_vol_20d': 'mean',  # Average volatility
                'Volume': 'mean',  # Average volume
                'rsi_14': 'mean',  # Average RSI
                'macd': 'mean',  # Average MACD
                'high_low_spread': 'mean'  # Average spread
            }).reset_index()

            # Calculate additional monthly features
            monthly_agg[f'{ticker}_monthly_return'] = monthly_agg['Close'].pct_change()
            monthly_agg[f'{ticker}_monthly_volatility'] = monthly_agg.groupby('month_end')['daily_return'].transform('std')
            monthly_agg[f'{ticker}_avg_volume'] = monthly_agg['Volume']
            monthly_agg[f'{ticker}_price_momentum'] = monthly_agg['Close'].pct_change(3)  # 3-month momentum

            # Select and rename columns
            ticker_features = monthly_agg[[
                'month_end',
                f'{ticker}_monthly_return',
                f'{ticker}_monthly_volatility',
                f'{ticker}_avg_volume',
                f'{ticker}_price_momentum'
            ]].rename(columns={'month_end': 'date'})

            monthly_features.append(ticker_features)

        # Merge all ticker features
        if monthly_features:
            yahoo_monthly = monthly_features[0]
            for ticker_df in monthly_features[1:]:
                yahoo_monthly = yahoo_monthly.merge(ticker_df, on='date', how='outer')

            # Sort by date and normalize to month-end, ensure timezone-naive
            yahoo_monthly['date'] = pd.to_datetime(yahoo_monthly['date']).dt.tz_localize(None)
            yahoo_monthly = yahoo_monthly.sort_values('date').reset_index(drop=True)

            return yahoo_monthly
        else:
            return pd.DataFrame()

    def load_mrts_category_data(self, category_id: str, category_info: dict) -> pd.DataFrame:
        """
        Load MRTS data for a specific category (SM seasonally adjusted only)

        Args:
            category_id: Category ID (e.g., '4431')
            category_info: Dictionary with category info including series_id

        Returns:
            DataFrame with MRTS data for the category
        """
        print(f" Loading MRTS data for {category_info['name']} ({category_info['series_id']})...")

        # Fetch category data for the last 10 years
        start_year = datetime.now().year - 10
        category_data = self.mrts_fetcher.fetch_series(
            series_id=category_info['series_id'],
            start_year=start_year
        )

        if category_data.empty:
            print(f" No MRTS data fetched for {category_info['name']}")
            return pd.DataFrame()

        # Filter for ONLY SM (Sales, Millions) AND seasonally adjusted data ("yes")
        filtered_data = category_data[
            (category_data['data_type'] == 'SM') &
            (category_data['seasonally_adj'] == 'yes')
        ].copy()

        if filtered_data.empty:
            print(f" CRITICAL: No seasonally adjusted SM ('SM' + 'yes') data found for {category_info['name']}")
            print(f"   Available data types: {category_data['data_type'].unique() if 'data_type' in category_data.columns else 'N/A'}")
            print(f"   Available seasonal adjustments: {category_data['seasonally_adj'].unique() if 'seasonally_adj' in category_data.columns else 'N/A'}")
            print(f"     This category will be skipped - only SM 'yes' data is allowed")
            return pd.DataFrame()

        # Normalize to month-end and ensure timezone-naive
        filtered_data['date'] = pd.to_datetime(filtered_data['date']).dt.tz_localize(None)
        filtered_data['date'] = filtered_data['date'] + pd.offsets.MonthEnd(0)

        # Sort by date
        filtered_data = filtered_data.sort_values('date').reset_index(drop=True)

        print(f" Loaded {len(filtered_data)} observations for {category_info['name']}")
        return filtered_data[['date', 'value']]

    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features (pct changes, rolling windows)

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with derived features
        """
        print(" Adding derived features...")

        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()

        # Target variable (y) - percentage change
        if 'y' in df.columns:
            df['y_pct_change'] = df['y'].pct_change()
            df['y_lag_1'] = df['y'].shift(1)
            df['y_lag_3'] = df['y'].shift(3)
            df['y_lag_6'] = df['y'].shift(6)
            df['y_lag_12'] = df['y'].shift(12)

        # Macroeconomic features - lags and changes
        macro_cols = ['cpi', 'interest_rates', 'unemployment', 'consumer_sentiment',
                     'money_supply', 'industrial_production', 'consumer_spending']

        for col in macro_cols:
            if col in df.columns:
                # Percentage changes
                df[f'{col}_pct_change'] = df[col].pct_change()
                df[f'{col}_pct_change_yoy'] = df[col].pct_change(12)  # Year-over-year

                # Lags
                for lag in [1, 3, 6, 12]:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)

                # Rolling windows
                for window in [3, 6, 12]:
                    df[f'{col}_ma_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                    df[f'{col}_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()

        # Yahoo Finance features - lags
        yahoo_cols = [col for col in df.columns if any(ticker in col for ticker in self.yahoo_tickers)]

        for col in yahoo_cols:
            if 'monthly_return' in col or 'monthly_volatility' in col:
                df[f'{col}_lag_1'] = df[col].shift(1)
                df[f'{col}_lag_3'] = df[col].shift(3)

        print(f" Added derived features")
        return df

    def add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add calendar features (month, quarter, year, holidays)

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with calendar features
        """
        print(" Adding calendar features...")

        df = df.copy()

        # Use 'ds' column (the date column after merging)
        date_col = 'ds' if 'ds' in df.columns else 'date'
        if date_col not in df.columns:
            print(f" No date column found. Available columns: {list(df.columns)}")
            return df

        # Basic time features
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['quarter'] = df[date_col].dt.quarter
        df['day_of_year'] = df[date_col].dt.dayofyear

        # Cyclical features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)

        # Holiday season indicators
        df['is_holiday_season'] = (df['month'].isin([11, 12])).astype(int)
        df['is_black_friday_month'] = (df['month'] == 11).astype(int)
        df['is_december'] = (df['month'] == 12).astype(int)

        # Back-to-school season
        df['is_back_to_school'] = (df['month'].isin([7, 8])).astype(int)

        # Summer season
        df['is_summer'] = (df['month'].isin([6, 7, 8])).astype(int)

        # Quarter-end indicators
        df['is_quarter_end'] = (df['month'].isin([3, 6, 9, 12])).astype(int)

        # Days in month
        df['days_in_month'] = df[date_col].dt.days_in_month

        print(f" Added calendar features")
        return df

    def build_category_dataset(self, category_id: str, category_info: dict,
                            fred_features: pd.DataFrame, yahoo_features: pd.DataFrame) -> pd.DataFrame:
        """
        Build complete dataset for a single MRTS category

        Args:
            category_id: Category ID (e.g., '4431')
            category_info: Category information dictionary
            fred_features: FRED macro features DataFrame
            yahoo_features: Yahoo Finance features DataFrame

        Returns:
            Complete dataset for the category
        """
        print(f"\n Building dataset for {category_info['name']}...")
        print("=" * 60)

        # Step 1: Load MRTS category data
        mrts_data = self.load_mrts_category_data(category_id, category_info)
        if mrts_data.empty:
            print(f" No MRTS data for {category_info['name']}, skipping")
            return pd.DataFrame()

        # Step 2: Create base DataFrame with MRTS dates
        category_df = mrts_data.rename(columns={'value': 'y'}).copy()
        category_df['unique_id'] = category_id + 'US'  # Format: "4431US" for Electronics

        # Step 3: Rename date column to 'ds' for final format
        category_df = category_df.rename(columns={'date': 'ds'})

        # Step 4: Left join FRED features
        print(" Merging FRED macro features...")
        # Ensure both date columns are timezone-naive
        fred_features_clean = fred_features.copy()
        fred_features_clean['date'] = pd.to_datetime(fred_features_clean['date']).dt.tz_localize(None)
        fred_merge = category_df.merge(fred_features_clean, left_on='ds', right_on='date', how='left')
        fred_merge = fred_merge.drop('date', axis=1)  # Remove redundant date column

        # Step 5: Left join Yahoo Finance features
        print(" Merging Yahoo Finance features...")
        # Ensure both date columns are timezone-naive
        yahoo_features_clean = yahoo_features.copy()
        yahoo_features_clean['date'] = pd.to_datetime(yahoo_features_clean['date']).dt.tz_localize(None)
        yahoo_merge = fred_merge.merge(yahoo_features_clean, left_on='ds', right_on='date', how='left')
        yahoo_merge = yahoo_merge.drop('date', axis=1)  # Remove redundant date column

        # Step 6: Add derived features
        final_df = self.add_derived_features(yahoo_merge)

        # Step 7: Add calendar features
        final_df = self.add_calendar_features(final_df)

        # Step 8: Reorder columns to match required format
        final_df = self._reorder_columns(final_df)

        # Step 9: Sort by date
        final_df = final_df.sort_values('ds').reset_index(drop=True)

        print("=" * 60)
        print(f" Final Dataset Summary for {category_info['name']}:")
        print(f"   Rows: {len(final_df)}")
        print(f"   Columns: {len(final_df.columns)}")
        if len(final_df) > 0:
            print(f"   Date range: {final_df['ds'].min()} to {final_df['ds'].max()}")
            print(f"   Missing values: {final_df.isnull().sum().sum()}")

        # Feature categories summary
        feature_count = len([col for col in final_df.columns if col not in ['unique_id', 'ds', 'y']])
        print(f"   Features: {feature_count}")

        return final_df

    def _reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reorder columns to match required format: unique_id | ds | y | features..."""

        # Required first columns
        required_cols = ['unique_id', 'ds', 'y']
        remaining_cols = [col for col in df.columns if col not in required_cols]

        # Put core macro features first, then others
        priority_features = [
            'cpi', 'interest_rates', 'unemployment', 'consumer_sentiment',
            'money_supply', 'industrial_production', 'consumer_spending'
        ]

        ordered_features = []
        other_features = []

        for feature in priority_features:
            matching_cols = [col for col in remaining_cols if feature in col and col == feature]
            ordered_features.extend(matching_cols)

        for feature in remaining_cols:
            if feature not in ordered_features:
                other_features.append(feature)

        final_order = required_cols + ordered_features + other_features

        # Ensure all columns are included
        missing_cols = [col for col in df.columns if col not in final_order]
        final_order.extend(missing_cols)

        return df[final_order]

    def build_all_category_datasets(self) -> List[pd.DataFrame]:
        """
        Build datasets for all MRTS categories

        Returns:
            List of DataFrames, one for each category
        """
        print(" Building category-wise datasets...")
        print("=" * 80)

        # Step 1: Load all features once
        fred_features = self.load_fred_features()
        yahoo_features = self.load_yahoo_features()

        if fred_features.empty or yahoo_features.empty:
            print(" Failed to load required features")
            return []

        # Step 2: Build dataset for each category
        category_datasets = []

        for category_id, category_info in self.mrts_categories.items():
            try:
                category_df = self.build_category_dataset(
                    category_id, category_info, fred_features, yahoo_features
                )

                if not category_df.empty:
                    category_datasets.append(category_df)
                    print(f" Successfully built dataset for {category_info['name']}")
                else:
                    print(f"  No data for {category_info['name']}")

            except Exception as e:
                print(f" Error building dataset for {category_info['name']}: {e}")
                continue

        print("=" * 80)
        print(f" Successfully built {len(category_datasets)} category datasets")

        return category_datasets

    def save_category_datasets(self, category_datasets: List[pd.DataFrame]) -> List[str]:
        """
        Save each category dataset to data/processed/<category>.parquet

        Args:
            category_datasets: List of DataFrames, one for each category

        Returns:
            List of saved file paths
        """
        print(" Saving category datasets...")

        # Create processed data directory
        processed_dir = os.path.join(os.path.dirname(__file__), '..', 'data_processed')
        os.makedirs(processed_dir, exist_ok=True)

        saved_paths = []

        for i, df in enumerate(category_datasets):
            try:
                # Get category info
                unique_id = df['unique_id'].iloc[0] if not df.empty else f"category_{i}"
                category_name = self.mrts_categories.get(unique_id.replace('US', ''), {}).get('name', unique_id)

                # Create filename
                filename = f"{category_name}.parquet"
                filepath = os.path.join(processed_dir, filename)

                # Save to parquet
                df.to_parquet(filepath, index=False)

                saved_paths.append(filepath)
                print(f" Saved {category_name}: {len(df)} rows, {len(df.columns)} columns")

            except Exception as e:
                print(f" Error saving dataset {i}: {e}")

        print(f"\n Saved {len(saved_paths)} category datasets to {processed_dir}")
        return saved_paths

    def validate_datasets(self, category_datasets: List[pd.DataFrame]) -> dict:
        """
        Validate all category datasets and generate quality report

        Args:
            category_datasets: List of category DataFrames

        Returns:
            Dictionary with validation results
        """
        print(" Validating category datasets...")

        validation_report = {
            'total_categories': len(category_datasets),
            'categories': {},
            'overall_stats': {
                'total_rows': 0,
                'total_columns': 0,
                'avg_rows_per_category': 0,
                'avg_columns_per_category': 0,
                'common_features': set(),
                'date_ranges': []
            }
        }

        if not category_datasets:
            return validation_report

        all_columns = None

        for df in category_datasets:
            if df.empty:
                continue

            category_id = df['unique_id'].iloc[0]
            category_name = self.mrts_categories.get(category_id.replace('US', ''), {}).get('name', category_id)

            # Category stats
            category_stats = {
                'rows': len(df),
                'columns': len(df.columns),
                'date_range': (df['ds'].min(), df['ds'].max()) if 'ds' in df.columns else None,
                'target_stats': None,
                'missing_values': df.isnull().sum().sum(),
                'missing_ratio': df.isnull().sum().sum() / (len(df) * len(df.columns))
            }

            # Target variable stats
            if 'y' in df.columns:
                y_stats = df['y'].describe()
                category_stats['target_stats'] = {
                    'mean': y_stats['mean'],
                    'std': y_stats['std'],
                    'min': y_stats['min'],
                    'max': y_stats['max'],
                    'non_null_count': df['y'].count()
                }

            validation_report['categories'][category_name] = category_stats
            validation_report['overall_stats']['total_rows'] += len(df)
            validation_report['overall_stats']['date_ranges'].append(category_stats['date_range'])

            # Track common features
            if all_columns is None:
                all_columns = set(df.columns)
            else:
                validation_report['overall_stats']['common_features'].intersection_update(df.columns)

        # Calculate averages
        if validation_report['total_categories'] > 0:
            validation_report['overall_stats']['avg_rows_per_category'] = (
                validation_report['overall_stats']['total_rows'] / validation_report['total_categories']
            )
            validation_report['overall_stats']['avg_columns_per_category'] = (
                validation_report['overall_stats']['total_columns'] / validation_report['total_categories']
            )

        # Print validation summary
        print(f"\n Validation Summary:")
        print(f"   Total categories: {validation_report['total_categories']}")
        print(f"   Total rows: {validation_report['overall_stats']['total_rows']:,}")
        print(f"   Average rows per category: {validation_report['overall_stats']['avg_rows_per_category']:.0f}")
        print(f"   Common features across all categories: {len(validation_report['overall_stats']['common_features'])}")

        return validation_report


def main():
    """Main execution function"""
    try:
        print(" Starting Category-wise Dataset Builder")
        print("=" * 80)

        # Initialize builder
        builder = CategoryDatasetBuilder()

        # Build all category datasets
        category_datasets = builder.build_all_category_datasets()

        if not category_datasets:
            print(" No category datasets built successfully")
            return 1

        # Save datasets
        saved_paths = builder.save_category_datasets(category_datasets)

        # Validate datasets
        validation_report = builder.validate_datasets(category_datasets)

        # Print final summary
        print("\n" + "=" * 80)
        print(" SUCCESS! Category-wise datasets built and saved")
        print("=" * 80)
        print(f" Categories processed: {len(category_datasets)}")
        print(f" Files saved: {len(saved_paths)}")

        for path in saved_paths:
            print(f"   - {path}")

        # Feature summary
        if category_datasets:
            avg_features = len(category_datasets[0].columns) - 3  # Subtract unique_id, ds, y
            print(f" Average features per category: {avg_features}")
            print(f" Date coverage: {validation_report['overall_stats']['avg_rows_per_category']:.0f} months avg")

        return 0

    except Exception as e:
        print(f" Error in main execution: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)