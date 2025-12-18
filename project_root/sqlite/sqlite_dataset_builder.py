"""
SQLite-based Category Dataset Builder
High-performance dataset builder using SQLite backend for caching and incremental updates
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlite.sqlite_loader import SQLiteLoader
from etl.fetch_fred import FREDFetcher
from etl.fetch_mrts import MRTSFetcher
from etl.fetch_yahoo import YahooFinanceFetcher

class SQLiteDatasetBuilder:
    """High-performance dataset builder with SQLite caching"""

    def __init__(self, db_path: str = None, force_rebuild: bool = False):
        """
        Initialize SQLite dataset builder

        Args:
            db_path: Path to SQLite database
            force_rebuild: Force complete rebuild of data
        """
        self.db_path = db_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data",
            "retailpred.db"
        )
        self.force_rebuild = force_rebuild
        self.loader = SQLiteLoader(self.db_path)

        # Initialize fetchers
        self.fred_fetcher = FREDFetcher()
        self.mrts_fetcher = MRTSFetcher()
        self.yahoo_fetcher = YahooFinanceFetcher()

        # MRTS categories (same as original)
        self.mrts_categories = {
            '4431': {
                'series_id': 'MRTSSM44X72USS',
                'name': 'Electronics_and_Appliances',
                'description': 'Electronics and Appliance Stores'
            },
            '4400': {
                'series_id': 'MRTSSM44000USS',
                'name': 'Total_Retail_Sales',
                'description': 'Total Retail Sales (Excluding Motor Vehicle and Parts Dealers)'
            },
            '4422': {
                'series_id': 'MRTSSM4422USS',
                'name': 'Furniture_Home_Furnishings',
                'description': 'Furniture and Home Furnishings Stores'
            },
            '4441': {
                'series_id': 'MRTSSM44413USS',
                'name': 'Food_Beverage_Stores',
                'description': 'Food Services and Drinking Places'
            },
            '4442': {
                'series_id': 'MRTSSM4442USS',
                'name': 'General_Merchandise',
                'description': 'General Merchandise Stores'
            },
            '4421': {
                'series_id': 'MRTSSM44211USS',
                'name': 'Building_Materials_Garden',
                'description': 'Building Material and Garden Equipment and Supplies Dealers'
            },
            '4451': {
                'series_id': 'MRTSSM44511USS',
                'name': 'Clothing_Accessories',
                'description': 'Clothing and Clothing Accessories Stores'
            },
            '4452': {
                'series_id': 'MRTSSM4452USS',
                'name': 'Sporting_Goods_Hobby',
                'description': 'Sporting Goods, Hobby, Book, and Music Stores'
            },
            '4471': {
                'series_id': 'MRTSSM44711USS',
                'name': 'Gasoline_Stations',
                'description': 'Gasoline Stations'
            },
            '4460': {
                'series_id': 'MRTSSM44600USS',
                'name': 'Health_Personal_Care',
                'description': 'Health and Personal Care Stores'
            },
            '4521': {
                'series_id': 'MRTSSM4521USS',
                'name': 'Nonstore_Retailers',
                'description': 'Nonstore Retailers'
            }
        }

        self.logger = logging.getLogger(__name__)

    def build_all_categories(self, start_date: str = None, end_date: str = None):
        """
        Build all category datasets with intelligent caching and incremental updates

        Args:
            start_date: Start date for data fetching (YYYY-MM-DD)
            end_date: End date for data fetching (YYYY-MM-DD)
        """
        self.logger.info("🚀 Starting SQLite-based category dataset building...")

        # Check if we need to rebuild or can use cached data
        if not self.force_rebuild and self.loader.is_data_fresh(max_age_days=1):
            self.logger.info("✅ Using cached data (fresh from recent update)")
            return

        self.logger.info("🔄 Building fresh datasets...")

        # Update categories in database
        self._update_categories_metadata()

        # Update exogenous data (shared across all categories)
        self._update_exogenous_data(start_date, end_date)

        # Update retail sales data for each category
        self._update_retail_sales_data(start_date, end_date)

        # Update derived features for all categories
        self._update_derived_features()

        # Update cache status
        self.loader.update_cache_status('time_series_data', True)

        self.logger.info("✅ All category datasets built successfully!")

    def _update_categories_metadata(self):
        """Update categories metadata in database"""
        self.logger.info("📝 Updating categories metadata...")

        for category_id, category_info in self.mrts_categories.items():
            self.loader.add_category(
                category_id=category_id,
                category_name=category_info['name'],
                description=category_info['description'],
                mrts_series_id=category_info['series_id']
            )

        self.logger.info(f"✅ Updated {len(self.mrts_categories)} categories")

    def _update_exogenous_data(self, start_date: str = None, end_date: str = None):
        """Update exogenous features (shared across all categories)"""
        self.logger.info("📊 Updating exogenous features...")

        try:
            # FRED data
            self._update_fred_data(start_date, end_date)

            # Yahoo Finance data
            self._update_yahoo_finance_data(start_date, end_date)

        except Exception as e:
            self.logger.error(f"❌ Failed to update exogenous data: {e}")
            raise

    def _update_fred_data(self, start_date: str = None, end_date: str = None):
        """Update FRED macroeconomic data"""
        self.logger.info("🏦 Fetching FRED data...")

        # FRED series IDs (consistent with original)
        fred_series = {
            'cpi': 'CPIAUCSL',  # Consumer Price Index
            'interest_rates': 'FEDFUNDS',  # Federal Funds Rate
            'unemployment': 'UNRATE',  # Unemployment Rate
            'gdp_growth': 'GDP',  # GDP (for growth rates)
            'inflation_rate': 'T10YIE',  # 10-Year Breakeven Inflation Rate
        }

        for data_type, series_id in fred_series.items():
            try:
                # Get last update date for this data type
                last_date = self._get_last_date_for_data_type(data_type)

                # Fetch only new data if we have existing data
                if last_date and not self.force_rebuild:
                    fetch_start = (datetime.strptime(last_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
                else:
                    fetch_start = start_date or '1992-01-01'

                # Fetch data
                df = self.fred_fetcher.fetch_series(series_id, fetch_start, end_date)

                if not df.empty:
                    # Add data to database for a representative category (will be shared)
                    # Use first category as representative for shared features
                    first_category_id = list(self.mrts_categories.keys())[0]

                    self.loader.add_time_series_data(
                        df, first_category_id, data_type, 'FRED'
                    )

                    self.logger.info(f"  ✅ {data_type}: {len(df)} records from {fetch_start}")

            except Exception as e:
                self.logger.warning(f"⚠️  Failed to fetch {data_type}: {e}")

    def _update_yahoo_finance_data(self, start_date: str = None, end_date: str = None):
        """Update Yahoo Finance data"""
        self.logger.info("📈 Fetching Yahoo Finance data...")

        yahoo_tickers = {
            'sp500': '^GSPC',  # S&P 500
            'oil_prices': 'CL=F',  # Crude Oil Futures
        }

        for data_type, ticker in yahoo_tickers.items():
            try:
                # Get last update date
                last_date = self._get_last_date_for_data_type(data_type)

                if last_date and not self.force_rebuild:
                    fetch_start = (datetime.strptime(last_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
                else:
                    fetch_start = start_date or '1992-01-01'

                # Fetch data
                df = self.yahoo_fetcher.fetch_data(ticker, fetch_start, end_date)

                if not df.empty:
                    # Use representative category
                    first_category_id = list(self.mrts_categories.keys())[0]

                    self.loader.add_time_series_data(
                        df, first_category_id, data_type, 'YAHOO'
                    )

                    self.logger.info(f"  ✅ {data_type}: {len(df)} records from {fetch_start}")

            except Exception as e:
                self.logger.warning(f"⚠️  Failed to fetch {data_type}: {e}")

    def _update_retail_sales_data(self, start_date: str = None, end_date: str = None):
        """Update MRTS retail sales data for all categories"""
        self.logger.info("🏪 Fetching MRTS retail sales data...")

        for category_id, category_info in self.mrts_categories.items():
            try:
                series_id = category_info['series_id']

                # Get last update date for this category
                last_date = self._get_last_date_for_category(category_id)

                if last_date and not self.force_rebuild:
                    fetch_start = (datetime.strptime(last_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
                else:
                    fetch_start = start_date or '1992-01-01'

                # Fetch data
                df = self.mrts_fetcher.fetch_category(series_id, fetch_start, end_date)

                if not df.empty:
                    self.loader.add_time_series_data(
                        df, category_id, 'retail_sales', 'MRTS'
                    )

                    self.logger.info(f"  ✅ {category_info['name']}: {len(df)} records from {fetch_start}")

            except Exception as e:
                self.logger.warning(f"⚠️  Failed to fetch {category_info['name']}: {e}")

    def _update_derived_features(self):
        """Update derived features for all categories"""
        self.logger.info("🔧 Computing derived features...")

        for category_id in self.mrts_categories.keys():
            try:
                # Get retail sales data
                df = self.loader.get_category_data(category_id, data_types=['retail_sales'])

                if df.empty:
                    self.logger.warning(f"⚠️  No retail sales data for {category_id}")
                    continue

                # Compute derived features
                features_df = self._compute_derived_features(df, category_id)

                if not features_df.empty:
                    self.loader.add_derived_features(features_df, category_id)

                self.logger.info(f"  ✅ {category_id}: {len(features_df)} features")

            except Exception as e:
                self.logger.warning(f"⚠️  Failed to compute features for {category_id}: {e}")

    def _compute_derived_features(self, df: pd.DataFrame, category_id: str) -> pd.DataFrame:
        """Compute derived features for a category"""
        features_list = []

        df = df.copy()
        df = df.sort_values('date')

        # Lag features
        for lag in [1, 3, 6, 12]:
            df[f'lag_{lag}'] = df['retail_sales'].shift(lag)

        # Moving averages
        for window in [3, 6, 12]:
            df[f'ma_{window}'] = df['retail_sales'].rolling(window=window, min_periods=1).mean()

        # Year-over-year change
        df['yoy_change'] = df['retail_sales'].pct_change(12)

        # Month-over-month change
        df['mom_change'] = df['retail_sales'].pct_change(1)

        # Seasonal component (simple approach)
        df['month'] = pd.to_datetime(df['date']).dt.month
        monthly_avg = df.groupby('month')['retail_sales'].mean()
        df['seasonal_component'] = df['month'].map(monthly_avg)

        # Trend component (12-month moving average)
        df['trend'] = df['retail_sales'].rolling(window=12, min_periods=1).mean()

        # Convert to features format
        feature_columns = [col for col in df.columns if col not in ['date', 'retail_sales', 'month']]

        for col in feature_columns:
            feature_df = df[['date', col]].copy()
            feature_df.columns = ['date', 'feature_value']
            feature_df['feature_name'] = col
            feature_df['feature_type'] = self._get_feature_type(col)
            features_list.append(feature_df)

        if features_list:
            return pd.concat(features_list, ignore_index=True)
        else:
            return pd.DataFrame()

    def _get_feature_type(self, feature_name: str) -> str:
        """Determine feature type from feature name"""
        if feature_name.startswith('lag_'):
            return 'lag'
        elif feature_name.startswith('ma_'):
            return 'moving_avg'
        elif feature_name in ['yoy_change', 'mom_change']:
            return 'pct_change'
        elif feature_name == 'seasonal_component':
            return 'seasonal'
        elif feature_name == 'trend':
            return 'trend'
        else:
            return 'other'

    def _get_last_date_for_category(self, category_id: str) -> Optional[str]:
        """Get last available date for a category"""
        try:
            result = self.loader.connection.execute("""
                SELECT MAX(date) as max_date FROM time_series_data
                WHERE category_id = ? AND data_type = 'retail_sales'
            """, (category_id,)).fetchone()

            return result['max_date'] if result and result['max_date'] else None
        except:
            return None

    def _get_last_date_for_data_type(self, data_type: str) -> Optional[str]:
        """Get last available date for a data type"""
        try:
            # Use first category as representative
            first_category_id = list(self.mrts_categories.keys())[0]

            result = self.loader.connection.execute("""
                SELECT MAX(date) as max_date FROM time_series_data
                WHERE category_id = ? AND data_type = ?
            """, (first_category_id, data_type)).fetchone()

            return result['max_date'] if result and result['max_date'] else None
        except:
            return None

    def get_category_dataset(self, category_id: str, start_date: str = None,
                           end_date: str = None, include_features: bool = True) -> pd.DataFrame:
        """
        Get complete dataset for a category

        Args:
            category_id: Category identifier
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            include_features: Whether to include derived features

        Returns:
            DataFrame with complete dataset
        """
        # Get base time series data
        df = self.loader.get_category_data(category_id, start_date, end_date)

        if df.empty:
            return df

        # Add derived features if requested
        if include_features:
            features_df = self.loader.get_derived_features(category_id, start_date, end_date)

            if not features_df.empty:
                # Pivot features
                features_pivot = features_df.pivot_table(
                    index='date',
                    columns='feature_name',
                    values='feature_value',
                    aggfunc='first'
                ).reset_index()

                # Merge with base data
                df = pd.merge(df, features_pivot, on='date', how='left')

        # Ensure date column exists and is properly formatted
        if 'date' in df.columns:
            df['ds'] = pd.to_datetime(df['date'])
            df = df.drop('date', axis=1)

        # Set date as index for consistency with original
        if 'ds' in df.columns:
            df = df.set_index('ds')

        return df

    def get_all_categories_dataset(self, start_date: str = None, end_date: str = None,
                                include_features: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Get datasets for all categories

        Returns:
            Dictionary mapping category names to datasets
        """
        datasets = {}

        for category_id, category_info in self.mrts_categories.items():
            category_name = category_info['name']
            df = self.get_category_dataset(category_id, start_date, end_date, include_features)

            if not df.empty:
                datasets[category_name] = df

        return datasets

    def get_database_stats(self) -> Dict:
        """Get comprehensive database statistics"""
        return self.loader.get_database_stats()

    def close(self):
        """Close database connection"""
        self.loader.close()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()