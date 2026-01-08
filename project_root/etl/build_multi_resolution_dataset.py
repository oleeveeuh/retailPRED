"""
Multi-Resolution Time Series Dataset Builder

Creates features at multiple temporal granularities (daily, weekly, monthly, yearly)
to improve model accuracy by capturing patterns at different time scales.

Data Sources:
- Yahoo Finance: Daily data (already has daily frequency)
- MRTS Census: Monthly data (will be upsampled to daily/weekly)

Approach:
1. Yahoo data: Aggregate UP to weekly/monthly/yearly
2. MRTS data: Interpolate DOWN to weekly/daily
3. Create hierarchical lag features at each granularity
4. Combine all features for rich multi-scale representation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class MultiResolutionBuilder:
    """Build multi-resolution time series features"""

    def __init__(self, data_dir: str = "../data_processed"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path("../data_multi_resolution")
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def resample_to_multiple_frequencies(self, df: pd.DataFrame, date_col: str = 'date') -> Dict[str, pd.DataFrame]:
        """
        Resample time series to multiple frequencies

        Args:
            df: Input DataFrame with datetime column
            date_col: Name of date column

        Returns:
            Dictionary with keys: 'daily', 'weekly', 'monthly', 'yearly'
        """
        # Ensure date column is datetime
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()

        resampled = {}

        # Daily (if not already daily)
        if len(df) > 365 * 10:  # If we have lots of data points, likely daily
            resampled['daily'] = df
        else:
            # Interpolate to daily if we have monthly/weekly data
            resampled['daily'] = self._interpolate_to_daily(df)

        # Weekly (W-MON for Monday start)
        resampled['weekly'] = df.resample('W-MON').agg({
            col: 'mean' if df[col].dtype in [np.float64, float] else 'last'
            for col in df.columns
        })

        # Monthly (MS for month start)
        resampled['monthly'] = df.resample('MS').agg({
            col: 'mean' if df[col].dtype in [np.float64, float] else 'last'
            for col in df.columns
        })

        # Yearly (YS for year start)
        resampled['yearly'] = df.resample('YS').agg({
            col: 'mean' if df[col].dtype in [np.float64, float] else 'last'
            for col in df.columns
        })

        return resampled

    def _interpolate_to_daily(self, df: pd.DataFrame, method: str = 'linear') -> pd.DataFrame:
        """
        Interpolate lower frequency data to daily frequency

        For retail sales data, we use:
        - Linear interpolation for trend between months
        - Add day-of-week seasonality factors
        """
        # Create daily date range
        date_range = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq='D'
        )

        # Reindex to daily and interpolate
        df_daily = df.reindex(date_range)
        df_daily = df_daily.interpolate(method=method)

        # Add day-of-week adjustment for retail sales
        # Retail sales are typically higher on weekends
        if 'y' in df_daily.columns:  # If target variable exists
            dow_adjustment = {
                0: 0.90,  # Monday
                1: 0.95,  # Tuesday
                2: 0.95,  # Wednesday
                3: 1.00,  # Thursday
                4: 1.05,  # Friday
                5: 1.25,  # Saturday
                6: 1.20,  # Sunday
            }

            # Apply day-of-week multipliers to interpolated values
            for day, factor in dow_adjustment.items():
                mask = df_daily.index.dayofweek == day
                df_daily.loc[mask, 'y'] = df_daily.loc[mask, 'y'] * factor

        return df_daily

    def create_multi_resolution_lags(self, df_daily: pd.DataFrame, target_col: str = 'y') -> pd.DataFrame:
        """
        Create lag features at multiple time scales

        Creates:
        - Daily lags: 1, 7, 14, 30 days
        - Weekly lags: 1, 4, 8, 12 weeks
        - Monthly lags: 1, 3, 6, 12 months

        Note: Lags are adaptive based on available data history
        """
        df = df_daily.copy()
        data_length = len(df)

        # Calculate safe lag windows based on data length
        # We need at least 50% of data after creating lags
        max_lag_days = int(data_length * 0.4)

        # === DAILY LAGS ===
        for lag in [1, 7, 14, 30]:
            if lag <= max_lag_days:
                df[f'lag_{lag}d'] = df[target_col].shift(lag)

        # === WEEKLY LAGS (using 7-day periods) ===
        # Removed lag_1w (duplicate of lag_7d)
        for weeks in [4, 8, 12]:
            days = weeks * 7
            if days <= max_lag_days:
                df[f'lag_{weeks}w'] = df[target_col].shift(days)

        # === MONTHLY LAGS (using 30-day periods) ===
        # Removed lag_1m (duplicate of lag_30d)
        for months in [3, 6, 12]:
            days = months * 30
            if days <= max_lag_days:
                df[f'lag_{months}m'] = df[target_col].shift(days)

        # === ROLLING STATISTICS AT MULTIPLE WINDOWS ===

        # Daily windows
        for window in [7, 14, 30]:
            if window <= max_lag_days:
                df[f'rolling_mean_{window}d'] = df[target_col].rolling(window=window, min_periods=1).mean()
                df[f'rolling_std_{window}d'] = df[target_col].rolling(window=window, min_periods=1).std()

        # Weekly windows
        for window in [4, 8, 12]:  # 1, 2, 3 months
            days = window * 7
            if days <= max_lag_days:
                df[f'rolling_mean_{window}w'] = df[target_col].rolling(window=days, min_periods=1).mean()
                df[f'rolling_std_{window}w'] = df[target_col].rolling(window=days, min_periods=1).std()

        # Monthly windows
        for window in [3, 6, 12]:
            days = window * 30
            if days <= max_lag_days:
                df[f'rolling_mean_{window}m'] = df[target_col].rolling(window=days, min_periods=1).mean()
                df[f'rolling_std_{window}m'] = df[target_col].rolling(window=days, min_periods=1).std()

        # === RATE OF CHANGE FEATURES ===

        # Day-over-day (removed duplicates: pct_change_1d, diff_1d - use pct_change_1, diff_1 instead)
        df['pct_change_1'] = df[target_col].pct_change(1)
        df['diff_1'] = df[target_col].diff(1)

        # Week-over-week
        if 7 <= max_lag_days:
            df['pct_change_1w'] = df[target_col].pct_change(7)
            df['diff_1w'] = df[target_col].diff(7)

        # Month-over-month
        if 30 <= max_lag_days:
            df['pct_change_1m'] = df[target_col].pct_change(30)
            df['diff_1m'] = df[target_col].diff(30)

        # Year-over-year (only if we have enough data)
        if 365 <= max_lag_days:
            df['pct_change_1y'] = df[target_col].pct_change(365)
            df['diff_1y'] = df[target_col].diff(365)

        # === MOMENTUM INDICATORS ===

        # 7-day momentum (removed as duplicate of diff_1w)
        # 30-day momentum
        if 30 <= max_lag_days:
            df['momentum_30d'] = (df[target_col] / df[target_col].shift(30) - 1) * 100

        # 90-day momentum (quarterly)
        if 90 <= max_lag_days:
            df['momentum_90d'] = (df[target_col] / df[target_col].shift(90) - 1) * 100

        return df

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create calendar-based features at multiple granularities"""

        df = df.copy()

        # Ensure date index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # === DAILY FEATURES ===
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['day_of_year'] = df.index.dayofyear
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        df['is_month_start'] = df.index.is_month_start.astype(int)
        df['is_month_end'] = df.index.is_month_end.astype(int)
        df['is_quarter_start'] = df.index.is_quarter_start.astype(int)
        df['is_quarter_end'] = df.index.is_quarter_end.astype(int)

        # === WEEKLY FEATURES ===
        df['week_of_year'] = df.index.isocalendar().week.values
        df['week_of_month'] = (df.index.day - 1) // 7 + 1

        # === MONTHLY FEATURES ===
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year

        # === CYCLICAL ENCODINGS (for periodic patterns) ===
        # These help models understand that Dec (12) is close to Jan (1)

        # Daily cycle
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

        # Weekly cycle
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Monthly cycle
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Quarterly cycle
        df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)

        return df

    def combine_multi_resolution_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main method to create multi-resolution feature set

        Steps:
        1. Resample to multiple frequencies
        2. Create temporal features
        3. Create multi-resolution lags
        4. Merge everything back to daily frequency
        """
        print(f"Processing data with shape: {df.shape}")

        # Step 1: Get multi-resolution versions
        resampled = self.resample_to_multiple_frequencies(df)

        print(f"✓ Resampled to {len(resampled)} frequencies:")
        for freq, data in resampled.items():
            print(f"  - {freq}: {len(data)} observations")

        # Step 2: Work with daily version as base
        df_daily = resampled['daily'].copy()

        # Step 3: Add temporal features
        print("✓ Creating temporal features...")
        df_daily = self.create_temporal_features(df_daily)

        # Step 4: Add multi-resolution lag features
        print("✓ Creating multi-resolution lag features...")
        df_daily = self.create_multi_resolution_lags(df_daily, target_col='y')

        # Step 5: Add aggregated statistics from other frequencies
        print("✓ Adding aggregated statistics...")

        # Removed cross-frequency aggregations (weekly_agg_rolling_mean_*, monthly_agg_rolling_mean_*)
        # These were duplicates of rolling_mean_* features and have been removed
        # Year-over-year change (from yearly data)
        yearly_data = resampled['yearly']
        df_daily['yoy_change'] = yearly_data['y'].pct_change(365)
        df_daily['yoy_change'] = df_daily['yoy_change'].ffill()

        # Drop rows with critical NaN values (but keep rows with some features)
        print(f"✓ Dropping rows with critical NaN values...")
        # Only drop rows where the target or key features are NaN
        critical_cols = ['y']  # Add other critical columns if needed
        df_daily = df_daily.dropna(subset=critical_cols)

        # For other features, fill forward/backward to preserve data
        df_daily = df_daily.fillna(method='ffill').fillna(method='bfill')

        print(f"✓ Final shape: {df_daily.shape}")
        print(f"✓ Total features: {len(df_daily.columns)}")

        return df_daily.reset_index()

    def process_category_file(self, input_file: Path, output_file: Path) -> None:
        """Process a single category file"""

        print(f"\n{'='*60}")
        print(f"Processing: {input_file.name}")
        print(f"{'='*60}")

        # Load data
        df = pd.read_csv(input_file)
        print(f"Input shape: {df.shape}")

        # Process
        df_multi = self.combine_multi_resolution_features(df)

        # Save
        df_multi.to_csv(output_file, index=False)
        print(f"✓ Saved to: {output_file}")
        print(f"✓ Memory usage: {df_multi.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    def process_all_categories(self) -> None:
        """Process all retail category files"""

        # Find all processed category files
        category_files = list(self.data_dir.glob("retail_*_processed.csv"))

        print(f"\nFound {len(category_files)} category files to process\n")

        for input_file in category_files:
            output_file = self.output_dir / input_file.name.replace(
                "_processed.csv",
                "_multi_resolution.csv"
            )
            self.process_category_file(input_file, output_file)

        print(f"\n{'='*60}")
        print(f"✓ Multi-resolution dataset creation complete!")
        print(f"{'='*60}")
        print(f"\nOutput directory: {self.output_dir}")
        print(f"Total files created: {len(list(self.output_dir.glob('*.csv')))}")


def main():
    """Main execution"""

    print("\n" + "="*60)
    print("MULTI-RESOLUTION TIME SERIES FEATURE BUILDER")
    print("="*60)

    builder = MultiResolutionBuilder()
    builder.process_all_categories()

    print("\n✓ All done!")


if __name__ == "__main__":
    main()
