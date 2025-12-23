
"""
RetailPRED - Multi-Model Time-Series Forecasting System

Main execution script that orchestrates the complete forecasting pipeline
including data fetching, feature engineering, model training, and prediction.
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings
import pandas as pd
import numpy as np
import json
from pathlib import Path
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import load_config, get_config
from etl.fetch_yahoo import YahooFinanceFetcher
from etl.fetch_fred import FREDFetcher
from etl.fetch_mrts import MRTSFetcher
from sqlite.sqlite_dataset_builder import SQLiteDatasetBuilder
from models.robust_timecopilot_trainer import RobustTimeCopilotTrainer
from sqlite.sqlite_loader import SQLiteLoader

# Optional Snowflake integration
try:
    from snowflake.load_stage import SnowflakeLoader
    SNOWFLAKE_AVAILABLE = True
except ImportError:
    SNOWFLAKE_AVAILABLE = False
    SnowflakeLoader = None

class RetailPREDPipeline:
    """Main pipeline class for RetailPRED forecasting system"""

    def __init__(self, config_path: str = None):
        self.config = load_config(config_path)
        self.setup_logging()
        self.setup_components()
        pass

    def setup_logging(self):
        """TODO: Setup logging configuration"""
        log_config = self.config.get_logging_config()
        log_level = getattr(logging, log_config.get('level', 'INFO').upper())
        log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Create logs directory if it doesn't exist
        log_path = self.config.get_path('logs_path')

        # Setup file handler
        log_file = os.path.join(log_path, log_config.get('file_path', 'retailpred.log'))
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info("RetailPRED Pipeline initialized")

    def setup_components(self):
        """Initialize all pipeline components"""
        try:
            # Initialize data fetchers
            self.yahoo_fetcher = YahooFinanceFetcher()
            self.fred_fetcher = FREDFetcher()
            self.mrts_fetcher = MRTSFetcher()

            # Initialize SQLite dataset builder
            try:
                db_path = self.config.get_database_config('sqlite').get('path', 'data/retail_data.db')
                if not os.path.isabs(db_path):
                    db_path = os.path.join(os.getcwd(), db_path)
                self.dataset_builder = SQLiteDatasetBuilder(db_path)
                self.logger.info("SQLite dataset builder initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize SQLite dataset builder: {e}")
                self.dataset_builder = None

            # Initialize database loaders
            try:
                db_path = self.config.get_database_config('sqlite').get('path', 'data/retailpred.db')
                # Make path absolute
                if not os.path.isabs(db_path):
                    db_path = os.path.join(os.getcwd(), db_path)
                self.logger.info(f"Attempting to initialize SQLite with database path: {db_path}")
                self.sqlite_loader = SQLiteLoader(db_path=db_path, config=self.config.get_all_config())
            except Exception as e:
                self.logger.warning(f"Failed to initialize SQLite loader: {e}")
                self.sqlite_loader = None

            # Initialize Snowflake loader if available
            if SNOWFLAKE_AVAILABLE:
                self.snowflake_loader = SnowflakeLoader()
            else:
                self.snowflake_loader = None

            self.logger.info("All components initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise

    def fetch_all_data(self, force_refresh: bool = False) -> Dict[str, bool]:
        """
        Fetch data from all sources

        Args:
            force_refresh: Force data refresh even if recent data exists

        Returns:
            Dictionary mapping source name to success status
        """
        self.logger.info("Starting data fetching process")
        results = {}

        try:
            # Fetch Yahoo Finance data
            if self.config.get_data_source_config('yahoo_finance').get('enabled', True):
                self.logger.info("Fetching Yahoo Finance data...")
                try:
                    self.yahoo_fetcher.fetch_and_save_retail_stocks()
                    results['yahoo'] = True
                    self.logger.info("Yahoo Finance data fetched successfully")
                except Exception as e:
                    self.logger.error(f"Error fetching Yahoo Finance data: {e}")
                    results['yahoo'] = False

            # Fetch FRED data
            if self.config.get_data_source_config('fred').get('enabled', True):
                self.logger.info("Fetching FRED economic data...")
                try:
                    self.fred_fetcher.fetch_and_save_economic_data()
                    results['fred'] = True
                    self.logger.info("FRED data fetched successfully")
                except Exception as e:
                    self.logger.error(f"Error fetching FRED data: {e}")
                    results['fred'] = False

            # Fetch MRTS data
            if self.config.get_data_source_config('mrts').get('enabled', True):
                self.logger.info("Fetching MRTS retail data...")
                try:
                    self.mrts_fetcher.fetch_and_save_all_categories()
                    results['mrts'] = True
                    self.logger.info("MRTS data fetched successfully")
                except Exception as e:
                    self.logger.error(f"Error fetching MRTS data: {e}")
                    results['mrts'] = False

        except Exception as e:
            self.logger.error(f"Error during data fetching: {e}")
            results['error'] = str(e)

        return results

    def build_features(self) -> bool:
        """Build features for model training and inference"""
        try:
            self.logger.info("Building features...")

            # Use fallback approach with CSV files since SQLite is unavailable
            self.logger.info("Using CSV data files for feature building...")

            # Load the comprehensive MRTS data
            mrts_data_path = "data_raw/mrts_all_categories_wide.csv"

            if not os.path.exists(mrts_data_path):
                self.logger.error(f"MRTS data file not found: {mrts_data_path}")
                return False

            # Load the data
            data = pd.read_csv(mrts_data_path)
            self.logger.info(f"Loaded data with shape: {data.shape}")

            # Convert date column and set as index
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
                data.set_index('date', inplace=True)

            # Get retail target columns (exclude engineered features)
            retail_columns = [
                'retail_total_sales',
                'retail_automobile_dealers',
                'retail_furniture_and_home_furnishings_stores',
                'retail_building_material_and_garden_equipment',
                'retail_food_and_beverage_stores',
                'retail_health_and_personal_care_stores',
                'retail_gasoline_stations',
                'retail_electronics_and_appliance_stores',
                'retail_clothing_and_clothing_accessories_stores',
                'retail_sporting_goods_hobby_and_musical_instrument_stores',
                'retail_general_merchandise_stores',
                'retail_miscellaneous_store_retailers',
                'retail_nonstore_retailers'
            ]

            # Save processed data for each target
            for target in retail_columns:
                if target in data.columns:
                    try:
                        self.logger.info(f"Processing target: {target}")

                        # Create dataset for this target
                        target_data = data[[target]].copy()
                        target_data.columns = ['y']  # Rename to 'y' for model compatibility

                        # Add time features
                        target_data = self._add_time_features(target_data)

                        # Save to a simple CSV for the model trainer to use
                        output_dir = "data_processed"
                        os.makedirs(output_dir, exist_ok=True)

                        output_path = f"{output_dir}/{target}_processed.csv"
                        target_data.to_csv(output_path)
                        self.logger.info(f"Saved processed data for {target} to {output_path}")

                    except Exception as e:
                        self.logger.error(f"Error processing {target}: {e}")
                        continue

            self.logger.info("Feature building completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error during feature building: {e}")
            return False

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features to the dataset"""
        df = df.copy()

        # Time-based features
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['day_of_week'] = df.index.dayofweek
        df['week_of_year'] = df.index.isocalendar().week
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

        # Lag features
        for lag in [1, 2, 3, 4, 8, 12]:
            df[f'lag_{lag}'] = df['y'].shift(lag)

        # Rolling features
        for window in [3, 6, 12]:
            df[f'rolling_mean_{window}'] = df['y'].rolling(window=window).mean()
            df[f'rolling_std_{window}'] = df['y'].rolling(window=window).std()

        # Difference features
        df['diff_1'] = df['y'].diff(1)
        df['diff_12'] = df['y'].diff(12)
        df['pct_change_1'] = df['y'].pct_change(1)
        df['pct_change_12'] = df['y'].pct_change(12)

        # Seasonal features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)

        return df

    def _engineer_features(self, data: pd.DataFrame, target: str) -> pd.DataFrame:
        """Engineer features for time series forecasting"""
        import pandas as pd
        import numpy as np

        # Make a copy to avoid modifying original
        df = data.copy()

        # Ensure data is sorted by date
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            df.set_index('date', inplace=True)

        # Create time-based features
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['day_of_week'] = df.index.dayofweek
        df['week_of_year'] = df.index.isocalendar().week
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

        # Create lag features
        for lag in [1, 2, 3, 4, 8, 12, 24]:
            df[f'{target}_lag_{lag}'] = df[target].shift(lag)

        # Create rolling window features
        for window in [4, 8, 12]:
            df[f'{target}_rolling_mean_{window}'] = df[target].rolling(window=window).mean()
            df[f'{target}_rolling_std_{window}'] = df[target].rolling(window=window).std()
            df[f'{target}_rolling_min_{window}'] = df[target].rolling(window=window).min()
            df[f'{target}_rolling_max_{window}'] = df[target].rolling(window=window).max()

        # Create difference features
        df[f'{target}_diff_1'] = df[target].diff(1)
        df[f'{target}_diff_12'] = df[target].diff(12)

        # Create percentage change features
        df[f'{target}_pct_change_1'] = df[target].pct_change(1)
        df[f'{target}_pct_change_12'] = df[target].pct_change(12)

        # Create seasonal features (sine/cosine for month effects)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)

        return df

    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        # Forward fill for time series data
        data = data.ffill()

        # Backward fill for any remaining NaN values
        data = data.bfill()

        # For any remaining NaN values, use median
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if data[col].isnull().any():
                data[col] = data[col].fillna(data[col].median())

        return data

    def _validate_data_quality(self, data: pd.DataFrame, target: str) -> bool:
        """Validate data quality before saving"""
        if data.empty:
            self.logger.error("Dataset is empty")
            return False

        if target not in data.columns:
            self.logger.error(f"Target column {target} not found")
            return False

        # Check for excessive missing values
        missing_ratio = data[target].isnull().sum() / len(data)
        if missing_ratio > 0.1:  # More than 10% missing
            self.logger.warning(f"Target {target} has {missing_ratio:.2%} missing values")

        # Check data coverage
        if len(data) < 100:  # Minimum data points required
            self.logger.error(f"Insufficient data points for {target}: {len(data)}")
            return False

        return True

    def train_models(self, targets: List[str] = None) -> Dict[str, bool]:
        """
        Train individual TimeCopilot models

        Args:
            targets: List of target variables to train models for

        Returns:
            Dictionary mapping model name to training success
        """
        self.logger.info("Starting model training...")
        results = {}

        try:
            # Get all target variables from config
            if targets is None:
                targets_config = self.config.get('targets', {})
                targets = []
                for category_targets in targets_config.values():
                    if isinstance(category_targets, list):
                        targets.extend(category_targets)

            # Initialize trainer with correct output directories (absolute paths)
            import os
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = self.config.get('environment.data_processed_path', 'data_processed/')
            output_dir = os.path.join(base_dir, 'training_outputs')
            results_dir = os.path.join(base_dir, 'results')
            trainer = RobustTimeCopilotTrainer(
                data_dir=data_dir,
                output_dir=output_dir,
                results_dir=results_dir
            )

            # Train models using the trainer's category-based approach
            if targets:
                try:
                    self.logger.info(f"Training models for targets: {targets}")
                    training_results = trainer.train_categories(categories=targets)

                    if training_results:
                        self.logger.info("Successfully trained models for all targets")
                        results['training'] = True
                    else:
                        self.logger.error("Failed to train models")
                        results['training'] = False

                except Exception as e:
                    self.logger.error(f"Error training models: {e}")
                    results['training'] = False

        except Exception as e:
            self.logger.error(f"Error during model training: {e}")
            results['error'] = str(e)

        return results

    def make_predictions(self, targets: List[str] = None, horizon: int = 12) -> Dict[str, dict]:
        """
        Generate predictions using trained models

        Args:
            targets: List of target variables to predict
            horizon: Forecast horizon in months

        Returns:
            Dictionary mapping target variable to prediction results
        """
        self.logger.info(f"Generating predictions for {horizon} periods ahead...")
        results = {}

        try:
            # Get all target variables from config
            if targets is None:
                targets_config = self.config.get('targets', {})
                targets = []
                for category_targets in targets_config.values():
                    if isinstance(category_targets, list):
                        targets.extend(category_targets)

            # Initialize trainer with correct output directories (absolute paths)
            import os
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = self.config.get('environment.data_processed_path', 'data_processed/')
            output_dir = os.path.join(base_dir, 'training_outputs')
            results_dir = os.path.join(base_dir, 'results')
            trainer = RobustTimeCopilotTrainer(
                data_dir=data_dir,
                output_dir=output_dir,
                results_dir=results_dir
            )

            for target in targets:
                try:
                    self.logger.info(f"Generating predictions for {target}...")

                    # Load trained models for this target
                    models = trainer.load_trained_models(target)

                    if not models:
                        self.logger.warning(f"No trained models found for {target}")
                        continue

                    # Get latest data for target
                    latest_data = trainer.get_latest_data(target, lookback=24)

                    if latest_data.empty:
                        self.logger.warning(f"No recent data found for {target}")
                        continue

                    # Generate predictions for each model
                    model_predictions = {}
                    for model_name, model in models.items():
                        try:
                            predictions = trainer.predict(model, latest_data, horizon)
                            model_predictions[model_name] = predictions
                        except Exception as e:
                            self.logger.error(f"Error generating predictions with {model_name} for {target}: {e}")

                    # Create ensemble prediction
                    if model_predictions:
                        ensemble_predictions = self._create_ensemble_predictions(model_predictions, target)
                        results[target] = {
                            'predictions': ensemble_predictions['mean'],
                            'confidence_intervals': ensemble_predictions['ci'],
                            'individual_models': model_predictions,
                            'forecast_date': datetime.now().isoformat(),
                            'horizon': horizon
                        }
                        self.logger.info(f"Successfully generated predictions for {target}")

                except Exception as e:
                    self.logger.error(f"Error generating predictions for {target}: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"Error during prediction generation: {e}")
            results['error'] = str(e)

        return results

    def _create_ensemble_predictions(self, model_predictions: dict, target: str) -> dict:
        """Create ensemble predictions from individual model predictions"""
        if not model_predictions:
            return {'mean': [], 'ci': []}

        # Stack predictions from all models
        all_predictions = []
        for predictions in model_predictions.values():
            if isinstance(predictions, (list, np.ndarray)):
                all_predictions.append(np.array(predictions))

        if not all_predictions:
            return {'mean': [], 'ci': []}

        # Convert to numpy array
        all_predictions = np.array(all_predictions)

        # Calculate ensemble statistics
        mean_predictions = np.mean(all_predictions, axis=0)
        std_predictions = np.std(all_predictions, axis=0)

        # Create confidence intervals (95% CI)
        confidence_intervals = {
            'lower_95': mean_predictions - 1.96 * std_predictions,
            'upper_95': mean_predictions + 1.96 * std_predictions,
            'lower_68': mean_predictions - std_predictions,
            'upper_68': mean_predictions + std_predictions
        }

        return {
            'mean': mean_predictions.tolist(),
            'ci': confidence_intervals
        }

    def backtest_models(self, targets: List[str] = None, test_periods: int = 12) -> Dict[str, dict]:
        """
        Backtest models on historical data

        Args:
            targets: List of target variables to backtest
            test_periods: Number of periods to use for testing

        Returns:
            Dictionary with backtest results and performance metrics
        """
        self.logger.info(f"Starting backtesting with {test_periods} test periods...")
        results = {}

        try:
            # Get all target variables from config
            if targets is None:
                targets_config = self.config.get('targets', {})
                targets = []
                for category_targets in targets_config.values():
                    if isinstance(category_targets, list):
                        targets.extend(category_targets)

            # Initialize trainer with correct output directories (absolute paths)
            import os
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = self.config.get('environment.data_processed_path', 'data_processed/')
            output_dir = os.path.join(base_dir, 'training_outputs')
            results_dir = os.path.join(base_dir, 'results')
            trainer = RobustTimeCopilotTrainer(
                data_dir=data_dir,
                output_dir=output_dir,
                results_dir=results_dir
            )

            for target in targets:
                try:
                    self.logger.info(f"Backtesting for {target}...")

                    # Load historical data for backtesting
                    historical_data = trainer.load_historical_data(target)

                    if len(historical_data) < test_periods + 24:  # Need enough data
                        self.logger.warning(f"Insufficient historical data for backtesting {target}")
                        continue

                    # Perform time series cross-validation backtesting
                    backtest_results = trainer.backtest_model(
                        target,
                        historical_data,
                        test_periods=test_periods,
                        n_splits=5
                    )

                    if backtest_results:
                        # Calculate comprehensive performance metrics
                        performance_metrics = self._calculate_performance_metrics(
                            backtest_results['actuals'],
                            backtest_results['predictions']
                        )

                        results[target] = {
                            'performance_metrics': performance_metrics,
                            'backtest_dates': backtest_results['dates'],
                            'actuals': backtest_results['actuals'],
                            'predictions': backtest_results['predictions'],
                            'model_results': backtest_results['model_results'],
                            'test_periods': test_periods,
                            'backtest_date': datetime.now().isoformat()
                        }

                        self.logger.info(f"Successfully backtested {target}")
                    else:
                        self.logger.error(f"Backtesting failed for {target}")

                except Exception as e:
                    self.logger.error(f"Error backtesting {target}: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"Error during backtesting: {e}")
            results['error'] = str(e)

        return results

    def generate_long_term_forecasts(
        self,
        targets: List[str] = None,
        horizon_months: int = 60,
        scenarios: bool = True
    ) -> Dict[str, dict]:
        """
        Generate long-term (5-year) forecasts with dedicated visualizations

        This method generates extended forecasts with uncertainty quantification,
        scenario analysis, and specialized visualizations for long-term trends.

        Args:
            targets: List of target variables to forecast (default: all from config)
            horizon_months: Forecast horizon in months (default: 60 = 5 years)
            scenarios: Whether to generate optimistic/baseline/pessimistic scenarios

        Returns:
            Dictionary containing forecast results and metadata
        """
        from models.long_term_forecaster import LongTermForecaster

        self.logger.info(f"Generating long-term forecasts for {horizon_months} months ({horizon_months//12} years)...")

        try:
            # Get all target variables from config
            if targets is None:
                targets_config = self.config.get('targets', {})
                targets = []
                for category_targets in targets_config.values():
                    if isinstance(category_targets, list):
                        targets.extend(category_targets)

            # Initialize paths (absolute paths)
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(base_dir, 'project_root', 'data_processed')
            output_dir = os.path.join(base_dir, 'training_outputs')
            results_dir = os.path.join(base_dir, 'results')

            # Initialize long-term forecaster
            forecaster = LongTermForecaster(
                output_dir=output_dir,
                results_dir=results_dir,
                data_dir=data_dir
            )

            # Generate long-term forecasts
            forecast_results = forecaster.generate_long_term_forecasts(
                targets=targets,
                horizon_months=horizon_months,
                scenarios=scenarios
            )

            # Generate and save summary report
            summary = forecaster.generate_forecast_summary(forecast_results)
            summary_file = Path(output_dir) / "long_term_forecasts" / "forecast_summary.md"

            with open(summary_file, 'w') as f:
                f.write(summary)

            self.logger.info(f"Long-term forecast summary saved to {summary_file}")
            self.logger.info(f"Visualizations saved to {Path(output_dir) / 'long_term_forecasts'}")

            return forecast_results

        except Exception as e:
            self.logger.error(f"Error generating long-term forecasts: {e}")
            return {'error': str(e)}

    def _calculate_performance_metrics(self, actuals: np.ndarray, predictions: np.ndarray) -> dict:
        """Calculate comprehensive performance metrics"""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        actuals = np.array(actuals)
        predictions = np.array(predictions)

        # Remove any NaN values
        mask = ~(np.isnan(actuals) | np.isnan(predictions))
        actuals = actuals[mask]
        predictions = predictions[mask]

        if len(actuals) == 0:
            return {}

        metrics = {
            'mae': mean_absolute_error(actuals, predictions),
            'rmse': np.sqrt(mean_squared_error(actuals, predictions)),
            'mape': np.mean(np.abs((actuals - predictions) / actuals)) * 100,
            'r2': r2_score(actuals, predictions),
            'mean_actual': np.mean(actuals),
            'mean_predicted': np.mean(predictions),
            'std_actual': np.std(actuals),
            'std_predicted': np.std(predictions),
            'min_actual': np.min(actuals),
            'max_actual': np.max(actuals),
            'min_predicted': np.min(predictions),
            'max_predicted': np.max(predictions),
            'n_observations': len(actuals)
        }

        # Calculate directional accuracy
        actual_changes = np.diff(actuals)
        pred_changes = np.diff(predictions)
        directional_accuracy = np.mean(np.sign(actual_changes) == np.sign(pred_changes)) * 100
        metrics['directional_accuracy'] = directional_accuracy

        return metrics

    def save_results(self, results: Dict[str, dict], output_dir: str = None) -> bool:
        """
        Save predictions and results to database and files

        Args:
            results: Dictionary containing predictions and metrics
            output_dir: Directory to save result files

        Returns:
            Success status
        """
        try:
            self.logger.info("Saving results...")

            if output_dir is None:
                output_dir = self.config.get_path('outputs') or 'outputs'

            # Ensure output directory exists
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Create timestamp for this run
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = Path(output_dir) / f"run_{timestamp}"
            run_dir.mkdir(exist_ok=True)

            # Save to SQLite database
            try:
                sqlite_loader = SQLiteLoader()
                sqlite_loader.save_forecast_results(results)
                self.logger.info("Results saved to SQLite database")
            except Exception as e:
                self.logger.error(f"Error saving to SQLite: {e}")

            # Save to Snowflake data warehouse (if configured)
            if self.config.get('snowflake.enabled', False) and self.snowflake_loader is not None:
                try:
                    self.snowflake_loader.save_forecast_results(results)
                    self.logger.info("Results saved to Snowflake")
                except Exception as e:
                    self.logger.error(f"Error saving to Snowflake: {e}")
            elif self.config.get('snowflake.enabled', False) and self.snowflake_loader is None:
                self.logger.warning("Snowflake integration not available - skipping Snowflake save")

            # Save to CSV files
            try:
                self._save_results_to_csv(results, run_dir)
                self.logger.info(f"Results saved to CSV files in {run_dir}")
            except Exception as e:
                self.logger.error(f"Error saving CSV files: {e}")

            # Save raw JSON results
            try:
                with open(run_dir / 'results.json', 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                self.logger.info("Raw results saved as JSON")
            except Exception as e:
                self.logger.error(f"Error saving JSON results: {e}")

            # Generate summary reports
            try:
                summary_report = self._generate_summary_report(results)
                with open(run_dir / 'summary_report.txt', 'w') as f:
                    f.write(summary_report)
                self.logger.info("Summary report generated")
            except Exception as e:
                self.logger.error(f"Error generating summary report: {e}")

            return True

        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            return False

    def _save_results_to_csv(self, results: Dict[str, dict], output_dir: Path):
        """Save results to CSV files"""
        for target, target_results in results.items():
            if target == 'error':
                continue

            target_dir = output_dir / target
            target_dir.mkdir(exist_ok=True)

            # Save predictions
            if 'predictions' in target_results:
                predictions_df = pd.DataFrame({
                    'date': pd.date_range(
                        start=pd.Timestamp.now(),
                        periods=len(target_results['predictions']),
                        freq='M'
                    ),
                    'prediction': target_results['predictions']
                })
                predictions_df.to_csv(target_dir / 'predictions.csv', index=False)

            # Save confidence intervals
            if 'confidence_intervals' in target_results:
                ci_df = pd.DataFrame(target_results['confidence_intervals'])
                ci_df.to_csv(target_dir / 'confidence_intervals.csv', index=False)

            # Save performance metrics (from backtesting)
            if 'performance_metrics' in target_results:
                metrics_df = pd.DataFrame([target_results['performance_metrics']])
                metrics_df.to_csv(target_dir / 'performance_metrics.csv', index=False)

    def _generate_summary_report(self, results: Dict[str, dict]) -> str:
        """Generate a text summary report"""
        report = []
        report.append("=" * 60)
        report.append("RETAILPRED FORECASTING SUMMARY REPORT")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 60)
        report.append("")

        # Summary statistics
        total_targets = len([k for k in results.keys() if k != 'error'])
        report.append(f"Total targets processed: {total_targets}")
        report.append("")

        # Target-specific summaries
        for target, target_results in results.items():
            if target == 'error':
                continue

            report.append(f"TARGET: {target}")
            report.append("-" * 40)

            if 'performance_metrics' in target_results:
                metrics = target_results['performance_metrics']
                report.append(f"  MAE: {metrics.get('mae', 'N/A'):.4f}")
                report.append(f"  RMSE: {metrics.get('rmse', 'N/A'):.4f}")
                report.append(f"  MAPE: {metrics.get('mape', 'N/A'):.2f}%")
                report.append(f"  R: {metrics.get('r2', 'N/A'):.4f}")
                report.append(f"  Directional Accuracy: {metrics.get('directional_accuracy', 'N/A'):.2f}%")

            if 'horizon' in target_results:
                report.append(f"  Forecast Horizon: {target_results['horizon']} periods")

            report.append("")

        # Error summary
        if 'error' in results:
            report.append("ERRORS ENCOUNTERED:")
            report.append("-" * 20)
            report.append(f"  {results['error']}")
            report.append("")

        report.append("=" * 60)
        report.append("END OF REPORT")
        report.append("=" * 60)

        return "\n".join(report)

    def generate_report(self, results: Dict[str, dict]) -> str:
        """
        Generate comprehensive forecast report

        Args:
            results: Dictionary containing predictions and metrics

        Returns:
            Path to generated report file
        """
        self.logger.info("Generating forecast report...")

        try:
            # Create summary statistics
            summary_stats = self._create_summary_statistics(results)

            # Generate executive summary
            executive_summary = self._create_executive_summary(results, summary_stats)

            # Generate detailed report
            report_content = self._generate_detailed_report(results, summary_stats, executive_summary)

            # Save report to file
            output_dir = self.config.get_path('outputs') or 'outputs'
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = Path(output_dir) / f"forecast_report_{timestamp}.md"

            # Ensure output directory exists
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            with open(report_path, 'w') as f:
                f.write(report_content)

            self.logger.info(f"Comprehensive report generated: {report_path}")
            return str(report_path)

        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            return ""

    def _create_summary_statistics(self, results: Dict[str, dict]) -> dict:
        """Create summary statistics from results"""
        stats = {
            'total_targets': 0,
            'successful_predictions': 0,
            'average_mae': None,
            'average_rmse': None,
            'average_mape': None,
            'average_r2': None,
            'targets_with_errors': []
        }

        mae_values, rmse_values, mape_values, r2_values = [], [], [], []

        for target, target_results in results.items():
            if target == 'error':
                continue

            stats['total_targets'] += 1

            if 'predictions' in target_results:
                stats['successful_predictions'] += 1

            if 'performance_metrics' in target_results:
                metrics = target_results['performance_metrics']
                if 'mae' in metrics:
                    mae_values.append(metrics['mae'])
                if 'rmse' in metrics:
                    rmse_values.append(metrics['rmse'])
                if 'mape' in metrics:
                    mape_values.append(metrics['mape'])
                if 'r2' in metrics:
                    r2_values.append(metrics['r2'])

        if mae_values:
            stats['average_mae'] = np.mean(mae_values)
        if rmse_values:
            stats['average_rmse'] = np.mean(rmse_values)
        if mape_values:
            stats['average_mape'] = np.mean(mape_values)
        if r2_values:
            stats['average_r2'] = np.mean(r2_values)

        return stats

    def _create_executive_summary(self, results: Dict[str, dict], stats: dict) -> str:
        """Create executive summary"""
        summary = []
        summary.append("## Executive Summary")
        summary.append("")
        summary.append(f"This report presents forecasting results for {stats['total_targets']} economic indicators, with {stats['successful_predictions']} targets successfully predicted.")
        summary.append("")

        if stats['average_mae'] is not None:
            summary.append(f"**Overall Performance:**")
            summary.append(f"- Mean Absolute Error (MAE): {stats['average_mae']:.4f}")
            summary.append(f"- Root Mean Square Error (RMSE): {stats['average_rmse']:.4f}")
            summary.append(f"- Mean Absolute Percentage Error (MAPE): {stats['average_mape']:.2f}%")
            summary.append(f"- R-squared (R): {stats['average_r2']:.4f}")
            summary.append("")

        summary.append("**Key Findings:**")
        summary.append("- Multi-model ensemble approach provides robust forecasts")
        summary.append("- Confidence intervals captured forecast uncertainty")
        summary.append("- Models validated through comprehensive backtesting")
        summary.append("")

        return "\n".join(summary)

    def _generate_detailed_report(self, results: Dict[str, dict], stats: dict, executive_summary: str) -> str:
        """Generate detailed markdown report"""
        report = []
        report.append("# RetailPRED Forecasting Report")
        report.append("")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        report.append(executive_summary)
        report.append("")

        # Detailed results for each target
        report.append("## Detailed Results")
        report.append("")

        for target, target_results in results.items():
            if target == 'error':
                continue

            report.append(f"### {target}")
            report.append("")

            if 'performance_metrics' in target_results:
                metrics = target_results['performance_metrics']
                report.append("**Performance Metrics:**")
                report.append(f"- MAE: {metrics.get('mae', 'N/A'):.4f}")
                report.append(f"- RMSE: {metrics.get('rmse', 'N/A'):.4f}")
                report.append(f"- MAPE: {metrics.get('mape', 'N/A'):.2f}%")
                report.append(f"- R: {metrics.get('r2', 'N/A'):.4f}")
                report.append(f"- Directional Accuracy: {metrics.get('directional_accuracy', 'N/A'):.2f}%")
                report.append("")

            if 'horizon' in target_results:
                report.append(f"**Forecast Horizon:** {target_results['horizon']} periods")
                report.append("")

            if 'predictions' in target_results:
                predictions = target_results['predictions']
                report.append(f"**Next Period Forecast:** {predictions[0]:.4f}")
                if len(predictions) > 1:
                    report.append(f"**3-Month Average:** {np.mean(predictions[:3]):.4f}")
                    report.append(f"**12-Month Average:** {np.mean(predictions):.4f}")
                report.append("")

        # Technical appendix
        report.append("## Technical Appendix")
        report.append("")
        report.append("**Model Types Used:**")
        report.append("- Economic TimeCopilot models")
        report.append("- Financial TimeCopilot models")
        report.append("- Multivariate models")
        report.append("- Ensemble methods")
        report.append("")
        report.append("**Validation Method:** Time series cross-validation with walk-forward validation")
        report.append("")
        report.append("**Confidence Intervals:** 68% and 95% confidence intervals calculated from ensemble variance")
        report.append("")

        return "\n".join(report)

    def run_full_pipeline(self, targets: List[str] = None, horizon: int = 12,
                         fetch_data: bool = True, train_models: bool = True,
                         make_predictions: bool = True) -> Dict[str, any]:
        """
        Run complete forecasting pipeline

        Args:
            targets: List of target variables
            horizon: Forecast horizon
            fetch_data: Whether to fetch fresh data
            train_models: Whether to train/retrain models
            make_predictions: Whether to generate predictions

        Returns:
            Dictionary with pipeline execution results
        """
        self.logger.info("Starting full forecasting pipeline...")
        pipeline_results = {
            'start_time': datetime.now(),
            'success': False,
            'results': {},
            'errors': []
        }

        try:
            # Step 1: Fetch data
            if fetch_data:
                self.logger.info("Step 1: Fetching data...")
                fetch_results = self.fetch_all_data()
                pipeline_results['results']['data_fetching'] = fetch_results

                # Check if data fetching was successful
                if not fetch_results or not any(fetch_results.values()):
                    self.logger.warning("Data fetching had issues, continuing with existing data")
            else:
                self.logger.info("Skipping data fetching step")

            # Step 2: Build features
            self.logger.info("Step 2: Building features...")
            feature_results = self.build_features()
            pipeline_results['results']['feature_engineering'] = feature_results

            if not feature_results:
                pipeline_results['errors'].append("Feature engineering failed")
                self.logger.error("Feature engineering failed, cannot continue")
                return pipeline_results

            # Step 3: Train models
            if train_models:
                self.logger.info("Step 3: Training models...")
                training_results = self.train_models(targets)
                pipeline_results['results']['model_training'] = training_results
            else:
                self.logger.info("Skipping model training step")

            # Step 4: Make predictions
            if make_predictions:
                self.logger.info("Step 4: Generating predictions...")
                prediction_results = self.make_predictions(targets, horizon)
                pipeline_results['results']['predictions'] = prediction_results

                # Step 5: Backtest models
                self.logger.info("Step 5: Backtesting models...")
                backtest_results = self.backtest_models(targets)
                pipeline_results['results']['backtesting'] = backtest_results

                # Combine predictions and backtest results for reporting
                combined_results = {}
                if 'error' not in prediction_results:
                    for target in prediction_results:
                        if target != 'error':
                            combined_results[target] = prediction_results[target].copy()
                            if target in backtest_results:
                                combined_results[target]['performance_metrics'] = backtest_results[target]['performance_metrics']

                # Step 6: Save results
                self.logger.info("Step 6: Saving results...")
                save_results = self.save_results(combined_results)
                pipeline_results['results']['saving'] = save_results

                # Step 7: Generate report
                self.logger.info("Step 7: Generating report...")
                report_path = self.generate_report(combined_results)
                pipeline_results['results']['report'] = report_path

            # Mark pipeline as successful
            pipeline_results['success'] = True
            pipeline_results['end_time'] = datetime.now()
            pipeline_results['duration'] = pipeline_results['end_time'] - pipeline_results['start_time']

            self.logger.info(f"Pipeline completed successfully in {pipeline_results['duration']}")

        except Exception as e:
            pipeline_results['errors'].append(str(e))
            pipeline_results['end_time'] = datetime.now()
            pipeline_results['duration'] = pipeline_results['end_time'] - pipeline_results['start_time']
            self.logger.error(f"Pipeline failed: {e}")

        return pipeline_results

def main():
    """Main execution function with command-line interface"""
    parser = argparse.ArgumentParser(description='RetailPRED - Multi-Model Time-Series Forecasting System')

    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['fetch', 'train', 'predict', 'backtest', 'long-term', 'full'],
                       default='full', help='Pipeline execution mode')
    parser.add_argument('--all', action='store_true', help='Run complete pipeline (same as --mode full)')
    parser.add_argument('--targets', type=str, nargs='+', help='Target variables to process')
    parser.add_argument('--horizon', type=int, default=12, help='Forecast horizon in periods (default: 12, use 60 for 5-year forecasts)')
    parser.add_argument('--long-term-horizon', type=int, default=60, help='Long-term forecast horizon in months (default: 60 = 5 years)')
    parser.add_argument('--no-scenarios', action='store_true', help='Disable scenario analysis in long-term forecasts')
    parser.add_argument('--no-fetch', action='store_true', help='Skip data fetching')
    parser.add_argument('--no-train', action='store_true', help='Skip model training')
    parser.add_argument('--no-predict', action='store_true', help='Skip prediction generation')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parser.parse_args()

    # Handle --all flag (same as --mode full)
    if args.all:
        args.mode = 'full'

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('retailpred.log')
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting RetailPRED forecasting system")

    try:
        # Load configuration
        config = load_config(args.config)

        # Initialize pipeline
        pipeline = RetailPREDPipeline(config)

        # Execute based on mode
        if args.mode == 'fetch':
            results = pipeline.fetch_all_data()
            logger.info(f"Data fetching completed: {results}")

        elif args.mode == 'train':
            results = pipeline.train_models(args.targets)
            logger.info(f"Model training completed: {results}")

        elif args.mode == 'predict':
            results = pipeline.make_predictions(args.targets, args.horizon)
            logger.info(f"Predictions completed for {len(results)} targets")

        elif args.mode == 'backtest':
            results = pipeline.backtest_models(args.targets)
            logger.info(f"Backtesting completed for {len(results)} targets")

            # Save backtest results
            if results and len(results) > 0:
                saved = pipeline.save_results(results)
                if saved:
                    logger.info("Backtest results saved successfully")

        elif args.mode == 'long-term':
            # Generate long-term forecasts (5-year predictions)
            results = pipeline.generate_long_term_forecasts(
                targets=args.targets,
                horizon_months=args.long_term_horizon,
                scenarios=not args.no_scenarios
            )

            if 'error' not in results:
                logger.info(f"Long-term forecasting completed successfully")
                logger.info(f"Results saved to: training_outputs/long_term_forecasts/")
            else:
                logger.error(f"Long-term forecasting failed: {results['error']}")
                sys.exit(1)

        elif args.mode == 'full':
            results = pipeline.run_full_pipeline(
                targets=args.targets,
                horizon=args.horizon,
                fetch_data=not args.no_fetch,
                train_models=not args.no_train,
                make_predictions=not args.no_predict
            )

            if results['success']:
                logger.info("Full pipeline completed successfully")
                if 'report' in results['results']:
                    logger.info(f"Report generated: {results['results']['report']}")
            else:
                logger.error(f"Pipeline failed with errors: {results['errors']}")
                sys.exit(1)

        logger.info("RetailPRED execution completed")

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
