"""
SQLite Database Loader for RetailPRED
Provides high-performance data loading and caching for time series forecasting
"""

import sqlite3
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

class SQLiteLoader:
    """High-performance SQLite data loader for RetailPRED pipeline"""

    def __init__(self, db_path: str = None, config: dict = None):
        """
        Initialize SQLite loader

        Args:
            db_path: Path to SQLite database file
            config: Configuration dictionary
        """
        self.db_path = db_path or os.path.join(
            Path(__file__).parent.parent.parent,
            "data",
            "retailpred.db"
        )
        self.config = config or {}
        self.connection = None
        self.logger = logging.getLogger(__name__)

        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        # Initialize database
        self.setup_database()

    def setup_database(self):
        """Create database schema and indexes if they don't exist"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row  # Enable dict-like access

            # Enable WAL mode for better concurrent access
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=NORMAL")
            self.connection.execute("PRAGMA cache_size=10000")

            # Create tables
            self._create_tables()
            self._create_indexes()
            self._initialize_metadata()

            self.logger.info(f"SQLite database initialized: {self.db_path}")

        except Exception as e:
            self.logger.error(f"Failed to initialize SQLite database: {e}")
            raise

    def _create_tables(self):
        """Create all database tables"""

        # Metadata table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Categories table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS categories (
                category_id TEXT PRIMARY KEY,
                category_name TEXT NOT NULL,
                description TEXT,
                mrts_series_id TEXT UNIQUE,
                is_active BOOLEAN DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Time series data table (optimized structure)
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS time_series_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category_id TEXT NOT NULL,
                date TEXT NOT NULL,
                value REAL NOT NULL,
                data_type TEXT NOT NULL,
                source TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(category_id, date, data_type)
            )
        """)

        # Derived features table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS derived_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category_id TEXT NOT NULL,
                date TEXT NOT NULL,
                feature_name TEXT NOT NULL,
                feature_value REAL NOT NULL,
                feature_type TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(category_id, date, feature_name)
            )
        """)

        # Cache status table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS cache_status (
                table_name TEXT PRIMARY KEY,
                last_updated DATETIME,
                row_count INTEGER,
                size_mb REAL,
                is_valid BOOLEAN DEFAULT 1,
                checksum TEXT
            )
        """)

        self.connection.commit()

    def _create_indexes(self):
        """Create performance indexes"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_ts_category_date ON time_series_data(category_id, date)",
            "CREATE INDEX IF NOT EXISTS idx_ts_date ON time_series_data(date)",
            "CREATE INDEX IF NOT EXISTS idx_ts_type ON time_series_data(data_type)",
            "CREATE INDEX IF NOT EXISTS idx_features_category_date ON derived_features(category_id, date)",
            "CREATE INDEX IF NOT EXISTS idx_features_name ON derived_features(feature_name)",
            "CREATE INDEX IF NOT EXISTS idx_categories_active ON categories(is_active) WHERE is_active = 1"
        ]

        for index_sql in indexes:
            self.connection.execute(index_sql)

        self.connection.commit()

    def _initialize_metadata(self):
        """Initialize metadata table with default values"""
        metadata_defaults = {
            'database_version': '1.0',
            'total_categories': '0',
            'date_range_start': '1992-01-01',
            'date_range_end': datetime.now().strftime('%Y-%m-%d')
        }

        for key, value in metadata_defaults.items():
            self.connection.execute(
                "INSERT OR IGNORE INTO metadata (key, value) VALUES (?, ?)",
                (key, value)
            )

        self.connection.commit()

    def add_category(self, category_id: str, category_name: str,
                    description: str = None, mrts_series_id: str = None):
        """Add or update a retail category"""
        self.connection.execute("""
            INSERT OR REPLACE INTO categories
            (category_id, category_name, description, mrts_series_id, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (category_id, category_name, description, mrts_series_id))

        self.connection.commit()
        self.logger.info(f"Added/updated category: {category_id} - {category_name}")

    def add_time_series_data(self, df: pd.DataFrame, category_id: str,
                           data_type: str, source: str):
        """
        Add time series data to database

        Args:
            df: DataFrame with columns ['date', 'value']
            category_id: Category identifier
            data_type: Type of data ('retail_sales', 'cpi', 'interest_rate', etc.)
            source: Data source ('MRTS', 'FRED', 'YAHOO')
        """
        if df.empty:
            return

        # Prepare data
        df = df.copy()
        df['category_id'] = category_id
        df['data_type'] = data_type
        df['source'] = source
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

        # Convert value to numeric
        df['value'] = pd.to_numeric(df['value'], errors='coerce')

        # Remove existing records for this category, data type, and date range
        date_list = df['date'].unique().tolist()
        placeholders = ','.join(['?'] * len(date_list))

        self.connection.execute(f"""
            DELETE FROM time_series_data
            WHERE category_id = ? AND data_type = ? AND date IN ({placeholders})
        """, [category_id, data_type] + date_list)

        # Select and rename columns
        df = df[['category_id', 'date', 'value', 'data_type', 'source']]

        # Insert new data
        df.to_sql('time_series_data', self.connection,
                 if_exists='append', index=False, method='multi')

        self.logger.info(f"Added {len(df)} records for {category_id} - {data_type}")

    def get_category_data(self, category_id: str, start_date: str = None,
                         end_date: str = None, data_types: List[str] = None) -> pd.DataFrame:
        """
        Get time series data for a specific category

        Args:
            category_id: Category identifier
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            data_types: List of data types to include

        Returns:
            DataFrame with time series data
        """
        query = """
            SELECT date, data_type, value, source
            FROM time_series_data
            WHERE category_id = ?
        """
        params = [category_id]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        if data_types:
            placeholders = ','.join(['?'] * len(data_types))
            query += f" AND data_type IN ({placeholders})"
            params.extend(data_types)

        query += " ORDER BY date, data_type"

        df = pd.read_sql_query(query, self.connection, params=params)

        if not df.empty:
            # Pivot data for easier use
            df = df.pivot_table(
                index='date',
                columns='data_type',
                values='value',
                aggfunc='first'
            ).reset_index()

        return df

    def get_all_categories(self, active_only: bool = True) -> pd.DataFrame:
        """Get list of all categories"""
        query = "SELECT * FROM categories"
        if active_only:
            query += " WHERE is_active = 1"
        query += " ORDER BY category_id"

        return pd.read_sql_query(query, self.connection)

    def add_derived_features(self, df: pd.DataFrame, category_id: str):
        """
        Add derived features for a category

        Args:
            df: DataFrame with columns ['date', 'feature_name', 'feature_value', 'feature_type']
            category_id: Category identifier
        """
        if df.empty:
            return

        df = df.copy()
        df['category_id'] = category_id
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

        df = df[['category_id', 'date', 'feature_name', 'feature_value', 'feature_type']]

        df.to_sql('derived_features', self.connection,
                 if_exists='append', index=False, method='multi')

        self.logger.info(f"Added {len(df)} derived features for {category_id}")

    def get_derived_features(self, category_id: str, start_date: str = None,
                           end_date: str = None, feature_names: List[str] = None) -> pd.DataFrame:
        """Get derived features for a category"""
        query = """
            SELECT date, feature_name, feature_value, feature_type
            FROM derived_features
            WHERE category_id = ?
        """
        params = [category_id]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        if feature_names:
            placeholders = ','.join(['?'] * len(feature_names))
            query += f" AND feature_name IN ({placeholders})"
            params.extend(feature_names)

        query += " ORDER BY date, feature_name"

        return pd.read_sql_query(query, self.connection, params=params)

    def is_data_fresh(self, max_age_days: int = 1) -> bool:
        """Check if data is fresh (updated within max_age_days)"""
        cutoff_date = (datetime.now() - timedelta(days=max_age_days)).strftime('%Y-%m-%d')

        result = self.connection.execute("""
            SELECT COUNT(*) as count FROM cache_status
            WHERE table_name = 'time_series_data'
            AND last_updated >= ? AND is_valid = 1
        """, (cutoff_date,)).fetchone()

        return result['count'] > 0

    def update_cache_status(self, table_name: str, is_valid: bool = True):
        """Update cache status for a table"""
        # Get table statistics
        result = self.connection.execute(f"SELECT COUNT(*) as count FROM {table_name}").fetchone()
        row_count = result['count'] if result else 0

        # Get file size
        db_size_mb = os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0

        self.connection.execute("""
            INSERT OR REPLACE INTO cache_status
            (table_name, last_updated, row_count, size_mb, is_valid)
            VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?)
        """, (table_name, row_count, db_size_mb, is_valid))

        self.connection.commit()

    def get_metadata(self, key: str) -> Optional[str]:
        """Get metadata value"""
        result = self.connection.execute(
            "SELECT value FROM metadata WHERE key = ?", (key,)
        ).fetchone()

        return result['value'] if result else None

    def set_metadata(self, key: str, value: str):
        """Set metadata value"""
        self.connection.execute(
            "INSERT OR REPLACE INTO metadata (key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
            (key, value)
        )
        self.connection.commit()

    def get_date_range(self, category_id: str = None) -> Tuple[str, str]:
        """Get date range of data in database"""
        if category_id:
            result = self.connection.execute("""
                SELECT MIN(date) as min_date, MAX(date) as max_date
                FROM time_series_data
                WHERE category_id = ?
            """, (category_id,)).fetchone()
        else:
            result = self.connection.execute("""
                SELECT MIN(date) as min_date, MAX(date) as max_date
                FROM time_series_data
            """).fetchone()

        if result and result['min_date']:
            return result['min_date'], result['max_date']
        return None, None

    def backup_database(self, backup_path: str = None):
        """Create backup of database"""
        if not backup_path:
            backup_path = self.db_path.replace('.db', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.db')

        # Close current connection
        if self.connection:
            self.connection.close()

        # Create backup
        import shutil
        shutil.copy2(self.db_path, backup_path)

        # Reopen connection
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row

        self.logger.info(f"Database backed up to: {backup_path}")
        return backup_path

    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        stats = {}

        # Table sizes
        for table in ['categories', 'time_series_data', 'derived_features']:
            result = self.connection.execute(f"SELECT COUNT(*) as count FROM {table}").fetchone()
            stats[f'{table}_rows'] = result['count'] if result else 0

        # File size
        if os.path.exists(self.db_path):
            stats['file_size_mb'] = os.path.getsize(self.db_path) / (1024 * 1024)

        # Date range
        stats['date_range'] = self.get_date_range()

        # Metadata
        stats['database_version'] = self.get_metadata('database_version')
        stats['total_categories'] = self.get_metadata('total_categories')

        return stats

    def save_forecast_results(self, results: Dict) -> bool:
        """
        Save forecast results to SQLite database

        Args:
            results: Dictionary containing forecast results for multiple targets

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.connection:
                self.logger.error("Database connection not available")
                return False

            cursor = self.connection.cursor()

            # Create forecast results table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS forecast_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    target_name TEXT NOT NULL,
                    forecast_type TEXT NOT NULL,
                    horizon INTEGER,
                    run_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    predictions TEXT,
                    confidence_intervals TEXT,
                    performance_metrics TEXT,
                    actual_values TEXT,
                    backtest_dates TEXT,
                    model_results TEXT,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create index for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_forecast_target
                ON forecast_results(target_name, run_timestamp)
            """)

            # Save results for each target
            success_count = 0
            total_count = 0

            for target_name, target_results in results.items():
                if target_name == 'error':
                    continue

                total_count += 1

                try:
                    # Determine forecast type
                    forecast_type = 'prediction'
                    if 'backtest_dates' in target_results:
                        forecast_type = 'backtest'
                    elif 'horizon' in target_results:
                        forecast_type = 'forecast'

                    # Extract data components
                    horizon = target_results.get('horizon', None)
                    predictions = target_results.get('predictions', [])
                    confidence_intervals = target_results.get('confidence_intervals', {})
                    performance_metrics = target_results.get('performance_metrics', {})
                    actual_values = target_results.get('actuals', [])
                    backtest_dates = target_results.get('backtest_dates', [])
                    model_results = target_results.get('model_results', {})

                    # Convert to JSON strings for storage
                    import json

                    # Helper function to safely convert data to JSON string
                    def safe_json_serialize(data):
                        """Safely serialize data to JSON string, handling numpy arrays"""
                        if data is None or (isinstance(data, str) and not data.strip()):
                            return None
                        try:
                            # Convert numpy arrays to lists first
                            if hasattr(data, 'tolist'):
                                data = data.tolist()
                            # Handle other non-serializable objects
                            elif hasattr(data, '__dict__'):
                                data = str(data)
                            # Serialize to JSON string
                            return json.dumps(data, default=str)
                        except (TypeError, ValueError) as e:
                            self.logger.warning(f"JSON serialization error: {e}, using string conversion")
                            return str(data)

                    cursor.execute("""
                        INSERT INTO forecast_results (
                            target_name, forecast_type, horizon, predictions,
                            confidence_intervals, performance_metrics, actual_values,
                            backtest_dates, model_results, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        target_name,
                        forecast_type,
                        horizon,
                        safe_json_serialize(predictions),
                        safe_json_serialize(confidence_intervals),
                        safe_json_serialize(performance_metrics),
                        safe_json_serialize(actual_values),
                        safe_json_serialize(backtest_dates),
                        safe_json_serialize(model_results),
                        json.dumps({
                            'run_id': datetime.now().isoformat(),
                            'total_results': len(results)
                        })
                    ))

                    success_count += 1
                    self.logger.debug(f"Saved forecast results for {target_name}")

                except Exception as e:
                    self.logger.warning(f"Failed to save results for {target_name}: {e}")
                    continue

            # Commit transaction
            self.connection.commit()

            self.logger.info(f"Successfully saved {success_count}/{total_count} forecast results to SQLite")

            # Update metadata
            self.set_metadata('last_forecast_run', datetime.now().isoformat())
            self.set_metadata('total_forecasts_saved', str(success_count))

            return success_count > 0

        except Exception as e:
            self.logger.error(f"Error saving forecast results: {e}")
            if self.connection:
                self.connection.rollback()
            return False

    def get_forecast_results(self, target_name: str = None,
                           forecast_type: str = None,
                           limit: int = 100) -> pd.DataFrame:
        """
        Retrieve forecast results from database

        Args:
            target_name: Filter by specific target name (optional)
            forecast_type: Filter by forecast type ('prediction', 'backtest', 'forecast')
            limit: Maximum number of results to return

        Returns:
            DataFrame with forecast results
        """
        try:
            if not self.connection:
                self.logger.error("Database connection not available")
                return pd.DataFrame()

            query = "SELECT * FROM forecast_results WHERE 1=1"
            params = []

            if target_name:
                query += " AND target_name = ?"
                params.append(target_name)

            if forecast_type:
                query += " AND forecast_type = ?"
                params.append(forecast_type)

            query += " ORDER BY run_timestamp DESC LIMIT ?"
            params.append(limit)

            df = pd.read_sql_query(query, self.connection, params=params)

            # Parse JSON columns back to objects
            import json

            json_columns = ['predictions', 'confidence_intervals', 'performance_metrics',
                          'actual_values', 'backtest_dates', 'model_results', 'metadata']

            for col in json_columns:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: json.loads(x) if x else None)

            return df

        except Exception as e:
            self.logger.error(f"Error retrieving forecast results: {e}")
            return pd.DataFrame()

    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()