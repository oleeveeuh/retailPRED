"""
Migration Script: Convert Parquet Data to SQLite
One-time migration script to populate SQLite database with existing data
"""

import os
import sys
import pandas as pd
import sqlite3
from pathlib import Path
from datetime import datetime
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlite.sqlite_loader import SQLiteLoader
from etl.build_category_dataset import CategoryDatasetBuilder

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SQLiteMigrator:
    """Migrate existing parquet data to SQLite database"""

    def __init__(self, db_path: str = None, data_dir: str = None):
        self.db_path = db_path or os.path.join(
            Path(__file__).parent.parent.parent,
            "data",
            "retailpred.db"
        )
        self.data_dir = data_dir or os.path.join(
            Path(__file__).parent.parent.parent,
            "data",
            "processed"
        )
        self.loader = SQLiteLoader(self.db_path)

        # Define category mapping based on actual data structure
        self.mrts_categories = {
            '447': {  # Health_Personal_Care (from unique_id: 447US)
                'name': 'Health_Personal_Care',
                'description': 'Health and Personal Care Stores'
            },
            '4400': {  # Total_Retail_Sales (from unique_id: 4400US)
                'series_id': 'MRTSSM44000USS',
                'name': 'Total_Retail_Sales',
                'description': 'Total Retail Sales (Excluding Motor Vehicle and Parts Dealers)'
            },
            '441': {  # Automobile_Dealers (from unique_id: 441US)
                'series_id': 'MRTSSM441USS',
                'name': 'Automobile_Dealers',
                'description': 'Automobile Dealers'
            },
            '442': {  # Furniture_Home_Furnishings (from unique_id: 442US)
                'series_id': 'MRTSSM442USS',
                'name': 'Furniture_Home_Furnishings',
                'description': 'Furniture and Home Furnishings Stores'
            },
            '443': {  # Building_Materials_Garden (from unique_id: 443US)
                'series_id': 'MRTSSM443USS',
                'name': 'Building_Materials_Garden',
                'description': 'Building Material and Garden Equipment and Supplies Dealers'
            },
            '445': {  # Food_Beverage_Stores (from unique_id: 445US)
                'series_id': 'MRTSSM445USS',
                'name': 'Food_Beverage_Stores',
                'description': 'Food Services and Drinking Places'
            },
            '448': {  # Gasoline_Stations (from unique_id: 448US)
                'series_id': 'MRTSSM448USS',
                'name': 'Gasoline_Stations',
                'description': 'Gasoline Stations'
            },
            '452': {  # Clothing_Accessories (from unique_id: 452US)
                'series_id': 'MRTSSM452USS',
                'name': 'Clothing_Accessories',
                'description': 'Clothing and Clothing Accessories Stores'
            },
            '453': {  # Sporting_Goods_Hobby (from unique_id: 453US)
                'series_id': 'MRTSSM453USS',
                'name': 'Sporting_Goods_Hobby',
                'description': 'Sporting Goods, Hobby, Book, and Music Stores'
            },
            '454': {  # General_Merchandise (from unique_id: 454US)
                'series_id': 'MRTSSM454USS',
                'name': 'General_Merchandise',
                'description': 'General Merchandise Stores'
            },
            '4431': {  # Electronics_and_Appliances (from unique_id: 4431US)
                'series_id': 'MRTSSM44X72USS',
                'name': 'Electronics_and_Appliances',
                'description': 'Electronics and Appliance Stores'
            },
            '456': {  # Nonstore_Retailers (not in current files but for completeness)
                'series_id': 'MRTSSM4521USS',
                'name': 'Nonstore_Retailers',
                'description': 'Nonstore Retailers'
            }
        }

    def migrate_all_data(self):
        """Complete migration of all existing data"""
        logger.info("Starting complete data migration to SQLite...")

        try:
            # Step 1: Migrate categories metadata
            self.migrate_categories()

            # Step 2: Migrate all parquet files
            self.migrate_parquet_files()

            # Step 3: Update metadata and cache status
            self.update_migration_metadata()

            # Step 4: Verify migration
            self.verify_migration()

            logger.info("✅ Migration completed successfully!")

        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            raise

    def migrate_categories(self):
        """Migrate categories metadata from predefined mapping"""
        logger.info("Migrating categories metadata...")

        categories = self.mrts_categories

        for category_id, category_info in categories.items():
            self.loader.add_category(
                category_id=category_id,
                category_name=category_info['name'],
                description=category_info.get('description', ''),
                mrts_series_id=category_info.get('series_id', '')
            )

        logger.info(f"✅ Migrated {len(categories)} categories")

    def migrate_parquet_files(self):
        """Migrate all parquet files from data/processed directory"""
        logger.info("Migrating parquet files...")

        if not os.path.exists(self.data_dir):
            logger.warning(f"Data directory not found: {self.data_dir}")
            return

        parquet_files = [f for f in os.listdir(self.data_dir) if f.endswith('.parquet')]
        logger.info(f"Found {len(parquet_files)} parquet files to migrate")

        total_records = 0

        for parquet_file in parquet_files:
            try:
                file_path = os.path.join(self.data_dir, parquet_file)
                category_name = parquet_file.replace('.parquet', '')

                logger.info(f"Migrating {category_name}...")

                # Read parquet file
                df = pd.read_parquet(file_path)

                if df.empty:
                    logger.warning(f"Empty parquet file: {parquet_file}")
                    continue

                # Get unique_id from the data (should match our category mapping)
                if 'unique_id' not in df.columns:
                    logger.warning(f"No unique_id column in {parquet_file}")
                    continue

                # Get the category ID from unique_id (remove 'US' suffix if present)
                sample_unique_id = df['unique_id'].iloc[0]
                if sample_unique_id.endswith('US'):
                    category_id = sample_unique_id[:-2]  # Remove 'US' suffix
                else:
                    category_id = sample_unique_id

                # Verify this category exists in our mapping
                if category_id not in self.mrts_categories:
                    logger.warning(f"Unknown category ID: {category_id} in {parquet_file}")
                    continue

                logger.info(f"Migrating {category_name} (ID: {category_id})")

                # Migrate retail sales data (y column)
                if 'y' in df.columns:
                    retail_df = df[['ds', 'y']].copy()
                    retail_df.columns = ['date', 'value']
                    self.loader.add_time_series_data(
                        retail_df, category_id, 'retail_sales', 'MRTS'
                    )
                    total_records += len(retail_df)

                # Migrate basic exogenous features (skip derived features for now)
                basic_features = ['cpi', 'interest_rates', 'unemployment', 'consumer_sentiment',
                                'money_supply', 'industrial_production', 'consumer_spending']

                for col in basic_features:
                    if col in df.columns:
                        feature_df = df[['ds', col]].copy()
                        feature_df.columns = ['date', 'value']

                        # Remove null values to avoid NOT NULL constraint violation
                        feature_df = feature_df.dropna(subset=['value'])

                        if not feature_df.empty:
                            # Determine data type and source
                            data_type = self._map_column_to_data_type(col)
                            source = self._map_column_to_source(col)

                            self.loader.add_time_series_data(
                                feature_df, category_id, data_type, source
                            )
                            total_records += len(feature_df)
                        else:
                            logger.warning(f"  ⚠️  Skipping {col} - all values are null")

                # Store derived features separately (for future use)
                derived_feature_columns = [col for col in df.columns if col not in
                                        ['unique_id', 'ds', 'y'] + basic_features]

                if derived_feature_columns:
                    logger.info(f"  📊 Skipping {len(derived_feature_columns)} derived features (will recompute)")

                logger.info(f"✅ Migrated {category_name}: {len(df)} records ({len(basic_features)} features)")

            except Exception as e:
                logger.error(f"❌ Failed to migrate {parquet_file}: {e}")
                continue

        logger.info(f"✅ Total records migrated: {total_records}")

    def _find_category_id_by_name(self, category_name: str) -> str:
        """Find category ID by matching category name"""
        categories = self.mrts_categories

        for category_id, category_info in categories.items():
            if category_info['name'] == category_name:
                return category_id

        return None

    def _map_column_to_data_type(self, column_name: str) -> str:
        """Map column name to standardized data type"""
        mapping = {
            'cpi': 'cpi',
            'interest_rates': 'interest_rate',
            'unemployment': 'unemployment_rate',
            'consumer_sentiment': 'consumer_sentiment',
            'money_supply': 'money_supply',
            'industrial_production': 'industrial_production',
            'consumer_spending': 'consumer_spending',
        }

        return mapping.get(column_name.lower(), column_name.lower())

    def _map_column_to_source(self, column_name: str) -> str:
        """Map column name to data source"""
        if column_name == 'y':
            return 'MRTS'
        elif column_name in ['cpi', 'interest_rates', 'unemployment', 'consumer_sentiment',
                           'money_supply', 'industrial_production', 'consumer_spending']:
            return 'FRED'
        elif column_name in ['sp500', 'oil_prices']:
            return 'YAHOO'
        else:
            return 'DERIVED'

    def update_migration_metadata(self):
        """Update metadata after migration"""
        logger.info("Updating migration metadata...")

        # Set database version
        self.loader.set_metadata('database_version', '1.0')

        # Update category count
        categories_df = self.loader.get_all_categories()
        self.loader.set_metadata('total_categories', str(len(categories_df)))

        # Update date range
        start_date, end_date = self.loader.get_date_range()
        if start_date and end_date:
            self.loader.set_metadata('date_range_start', start_date)
            self.loader.set_metadata('date_range_end', end_date)

        # Set migration timestamp
        self.loader.set_metadata('migration_date', datetime.now().isoformat())

        # Update cache status
        self.loader.update_cache_status('time_series_data', True)

        logger.info("✅ Migration metadata updated")

    def verify_migration(self):
        """Verify migration integrity"""
        logger.info("Verifying migration integrity...")

        # Get database statistics
        stats = self.loader.get_database_stats()

        logger.info("📊 Database Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

        # Check that we have categories
        if stats.get('categories_rows', 0) == 0:
            raise ValueError("No categories found in database!")

        # Check that we have time series data
        if stats.get('time_series_data_rows', 0) == 0:
            raise ValueError("No time series data found in database!")

        # Verify date range makes sense
        if 'date_range' in stats and stats['date_range'][0]:
            start_date, end_date = stats['date_range']
            logger.info(f"📅 Data range: {start_date} to {end_date}")

            # Basic sanity check
            try:
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                if start_dt.year < 1990 or end_dt.year < 1990:
                    logger.warning("⚠️  Date range seems unusual")
            except:
                logger.warning("⚠️  Could not parse date range")

        logger.info("✅ Migration verification completed")

    def create_backup(self):
        """Create backup of migrated database"""
        backup_path = self.loader.backup_database()
        logger.info(f"📦 Database backup created: {backup_path}")
        return backup_path

def main():
    """Main migration function"""
    logger.info("🚀 Starting SQLite Migration for RetailPRED")

    # Create migrator
    migrator = SQLiteMigrator()

    try:
        # Run complete migration
        migrator.migrate_all_data()

        # Create backup
        backup_path = migrator.create_backup()

        # Display final statistics
        stats = migrator.loader.get_database_stats()

        logger.info("🎉 Migration completed successfully!")
        logger.info(f"📁 Database location: {migrator.db_path}")
        logger.info(f"📦 Backup location: {backup_path}")
        logger.info(f"📊 Database size: {stats.get('file_size_mb', 0):.2f} MB")
        logger.info(f"📈 Total records: {stats.get('time_series_data_rows', 0):,}")
        logger.info(f"🏪 Categories: {stats.get('categories_rows', 0):,}")

    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise

    finally:
        migrator.loader.close()

if __name__ == "__main__":
    main()