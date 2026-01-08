"""
Update Historical Data from Multi-Resolution CSV Files

This script updates the time_series_data table in the database with the latest
data from the multi-resolution CSV files to ensure consistency.
"""

import sys
from pathlib import Path
import logging
import sqlite3
import pandas as pd
from datetime import datetime

# Add app directory to path
app_path = Path(__file__).parent.parent
sys.path.insert(0, str(app_path))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def update_historical_data():
    """Update time_series_data table with CSV data"""

    db_path = "/Users/olivialiau/retailPRED/data/retailpred.db"
    csv_base_path = "/Users/olivialiau/retailPRED/project_root/data_multi_resolution"

    # CSV file mapping (key maps to both csv_files and category_mapping)
    csv_files = {
        "total_sales": "retail_total_sales_multi_resolution.csv",
        "automobile_dealers": "retail_automobile_dealers_multi_resolution.csv",
        "building_materials": "retail_building_material_and_garden_equipment_multi_resolution.csv",
        "clothing_accessories": "retail_clothing_and_clothing_accessories_stores_multi_resolution.csv",
        "electronics_and_appliances": "retail_electronics_and_appliance_stores_multi_resolution.csv",
        "food_beverage": "retail_food_and_beverage_stores_multi_resolution.csv",
        "furniture_home_furnishings": "retail_furniture_and_home_furnishings_stores_multi_resolution.csv",
        "gasoline_stations": "retail_gasoline_stations_multi_resolution.csv",
        "general_merchandise": "retail_general_merchandise_stores_multi_resolution.csv",
        "health_personal_care": "retail_health_and_personal_care_stores_multi_resolution.csv",
        "sporting_goods_hobby": "retail_sporting_goods_hobby_and_musical_instrument_stores_multi_resolution.csv",
    }

    # Category ID mapping (from categories table)
    category_mapping = {
        "total_sales": "4400",
        "automobile_dealers": "441",
        "building_materials": "443",
        "clothing_accessories": "452",
        "electronics_and_appliances": "4431",
        "food_beverage": "445",
        "furniture_home_furnishings": "442",
        "gasoline_stations": "448",
        "general_merchandise": "454",
        "health_personal_care": "447",
        "sporting_goods_hobby": "453",
    }

    logger.info("=" * 80)
    logger.info("Updating Historical Data from CSV Files")
    logger.info("=" * 80)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    total_updated = 0
    total_inserted = 0
    total_skipped = 0

    for category_key, csv_filename in csv_files.items():
        category_id = category_mapping.get(category_key)
        if not category_id:
            logger.warning(f"No category_id found for {category_key}, skipping")
            continue

        csv_path = Path(csv_base_path) / csv_filename
        if not csv_path.exists():
            logger.warning(f"CSV file not found: {csv_path}, skipping")
            continue

        logger.info(f"\nProcessing {category_key} (category_id: {category_id})")

        try:
            # Read CSV data
            df = pd.read_csv(str(csv_path))
            df['index'] = pd.to_datetime(df['index'])

            # Get data from 2025 onwards
            df_recent = df[df['index'] >= '2025-01-01'].copy()

            logger.info(f"  Found {len(df_recent)} records from 2025+ in CSV")

            updated = 0
            inserted = 0
            skipped = 0

            for _, row in df_recent.iterrows():
                date_str = row['index'].strftime('%Y-%m-%d')
                value = float(row['y'])

                # Check if record exists
                cursor.execute("""
                    SELECT id FROM time_series_data
                    WHERE category_id = ? AND date = ? AND data_type = 'retail_sales'
                """, (category_id, date_str))

                existing = cursor.fetchone()

                if existing:
                    # Update existing record
                    cursor.execute("""
                        UPDATE time_series_data
                        SET value = ?, source = 'multi_resolution_csv'
                        WHERE id = ?
                    """, (value, existing[0]))
                    updated += 1
                else:
                    # Insert new record
                    cursor.execute("""
                        INSERT INTO time_series_data (category_id, date, value, data_type, source, created_at)
                        VALUES (?, ?, ?, 'retail_sales', 'multi_resolution_csv', ?)
                    """, (category_id, date_str, value, datetime.now().isoformat()))
                    inserted += 1

            logger.info(f"  Updated: {updated}, Inserted: {inserted}")
            total_updated += updated
            total_inserted += inserted

        except Exception as e:
            logger.error(f"  Error processing {csv_filename}: {e}")
            continue

    # Commit changes
    conn.commit()

    # Verify results
    logger.info("\n" + "=" * 80)
    logger.info("UPDATE SUMMARY")
    logger.info("=" * 80)

    cursor.execute("""
        SELECT COUNT(*) FROM time_series_data
        WHERE date >= '2025-01-01' AND data_type = 'retail_sales'
    """)
    total_records = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(DISTINCT date) FROM time_series_data
        WHERE date >= '2025-01-01' AND data_type = 'retail_sales'
    """)
    unique_dates = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(DISTINCT category_id) FROM time_series_data
        WHERE date >= '2025-01-01' AND data_type = 'retail_sales'
    """)
    unique_categories = cursor.fetchone()[0]

    logger.info(f"✓ Records updated: {total_updated}")
    logger.info(f"✓ Records inserted: {total_inserted}")
    logger.info(f"✓ Total 2025+ records in database: {total_records}")
    logger.info(f"✓ Unique dates: {unique_dates}")
    logger.info(f"✓ Unique categories: {unique_categories}")

    # Sample verification
    logger.info("\n" + "=" * 80)
    logger.info("VERIFICATION SAMPLE (Total Retail Sales)")
    logger.info("=" * 80)

    cursor.execute("""
        SELECT date, value
        FROM time_series_data
        WHERE category_id = '4400' AND data_type = 'retail_sales'
          AND date >= '2025-11-28'
        ORDER BY date
    """)

    print(f"\n{'Date':<12} {'Database Value':>15}")
    print("-" * 30)
    for date, value in cursor.fetchall():
        print(f"{date:<12} ${value:>13,.2f}")

    # Compare with CSV
    logger.info("\nCSV Values for Same Dates:")
    df_sample = pd.read_csv(f"{csv_base_path}/retail_total_sales_multi_resolution.csv")
    df_sample['index'] = pd.to_datetime(df_sample['index'])
    df_sample = df_sample[df_sample['index'] >= '2025-11-28']
    for _, row in df_sample.iterrows():
        print(f"{row['index'].strftime('%Y-%m-%d'):<12} ${row['y']:>13,.2f}")

    conn.close()
    logger.info("\n" + "=" * 80)
    logger.info("✓ Update Complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    update_historical_data()
