#!/usr/bin/env python3
"""
Backfill Actual Values for 2025 Predictions from time_series_data

This script populates actual values for all 2025 predictions by fetching
from time_series_data table using category_id mappings.
"""

import sqlite3
from pathlib import Path
from collections import defaultdict

DB_PATH = Path(__file__).parent.parent / "data/retailpred.db"

# Category mappings: short_name -> FRED category_id
# Mappings from the categories table in the database
CATEGORY_ID_MAPPING = {
    'automobile_dealers': '441',
    'building_materials': '443',
    'clothing_accessories': '452',  # Fixed: was 448, correct is 452
    'electronics_and_appliances': '4431',
    'food_beverage': '445',  # Fixed: was 447, correct is 445
    'furniture_home_furnishings': '442',
    'gasoline_stations': '448',
    'general_merchandise': '454',
    'health_personal_care': '447',
    'sporting_goods_hobby': '453',  # CRITICAL FIX: was 454, correct is 453
    'total_sales': '4400',

    # Full category key mappings
    'building_material_and_garden_equipment': '443',
    'clothing_and_clothing_accessories_stores': '452',
    'electronics_and_appliance_stores': '4431',
    'food_and_beverage_stores': '445',
    'furniture_and_home_furnishings_stores': '442',
    'general_merchandise_stores': '454',
    'health_and_personal_care_stores': '447',
    'sporting_goods_hobby_and_musical_instrument_stores': '453',
}


def extract_category_from_model_name(model_name: str) -> str:
    """Extract category key from model name"""
    # Remove model type suffix
    for suffix in ['_RandomForest_model', '_LGBM_model', '_SeasonalNaive_model',
                   '_AutoARIMA_model', '_AutoETS_model', '_PatchTST_model', '_TimesNet_model']:
        if model_name.endswith(suffix):
            return model_name[:-len(suffix)]
    return model_name


def get_category_id(category_key: str) -> str:
    """Get FRED category_id for a category key"""
    return CATEGORY_ID_MAPPING.get(category_key, None)


def backfill_actual_values():
    """Backfill actual values for all 2025 predictions"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("=" * 80)
    print("BACKFILLING ACTUAL VALUES FOR 2025 PREDICTIONS")
    print("=" * 80)

    # Get all 2025 predictions without actual values
    cursor.execute("""
        SELECT
            model_name,
            prediction_date,
            predicted_value
        FROM prediction_log
        WHERE prediction_date >= '2025-01-01'
        AND prediction_date <= '2025-12-31'
        AND actual_value IS NULL
        ORDER BY model_name, prediction_date
    """)

    predictions_to_update = cursor.fetchall()

    print(f"\nFound {len(predictions_to_update)} predictions without actual values")

    if len(predictions_to_update) == 0:
        print("✅ All 2025 predictions already have actual values!")
        conn.close()
        return

    # Group by model
    by_model = defaultdict(list)
    for pred in predictions_to_update:
        model_name = pred[0]  # First column is model_name (changed from id)
        by_model[model_name].append(pred)

    print(f"Affected models: {len(by_model)}")

    # Process each model
    total_updated = 0
    total_not_found = 0

    for model_name, predictions in by_model.items():
        print(f"\nProcessing {model_name}...")

        # Extract category key from model name
        category_key = extract_category_from_model_name(model_name)
        category_id = get_category_id(category_key)

        if not category_id:
            print(f"  ⚠️  No category_id mapping found for {category_key}")
            total_not_found += len(predictions)
            continue

        # Update each prediction
        updated = 0
        not_found = 0

        for pred_model_name, pred_date, pred_value in predictions:
            # Get actual value from time_series_data
            # Try exact match first, then try any day in the same week
            cursor.execute("""
                SELECT value
                FROM time_series_data
                WHERE category_id = ?
                AND date = ?
                AND data_type = 'retail_sales'
                LIMIT 1
            """, (category_id, pred_date))

            result = cursor.fetchone()

            # If no exact match, try to find any value within the same week
            if not result or not result[0]:
                # Get the week's date range (pred_date is Wednesday, so Monday is pred_date - 2 days)
                from datetime import datetime, timedelta
                pred_dt = datetime.strptime(pred_date, '%Y-%m-%d')
                week_start = (pred_dt - timedelta(days=2)).strftime('%Y-%m-%d')  # Monday
                week_end = (pred_dt + timedelta(days=4)).strftime('%Y-%m-%d')    # Sunday

                cursor.execute("""
                    SELECT value
                    FROM time_series_data
                    WHERE category_id = ?
                    AND date >= ?
                    AND date <= ?
                    AND data_type = 'retail_sales'
                    ORDER BY ABS(strftime('%s', date) - strftime('%s', ?)) ASC
                    LIMIT 1
                """, (category_id, week_start, week_end, pred_date))

                result = cursor.fetchone()

            if result and result[0]:
                # Update the prediction using model_name and prediction_date
                cursor.execute("""
                    UPDATE prediction_log
                    SET actual_value = ?
                    WHERE model_name = ?
                    AND prediction_date = ?
                """, (result[0], model_name, pred_date))

                updated += 1
            else:
                not_found += 1

        conn.commit()

        if updated > 0:
            print(f"  ✅ Updated {updated} predictions")
        if not_found > 0:
            print(f"  ⚠️  {not_found} predictions not found in time_series_data")

        total_updated += updated
        total_not_found += not_found

    # Verify results
    cursor.execute("""
        SELECT COUNT(*)
        FROM prediction_log
        WHERE prediction_date >= '2025-01-01'
        AND prediction_date <= '2025-12-31'
        AND actual_value IS NOT NULL
    """)

    validated_count = cursor.fetchone()[0]

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Total 2025 predictions: {len(predictions_to_update)}")
    print(f"Validated: {validated_count} ({validated_count/len(predictions_to_update)*100:.1f}%)")
    print(f"Not found in time_series_data: {total_not_found}")

    if validated_count > 0:
        print("\n✅ Backfill complete!")
    else:
        print("\n⚠️  No predictions could be validated (check category_id mappings)")

    conn.close()


if __name__ == "__main__":
    backfill_actual_values()
