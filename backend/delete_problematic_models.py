#!/usr/bin/env python3
"""
Delete Problematic Models from Database

These 4 models have a 19% systematic over-prediction bias:
1. furniture_and_home_furnishings_stores_lgbm_model (25.19% MAPE)
2. furniture_and_home_furnishings_stores_randomforest_model (22.07% MAPE)
3. general_merchandise_stores_lgbm_model (25.66% MAPE)
4. sporting_goods_hobby_and_musical_instrument_stores_lgbm_model (25.59% MAPE)

Better alternatives exist for all 3 categories:
- Furniture: furniture_home_furnishings_AutoARIMA_model (3.74% MAPE)
- General Merchandise: general_merchandise_TimesNet_model (3.19% MAPE)
- Sporting Goods: sporting_goods_hobby_AutoARIMA_model (3.68% MAPE)
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "retailpred.db"

PROBLEMATIC_MODELS = [
    'furniture_and_home_furnishings_stores_lgbm_model',
    'furniture_and_home_furnishings_stores_randomforest_model',
    'general_merchandise_stores_lgbm_model',
    'sporting_goods_hobby_and_musical_instrument_stores_lgbm_model',
]


def delete_problematic_models():
    """Delete problematic models and their predictions from database"""

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Count predictions before deletion
    cursor.execute("SELECT COUNT(*) FROM prediction_log")
    total_before = cursor.fetchone()[0]

    # Delete predictions for each problematic model
    for model_name in PROBLEMATIC_MODELS:
        cursor.execute("SELECT COUNT(*) FROM prediction_log WHERE model_name = ?", (model_name,))
        count = cursor.fetchone()[0]

        print(f"Deleting {count} predictions for {model_name}...")

        cursor.execute("DELETE FROM prediction_log WHERE model_name = ?", (model_name,))

    # Delete from model_metadata
    for model_name in PROBLEMATIC_MODELS:
        cursor.execute("DELETE FROM model_metadata WHERE model_name = ?", (model_name,))
        print(f"Deleted metadata for {model_name}")

    # Commit changes
    conn.commit()

    # Count predictions after deletion
    cursor.execute("SELECT COUNT(*) FROM prediction_log")
    total_after = cursor.fetchone()[0]

    deleted = total_before - total_after

    print(f"\n✅ Deleted {deleted} predictions total")
    print(f"✅ Remaining predictions: {total_after}")

    # Verify deletion
    cursor.execute("""
        SELECT model_name, COUNT(*) as count
        FROM prediction_log
        WHERE model_name IN (
            'furniture_and_home_furnishings_stores_lgbm_model',
            'furniture_and_home_furnishings_stores_randomforest_model',
            'general_merchandise_stores_lgbm_model',
            'sporting_goods_hobby_and_musical_instrument_stores_lgbm_model'
        )
        GROUP BY model_name
    """)

    remaining = cursor.fetchall()
    if remaining:
        print(f"\n⚠️  WARNING: Some predictions still remain:")
        for model, count in remaining:
            print(f"  {model}: {count}")
    else:
        print(f"\n✅ All problematic models successfully removed")

    conn.close()


if __name__ == "__main__":
    print("=" * 80)
    print("Deleting Problematic Models from Database")
    print("=" * 80)
    print(f"\nProblematic models to delete:")
    for model in PROBLEMATIC_MODELS:
        print(f"  - {model}")
    print()

    # Confirm
    response = input("This will delete these models and all their predictions. Continue? (yes/no): ")
    if response.lower() == 'yes':
        delete_problematic_models()
    else:
        print("Aborted.")
