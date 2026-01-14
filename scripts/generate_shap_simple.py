#!/usr/bin/env python3
"""
Generate Mock SHAP Values for Demo Data - Simple Version

Generates mock SHAP values without requiring numpy/pandas
"""

import sys
import sqlite3
import json
import random
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = Path('data/retailpred.db')

def generate_simple_shap():
    """Generate simple mock SHAP values for all predictions"""
    logger.info("\n" + "="*80)
    logger.info("GENERATING MOCK SHAP VALUES FOR DEMO DATA")
    logger.info("="*80)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    conn.isolation_level = None  # Enable autocommit mode

    # Get all tree-based model predictions without proper SHAP values
    cursor.execute("""
        SELECT rowid, model_name FROM prediction_log
        WHERE (model_name LIKE '%lgbm%' OR model_name LIKE '%randomforest%')
        AND (shap_values IS NULL OR shap_values = 'null')
        ORDER BY model_name, prediction_date DESC
    """)

    rows = cursor.fetchall()
    logger.info(f"Found {len(rows)} predictions without SHAP values")

    if len(rows) == 0:
        logger.info("All predictions already have SHAP values!")
        conn.close()
        return 0

    # Common time series features
    common_features = [
        'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7', 'lag_8',
        'lag_9', 'lag_10', 'lag_11', 'lag_12',
        'rolling_mean_4', 'rolling_mean_12', 'rolling_std_4',
        'month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6',
        'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12',
        'quarter_1', 'quarter_2', 'quarter_3', 'quarter_4',
        'trend', 'seasonal'
    ]

    updated_count = 0

    for row_id, model_name in rows:
        # Generate realistic SHAP values
        shap_dict = {}

        # Lag features - highest importance
        for i in range(1, 13):
            lag_name = f'lag_{i}'
            # Decreasing importance
            importance = 45.0 / (i ** 0.7)
            shap_dict[lag_name] = round(importance + random.uniform(-2, 2), 2)

        # Rolling features - moderate importance
        shap_dict['rolling_mean_4'] = round(random.uniform(8, 12), 2)
        shap_dict['rolling_mean_12'] = round(random.uniform(10, 15), 2)
        shap_dict['rolling_std_4'] = round(random.uniform(4, 7), 2)

        # Month features - seasonal importance
        for i in range(1, 13):
            month_name = f'month_{i}'
            shap_dict[month_name] = round(random.uniform(1, 5), 2)

        # Quarter features
        for i in range(1, 5):
            quarter_name = f'quarter_{i}'
            shap_dict[quarter_name] = round(random.uniform(2, 6), 2)

        # Trend and seasonal
        shap_dict['trend'] = round(random.uniform(5, 10), 2)
        shap_dict['seasonal'] = round(random.uniform(3, 7), 2)

        # Convert to JSON
        shap_json = json.dumps(shap_dict)

        # Update database
        cursor.execute("""
            UPDATE prediction_log
            SET shap_values = ?
            WHERE rowid = ?
        """, (shap_json, row_id))

        updated_count += 1

        if updated_count % 50 == 0:
            conn.commit()  # Commit every 50 updates
            logger.info(f"  Updated {updated_count}/{len(rows)} predictions...")

    conn.commit()  # Final commit
    conn.close()

    logger.info("\n" + "="*80)
    logger.info(f"✓ Updated {updated_count} predictions with mock SHAP values")
    logger.info("="*80)

    return updated_count

if __name__ == '__main__':
    try:
        count = generate_simple_shap()
        sys.exit(0 if count > 0 else 0)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
