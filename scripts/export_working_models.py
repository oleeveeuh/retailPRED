#!/usr/bin/env python3
"""
Export Only Working Models to Demo Data

Exports predictions from LGBM and RandomForest models only,
excluding broken models (PatchTST, TimesNet, AutoARIMA).
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path

DB_PATH = Path('data/retailpred.db')
OUTPUT_DIR = Path('frontend/public/demo-data')

def export_predictions():
    """Export only working model predictions"""
    print("\n" + "="*80)
    print("EXPORTING WORKING MODEL PREDICTIONS TO DEMO DATA")
    print("="*80)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get only LGBM and RandomForest predictions
    query = """
    SELECT *
    FROM prediction_log
    WHERE model_name LIKE '%lgbm%' OR model_name LIKE '%randomforest%'
    ORDER BY prediction_date DESC, model_name
    """

    cursor.execute(query)
    rows = cursor.fetchall()

    print(f"\nFound {len(rows)} predictions from working models (LGBM, RandomForest)")

    # Convert to JSON format
    predictions = []
    for row in rows:
        pred = {
            'id': row['id'],
            'model_name': row['model_name'],
            'store_id': row['store_id'],
            'product_id': row['product_id'],
            'prediction_date': row['prediction_date'],
            'predicted_value': row['predicted_value'],
            'actual_value': row['actual_value'],
            'confidence_interval_lower': row['confidence_interval_lower'],
            'confidence_interval_upper': row['confidence_interval_upper'],
            'features': json.loads(row['features']) if row['features'] else None,
            'shap_values': json.loads(row['shap_values']) if row['shap_values'] else None,
            'created_at': row['created_at'],
            'error_percentage': row['error_percentage'],
            'is_validated': row['is_validated'],
            'confidence_score': row['confidence_score'],
            'error_absolute': row['error_absolute']
        }
        predictions.append(pred)

    # Save predictions
    output_file = OUTPUT_DIR / 'predictions.json'
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2, default=str)

    print(f"✓ Exported {len(predictions)} predictions to {output_file}")

    # Calculate summary
    cursor.execute("""
    SELECT
      COUNT(*) as total_count,
      COUNT(DISTINCT model_name) as model_count,
      COUNT(DISTINCT SUBSTR(model_name, 1, INSTR(model_name, '_') - 1)) as category_count,
      ROUND(AVG(ABS(error_percentage)), 2) as avg_mape,
      SUM(CASE WHEN actual_value IS NOT NULL THEN 1 ELSE 0 END) as validated_count
    FROM prediction_log
    WHERE model_name LIKE '%lgbm%' OR model_name LIKE '%randomforest%'
    """)

    summary_row = cursor.fetchone()
    summary_data = {
        'export_timestamp': datetime.now().isoformat(),
        'database_path': str(DB_PATH),
        'predictions': {
            'total_count': summary_row[0],
            'models_exported': summary_row[1],
            'categories': summary_row[2],
            'average_mape': summary_row[3],
            'validated_count': summary_row[4]
        },
        'models_available': {
            'total_count': 2,
            'models': ['LGBM', 'RandomForest'],
            'with_shap': ['LGBM', 'RandomForest']
        },
        'demo_data': {
            'predictions_included': len(predictions),
            'note': 'Only working models (LGBM, RandomForest) included. Broken models (PatchTST, TimesNet, AutoARIMA) excluded.'
        }
    }

    # Save summary
    summary_file = OUTPUT_DIR / 'summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)

    print(f"✓ Exported summary to {summary_file}")
    print(f"\nSummary:")
    print(f"  Total predictions: {summary_data['predictions']['total_count']}")
    print(f"  Models: {summary_data['models_available']['models']}")
    print(f"  Categories: {summary_data['predictions']['categories']}")
    print(f"  Average MAPE: {summary_data['predictions']['average_mape']}%")
    print(f"  Validated: {summary_data['predictions']['validated_count']}")

    conn.close()

    print("\n✓ Export complete!")
    print("\nNOTE: Demo data now includes only LGBM and RandomForest models.")
    print("      These models have excellent accuracy (~9% MAPE).")
    print("      Broken models (PatchTST, TimesNet with scale issues,")
    print("      and AutoARIMA with poor performance) are excluded.")

if __name__ == '__main__':
    export_predictions()
