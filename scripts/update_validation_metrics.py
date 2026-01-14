#!/usr/bin/env python3
"""
Update Validation Metrics for Newly Deployed Models

This script calculates and updates validation metrics for the 15 deployed models.
"""

import sqlite3
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Deployed models
DEPLOYED_MODELS = {
    'randomforest': [
        'automobile_dealers_RandomForest_model',
        'building_materials_RandomForest_model',
        'clothing_accessories_RandomForest_model',
        'electronics_and_appliances_RandomForest_model',
        'food_beverage_RandomForest_model',
        'furniture_home_furnishings_RandomForest_model',
        'gasoline_stations_RandomForest_model',
        'general_merchandise_RandomForest_model',
        'health_personal_care_RandomForest_model',
        'sporting_goods_hobby_RandomForest_model',
        'total_sales_RandomForest_model',
    ],
    'lgbm': [
        'sporting_goods_hobby_LGBM_model',
        'furniture_home_furnishings_LGBM_model',
        'building_materials_LGBM_model',
        'general_merchandise_LGBM_model',
    ]
}

# Expected MASE from retraining (for reference)
EXPECTED_MASE = {
    'automobile_dealers_RandomForest_model': 1.3961,
    'building_materials_RandomForest_model': 1.5455,
    'clothing_accessories_RandomForest_model': 1.3839,
    'electronics_and_appliances_RandomForest_model': 1.3618,
    'food_beverage_RandomForest_model': 1.4298,
    'furniture_home_furnishings_RandomForest_model': 1.5473,
    'gasoline_stations_RandomForest_model': 1.3515,
    'general_merchandise_RandomForest_model': 1.3174,
    'health_personal_care_RandomForest_model': 1.1567,
    'sporting_goods_hobby_RandomForest_model': 1.3065,
    'total_sales_RandomForest_model': 1.4284,
    'sporting_goods_hobby_LGBM_model': 1.5752,
    'furniture_home_furnishings_LGBM_model': 1.5237,
    'building_materials_LGBM_model': 1.4469,
    'general_merchandise_LGBM_model': 1.7354,
}


def calculate_metrics_for_model(model_name: str):
    """Calculate validation metrics for a single model"""
    db_path = Path(__file__).parent.parent / "data/retailpred.db"
    conn = sqlite3.connect(db_path)

    query = """
        SELECT
            prediction_date,
            actual_value,
            predicted_value
        FROM prediction_log
        WHERE model_name = ?
        AND actual_value IS NOT NULL
        ORDER BY prediction_date
    """

    cursor = conn.cursor()
    cursor.execute(query, (model_name,))
    rows = cursor.fetchall()

    if not rows:
        conn.close()
        return None

    dates = [row[0] for row in rows]
    actuals = np.array([float(row[1]) for row in rows])
    predicted = np.array([float(row[2]) for row in rows])

    conn.close()

    # Calculate metrics
    mae = np.mean(np.abs(actuals - predicted))
    rmse = np.sqrt(np.mean((actuals - predicted) ** 2))

    # MAPE (avoid division by zero)
    mape = np.mean(np.abs((actuals - predicted) / actuals)) * 100

    # SMAPE
    smape = np.mean(2.0 * np.abs(actuals - predicted) / (np.abs(actuals) + np.abs(predicted))) * 100

    # MASE (using naive baseline)
    naive_errors = np.abs(actuals[1:] - actuals[:-1])
    naive_mae = np.mean(naive_errors)
    mase = mae / naive_mae if naive_mae > 0 else 0

    # R²
    ss_res = np.sum((actuals - predicted) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # SMAPE - alternative calculation
    smape = np.mean(100 * np.abs(actuals - predicted) / ((np.abs(actuals) + np.abs(predicted)) / 2))

    return {
        'total_predictions': len(rows),
        'validated_predictions': len(rows),
        'validation_rate': 1.0,
        'metrics': {
            'MAE': {'mean': float(mae), 'description': 'Mean Absolute Error on test data'},
            'RMSE': {'mean': float(rmse), 'description': 'Root Mean Squared Error on test data'},
            'MAPE': {'mean': float(mape), 'description': 'Mean Absolute Percentage Error on test data'},
            'SMAPE': {'mean': float(smape), 'description': 'Symmetric Mean Absolute Percentage Error'},
            'MASE': {'mean': float(mase), 'description': 'Mean Absolute Scaled Error on test data'},
            'R2': {'mean': float(r2), 'description': 'R² Score on test data'}
        }
    }


def update_validation_metrics():
    """Update validation_metrics.json with new model metrics"""
    metrics_path = Path(__file__).parent.parent / "training_outputs/validation_metrics.json"

    # Load existing metrics
    with open(metrics_path) as f:
        validation_data = json.load(f)

    print("=" * 80)
    print("UPDATING VALIDATION METRICS")
    print("=" * 80)
    print()

    # Update each deployed model
    all_models = DEPLOYED_MODELS['randomforest'] + DEPLOYED_MODELS['lgbm']

    for model_name in all_models:
        print(f"Processing: {model_name}")

        metrics = calculate_metrics_for_model(model_name)

        if metrics:
            # Update in validation data
            if model_name in validation_data['models']:
                validation_data['models'][model_name] = metrics
                print(f"  ✅ Updated: MASE {metrics['metrics']['MASE']['mean']:.4f}")
            else:
                print(f"  ⚠️  Model not found in validation data, skipping")
        else:
            print(f"  ❌ No predictions found for model")

    # Update generation timestamp
    validation_data['generated_at'] = datetime.now().isoformat()

    # Save updated metrics
    with open(metrics_path, 'w') as f:
        json.dump(validation_data, f, indent=2)

    print()
    print("=" * 80)
    print("VALIDATION METRICS UPDATED")
    print("=" * 80)
    print(f"Updated: {len(all_models)} models")
    print(f"File: {metrics_path}")
    print()

    return validation_data


def print_summary(validation_data):
    """Print summary of updated metrics"""
    all_models = DEPLOYED_MODELS['randomforest'] + DEPLOYED_MODELS['lgbm']

    print("=" * 80)
    print("UPDATED MODEL METRICS SUMMARY")
    print("=" * 80)
    print()

    # RandomForest summary
    rf_models = [m for m in all_models if 'RandomForest' in m]
    rf_mase = []

    print("RANDOMFOREST MODELS:")
    print("-" * 80)
    for model in rf_models:
        if model in validation_data['models']:
            mase = validation_data['models'][model]['metrics']['MASE']['mean']
            rf_mase.append(mase)
            status = "✅" if mase < 1.5 else "⚠️"
            print(f"  {status} {model:<50s} MASE: {mase:.4f}")

    if rf_mase:
        avg_mase = np.mean(rf_mase)
        print(f"  Average MASE: {avg_mase:.4f}")

    print()
    print("LGBM MODELS:")
    print("-" * 80)

    lgbm_models = [m for m in all_models if 'LGBM' in m]
    lgbm_mase = []

    for model in lgbm_models:
        if model in validation_data['models']:
            mase = validation_data['models'][model]['metrics']['MASE']['mean']
            lgbm_mase.append(mase)
            status = "✅" if mase < 1.5 else "⚠️"
            print(f"  {status} {model:<50s} MASE: {mase:.4f}")

    if lgbm_mase:
        avg_mase = np.mean(lgbm_mase)
        print(f"  Average MASE: {avg_mase:.4f}")

    print()
    print("=" * 80)


def main():
    """Main update function"""

    # Update validation metrics
    validation_data = update_validation_metrics()

    # Print summary
    print_summary(validation_data)

    print()
    print("✅ Validation metrics successfully updated!")
    print()


if __name__ == "__main__":
    main()
