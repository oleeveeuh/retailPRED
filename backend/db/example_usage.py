"""
Example Usage of RetailPRED Database Helper
Demonstrates how to use the database module for prediction tracking
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.database import RetailPREDDatabase


def example_usage():
    """Example of how to use the database helper"""

    # Initialize database connection
    # Path is relative to the project root (backend/../data/)
    db = RetailPREDDatabase("../data/retailpred.db")

    print("=" * 60)
    print("RetailPRED Database Helper - Example Usage")
    print("=" * 60)
    print()

    # Example 1: Register a trained model
    print("1. Registering a model...")
    model_id = db.register_model(
        model_name="random_forecast_v1",
        model_type="RandomForest",
        file_path="/path/to/model.pkl",
        metrics={"rmse": 0.85, "mae": 0.62, "r2": 0.92},
        hyperparameters={"n_estimators": 100, "max_depth": 10, "random_state": 42},
        is_active=True,
    )
    print(f"   ✓ Model registered with ID: {model_id}")
    print()

    # Example 2: Log a prediction
    print("2. Logging a prediction...")
    prediction_id = db.log_prediction(
        model_name="random_forecast_v1",
        prediction_date="2024-01-15",
        predicted_value=1500.50,
        features={
            "month": 1,
            "year": 2024,
            "promotion": 1,
            "holiday": 0,
            "day_of_week": 1,
            "lag_1": 1450.0,
            "moving_avg_7": 1480.0,
        },
        confidence_interval_lower=1400.0,
        confidence_interval_upper=1600.0,
        shap_values={
            "lag_1": 0.35,
            "promotion": 0.25,
            "moving_avg_7": 0.20,
            "month": 0.10,
            "holiday": 0.05,
            "day_of_week": 0.05,
        },
        store_id=1,
        product_id=101,
    )
    print(f"   ✓ Prediction logged with ID: {prediction_id}")
    print()

    # Example 3: Get active models
    print("3. Retrieving active models...")
    active_models = db.get_active_models()
    print(f"   Found {len(active_models)} active model(s):")
    for model in active_models:
        print(f"   - {model['model_name']} ({model['model_type']})")
        print(f"     RMSE: {model['metrics']['rmse']}, R²: {model['metrics']['r2']}")
    print()

    # Example 4: Get predictions
    print("4. Retrieving predictions...")
    predictions = db.get_predictions(model_name="random_forecast_v1", limit=5)
    print(f"   Found {len(predictions)} prediction(s):")
    for pred in predictions[:3]:  # Show first 3
        print(f"   - {pred['prediction_date']}: ${pred['predicted_value']:.2f}")
    print()

    # Example 5: Update actual value (when actual sales data becomes available)
    print("5. Updating actual value...")
    db.update_actual_value(prediction_id, actual_value=1525.75)
    print(f"   ✓ Updated actual value for prediction {prediction_id}")
    print()

    # Example 6: Get prediction accuracy
    print("6. Calculating prediction accuracy...")
    accuracy = db.get_prediction_accuracy(model_name="random_forecast_v1")
    for stats in accuracy:
        print(f"   Model: {stats['model_name']}")
        print(f"   - Total predictions: {stats['total_predictions']}")
        print(f"   - With actuals: {stats['predictions_with_actuals']}")
        print(f"   - Coverage: {stats['coverage_pct']:.1f}%")
        if stats['avg_actual']:
            print(f"   - Avg predicted: ${stats['avg_predicted']:.2f}")
            print(f"   - Avg actual: ${stats['avg_actual']:.2f}")
    print()

    print("=" * 60)
    print("Example completed!")
    print("=" * 60)


def cleanup_example_data():
    """Clean up example data from database"""
    db = RetailPREDDatabase("../data/retailpred.db")

    print("Cleaning up example data...")

    # Delete example predictions
    conn = db.get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM prediction_log WHERE model_name = ?", ("random_forecast_v1",))
    cursor.execute("DELETE FROM model_metadata WHERE model_name = ?", ("random_forecast_v1",))
    conn.commit()
    conn.close()

    print("✓ Cleanup complete")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RetailPRED Database Example")
    parser.add_argument(
        "--cleanup", action="store_true", help="Clean up example data"
    )

    args = parser.parse_args()

    if args.cleanup:
        cleanup_example_data()
    else:
        example_usage()
