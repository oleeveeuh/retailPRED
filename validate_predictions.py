#!/usr/bin/env python3
"""
RetailPRED Prediction Validation Script

This script validates predictions against actual values, calculates metrics,
finds anomalies, and exports metrics for the dashboard.

Usage:
    python validate_predictions.py --help
    python validate_predictions.py --metrics
    python validate_predictions.py --anomalies --threshold 10
    python validate_predictions.py --update-date 2025-01-05 --actual 6204.22
    python validate_predictions.py --export
"""

import sys
import os
import argparse
import json
import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATABASE_PATH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Database Connection
# ============================================================================

def get_db_connection() -> sqlite3.Connection:
    """Get database connection"""
    return sqlite3.connect(DATABASE_PATH)


# ============================================================================
# Update Prediction with Actual Value
# ============================================================================

def update_prediction_with_actual(
    prediction_date: str,
    actual_value: float,
    model_name: Optional[str] = None,
    store_id: int = 1,
    product_id: int = 1
) -> Dict[str, Any]:
    """
    Update a prediction with the actual value and recalculate error metrics

    Args:
        prediction_date: Date in YYYY-MM-DD format
        actual_value: Actual sales value
        model_name: Specific model to update (default: all models for this date)
        store_id: Store ID (default: 1)
        product_id: Product ID (default: 1)

    Returns:
        Dictionary with update results
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Find predictions to update
        if model_name:
            cursor.execute("""
                SELECT id, model_name, predicted_value, actual_value
                FROM prediction_log
                WHERE prediction_date = ?
                AND model_name = ?
                AND store_id = ?
                AND product_id = ?
            """, (prediction_date, model_name, store_id, product_id))
        else:
            cursor.execute("""
                SELECT id, model_name, predicted_value, actual_value
                FROM prediction_log
                WHERE prediction_date = ?
                AND store_id = ?
                AND product_id = ?
            """, (prediction_date, store_id, product_id))

        predictions = cursor.fetchall()

        if not predictions:
            logger.warning(f"No predictions found for date {prediction_date}")
            return {
                "success": False,
                "message": f"No predictions found for date {prediction_date}",
                "updated_count": 0
            }

        updated_count = 0
        errors = []

        for pred_id, pred_model_name, predicted_value, existing_actual in predictions:
            # Calculate error metrics
            error_absolute = abs(actual_value - predicted_value)
            error_percentage = (error_absolute / actual_value * 100) if actual_value != 0 else None

            # Update the prediction
            cursor.execute("""
                UPDATE prediction_log
                SET actual_value = ?,
                    error_absolute = ?,
                    error_percentage = ?,
                    is_validated = 1
                WHERE id = ?
            """, (actual_value, error_absolute, error_percentage, pred_id))

            updated_count += 1
            logger.info(f"Updated {pred_model_name} for {prediction_date}: "
                       f"predicted={predicted_value:.2f}, actual={actual_value:.2f}, "
                       f"error={error_percentage:.2f}%")

        conn.commit()

        result = {
            "success": True,
            "message": f"Updated {updated_count} prediction(s)",
            "updated_count": updated_count,
            "prediction_date": prediction_date,
            "actual_value": actual_value
        }

    except Exception as e:
        conn.rollback()
        logger.error(f"Error updating prediction: {e}")
        result = {
            "success": False,
            "message": f"Error: {e}",
            "updated_count": 0
        }
    finally:
        conn.close()

    return result


# ============================================================================
# Calculate Metrics
# ============================================================================

def calculate_metrics(
    model_name: Optional[str] = None,
    category: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Calculate validation metrics from predictions with actual values

    Args:
        model_name: Filter by specific model (optional)
        category: Filter by category (extracted from model name)
        start_date: Start date for metrics calculation (YYYY-MM-DD)
        end_date: End date for metrics calculation (YYYY-MM-DD)

    Returns:
        Dictionary with metrics by model
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Build query
    where_clauses = ["actual_value IS NOT NULL"]
    params = []

    if model_name:
        where_clauses.append("model_name = ?")
        params.append(model_name)

    if start_date:
        where_clauses.append("prediction_date >= ?")
        params.append(start_date)

    if end_date:
        where_clauses.append("prediction_date <= ?")
        params.append(end_date)

    where_sql = " AND ".join(where_clauses)

    # Get predictions for metrics calculation
    query = f"""
        SELECT
            model_name,
            prediction_date,
            predicted_value,
            actual_value
        FROM prediction_log
        WHERE {where_sql}
        ORDER BY model_name, prediction_date
    """

    cursor.execute(query, params)
    rows = cursor.fetchall()

    if not rows:
        conn.close()
        return {
            "success": False,
            "message": "No validated predictions found",
            "metrics": {}
        }

    # Group by model
    model_data = {}
    for model_name, pred_date, predicted, actual in rows:
        if model_name not in model_data:
            model_data[model_name] = {
                "dates": [],
                "predicted": [],
                "actuals": []
            }
        model_data[model_name]["dates"].append(pred_date)
        model_data[model_name]["predicted"].append(float(predicted))
        model_data[model_name]["actuals"].append(float(actual))

    conn.close()

    # Calculate metrics for each model
    metrics_by_model = {}

    for model_name, data in model_data.items():
        predicted = np.array(data["predicted"])
        actuals = np.array(data["actuals"])

        # MAE (Mean Absolute Error)
        mae = np.mean(np.abs(actuals - predicted))

        # RMSE (Root Mean Squared Error)
        rmse = np.sqrt(np.mean((actuals - predicted) ** 2))

        # MAPE (Mean Absolute Percentage Error)
        # Handle division by zero
        mask = actuals != 0
        if mask.any():
            mape = np.mean(np.abs((actuals[mask] - predicted[mask]) / actuals[mask])) * 100
        else:
            mape = 0.0

        # SMAPE (Symmetric MAPE)
        smape = np.mean(2.0 * np.abs(actuals - predicted) / (np.abs(actuals) + np.abs(predicted))) * 100

        # MASE (Mean Absolute Scaled Error)
        # Using naive forecast as baseline
        if len(actuals) > 1:
            naive_errors = np.abs(actuals[1:] - actuals[:-1])
            naive_mae = np.mean(naive_errors)
            mase = mae / naive_mae if naive_mae > 0 else 0
        else:
            mase = 0

        # R² (R-squared)
        ss_res = np.sum((actuals - predicted) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Accuracy (100 - MAPE, clamped to 0-100)
        accuracy = max(0, 100 - mape)

        metrics_by_model[model_name] = {
            "total_predictions": len(predicted),
            "MAE": float(mae),
            "RMSE": float(rmse),
            "MAPE": float(mape),
            "SMAPE": float(smape),
            "MASE": float(mase),
            "R2": float(r2),
            "accuracy": float(accuracy),
            "date_range": {
                "start": data["dates"][0],
                "end": data["dates"][-1]
            }
        }

    return {
        "success": True,
        "metrics": metrics_by_model,
        "total_predictions": sum(len(data["predicted"]) for data in model_data.values())
    }


def print_metrics_summary(metrics: Dict[str, Any]):
    """Print formatted metrics summary"""
    print("\n" + "=" * 80)
    print("VALIDATION METRICS SUMMARY")
    print("=" * 80)

    for model_name, model_metrics in metrics.get("metrics", {}).items():
        print(f"\n{model_name}:")
        print(f"  Predictions: {model_metrics['total_predictions']}")
        print(f"  MAE: ${model_metrics['MAE']:,.2f}")
        print(f"  RMSE: ${model_metrics['RMSE']:,.2f}")
        print(f"  MAPE: {model_metrics['MAPE']:.2f}%")
        print(f"  SMAPE: {model_metrics['SMAPE']:.2f}%")
        print(f"  MASE: {model_metrics['MASE']:.4f}")
        print(f"  R²: {model_metrics['R2']:.4f}")
        print(f"  Accuracy: {model_metrics['accuracy']:.2f}%")
        print(f"  Date Range: {model_metrics['date_range']['start']} to {model_metrics['date_range']['end']}")

    print("\n" + "=" * 80)


# ============================================================================
# Find Anomalies
# ============================================================================

def find_anomalies(
    threshold: float = 10.0,
    model_name: Optional[str] = None,
    top_n: int = 20
) -> Dict[str, Any]:
    """
    Find predictions with large errors (anomalies)

    Args:
        threshold: Error percentage threshold to consider as anomaly
        model_name: Filter by specific model (optional)
        top_n: Number of top anomalies to return

    Returns:
        Dictionary with anomalies found
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    where_clauses = ["actual_value IS NOT NULL", "error_percentage >= ?"]
    params = [threshold]

    if model_name:
        where_clauses.append("model_name = ?")
        params.append(model_name)

    where_sql = " AND ".join(where_clauses)

    query = f"""
        SELECT
            model_name,
            prediction_date,
            predicted_value,
            actual_value,
            error_percentage,
            error_absolute,
            is_validated
        FROM prediction_log
        WHERE {where_sql}
        ORDER BY error_percentage DESC
        LIMIT ?
    """

    cursor.execute(query, params + [top_n])
    rows = cursor.fetchall()

    conn.close()

    anomalies = []
    for row in rows:
        anomalies.append({
            "model_name": row[0],
            "prediction_date": row[1],
            "predicted_value": float(row[2]),
            "actual_value": float(row[3]),
            "error_percentage": float(row[4]),
            "error_absolute": float(row[5]),
            "is_validated": bool(row[6])
        })

    return {
        "success": True,
        "threshold": threshold,
        "anomalies_found": len(anomalies),
        "anomalies": anomalies
    }


def print_anomalies(anomalies_result: Dict[str, Any]):
    """Print formatted anomalies"""
    print("\n" + "=" * 80)
    print(f"ANOMALIES (Error >= {anomalies_result['threshold']}%)")
    print("=" * 80)

    for i, anomaly in enumerate(anomalies_result["anomalies"][:20], 1):
        print(f"\n{i}. {anomaly['model_name']} - {anomaly['prediction_date']}")
        print(f"   Predicted: ${anomaly['predicted_value']:,.2f}")
        print(f"   Actual:    ${anomaly['actual_value']:,.2f}")
        print(f"   Error:     {anomaly['error_percentage']:.2f}% (${anomaly['error_absolute']:,.2f})")

    print("\n" + "=" * 80)


# ============================================================================
# Export Metrics for Dashboard
# ============================================================================

def export_metrics_json(
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Export validation metrics to JSON for dashboard consumption

    Args:
        output_path: Custom output path (optional)

    Returns:
        Dictionary with export results
    """
    # Get current metrics
    metrics_result = calculate_metrics()

    if not metrics_result["success"]:
        return {
            "success": False,
            "message": metrics_result.get("message", "No metrics to export")
        }

    # Prepare dashboard-friendly format
    dashboard_data = {
        "generated_at": datetime.now().isoformat(),
        "total_predictions": metrics_result["total_predictions"],
        "models": {}
    }

    # Group by category for dashboard
    category_metrics = {}

    for model_name, model_metrics in metrics_result["metrics"].items():
        # Extract category from model name
        # e.g., "total_retail_sales_lgbm_model" -> "total_retail_sales"
        parts = model_name.replace("_model", "").split("_")
        if parts[-1] in ["lgbm", "randomforest", "patchtst", "timesnet", "seasonalnaive", "autoarima"]:
            category = "_".join(parts[:-1])  # Everything except model type
            model_type = parts[-1]
        else:
            category = model_name
            model_type = "unknown"

        if category not in category_metrics:
            category_metrics[category] = {}

        category_metrics[category][model_type] = {
            "MAE": model_metrics["MAE"],
            "RMSE": model_metrics["RMSE"],
            "MAPE": model_metrics["MAPE"],
            "accuracy": model_metrics["accuracy"],
            "R2": model_metrics["R2"],
            "predictions": model_metrics["total_predictions"]
        }

    dashboard_data["models"] = category_metrics

    # Determine output path
    if output_path is None:
        output_path = PROJECT_ROOT / "data" / "validation_metrics.json"
    else:
        output_path = Path(output_path)

    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to file
    with open(output_path, 'w') as f:
        json.dump(dashboard_data, f, indent=2)

    logger.info(f"Metrics exported to: {output_path}")

    return {
        "success": True,
        "output_path": str(output_path),
        "models_exported": len(category_metrics)
    }


# ============================================================================
# Get Prediction Status
# ============================================================================

def get_prediction_status() -> Dict[str, Any]:
    """
    Get overall prediction validation status

    Returns:
        Dictionary with status information
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Overall stats
    cursor.execute("""
        SELECT
            COUNT(*) as total_predictions,
            SUM(CASE WHEN actual_value IS NOT NULL THEN 1 ELSE 0 END) as validated_count,
            SUM(CASE WHEN actual_value IS NULL THEN 1 ELSE 0 END) as pending_count,
            COUNT(DISTINCT model_name) as total_models,
            MIN(prediction_date) as first_prediction_date,
            MAX(prediction_date) as last_prediction_date
        FROM prediction_log
    """)

    row = cursor.fetchone()
    conn.close()

    total, validated, pending, total_models, first_date, last_date = row

    validation_rate = (validated / total * 100) if total > 0 else 0

    return {
        "total_predictions": total,
        "validated_predictions": validated,
        "pending_predictions": pending,
        "validation_rate": validation_rate,
        "total_models": total_models,
        "first_prediction_date": first_date,
        "last_prediction_date": last_date
    }


def print_status(status: Dict[str, Any]):
    """Print formatted status"""
    print("\n" + "=" * 80)
    print("PREDICTION VALIDATION STATUS")
    print("=" * 80)
    print(f"Total Predictions:    {status['total_predictions']:,}")
    print(f"Validated:            {status['validated_predictions']:,} ({status['validation_rate']:.1f}%)")
    print(f"Pending (no actual):   {status['pending_predictions']:,}")
    print(f"Total Models:         {status['total_models']}")
    print(f"Date Range:           {status['first_prediction_date']} to {status['last_prediction_date']}")
    print("=" * 80)


# ============================================================================
# Main CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validate predictions and calculate metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show current validation status
  python validate_predictions.py --status

  # Calculate and display metrics
  python validate_predictions.py --metrics

  # Find anomalies (predictions with >10% error)
  python validate_predictions.py --anomalies --threshold 10

  # Update a prediction with actual value
  python validate_predictions.py --update-date 2025-01-05 --actual 6204.22

  # Export metrics for dashboard
  python validate_predictions.py --export

  # Calculate metrics for specific model
  python validate_predictions.py --metrics --model total_retail_sales_lgbm_model
        """
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Show prediction validation status"
    )

    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Calculate and display validation metrics"
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Filter by model name (e.g., total_retail_sales_lgbm_model)"
    )

    parser.add_argument(
        "--anomalies",
        action="store_true",
        help="Find predictions with high error"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="Error percentage threshold for anomalies (default: 10%%)"
    )

    parser.add_argument(
        "--update-date",
        type=str,
        help="Update predictions for this date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--actual",
        type=float,
        help="Actual value to use with --update-date"
    )

    parser.add_argument(
        "--export",
        action="store_true",
        help="Export metrics to JSON for dashboard"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Custom output path for exported JSON"
    )

    args = parser.parse_args()

    # If no args, show status
    if len(sys.argv) == 1:
        args.status = True

    exit_code = 0

    # Show status
    if args.status:
        status = get_prediction_status()
        print_status(status)

    # Calculate metrics
    if args.metrics:
        metrics = calculate_metrics(model_name=args.model)
        if metrics["success"]:
            print_metrics_summary(metrics)
        else:
            print(f"Error: {metrics.get('message')}")
            exit_code = 1

    # Find anomalies
    if args.anomalies:
        anomalies = find_anomalies(threshold=args.threshold, model_name=args.model)
        print_anomalies(anomalies)

    # Update prediction
    if args.update_date:
        if args.actual is None:
            print("Error: --actual value required when using --update-date")
            exit_code = 1
        else:
            result = update_prediction_with_actual(args.update_date, args.actual)
            print(f"\n{result['message']}")

    # Export metrics
    if args.export:
        result = export_metrics_json(output_path=args.output)
        print(f"\n✅ Exported {result['models_exported']} categories to: {result['output_path']}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
