#!/usr/bin/env python3
"""
Weekly Validation Script for Airflow

This script runs weekly validation tasks for RetailPRED:
1. Fetches latest actual values from MRTS API
2. Updates predictions with actuals
3. Calculates validation metrics
4. Detects anomalies
5. Exports metrics for dashboard

Designed for automated execution via Airflow DAG.

Usage:
    python weekly_validation.py [--date YYYY-MM-DD]

Exit Codes:
    0: Success
    1: Validation errors occurred
    2: Configuration error
    3: API fetch error
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# Import from config
try:
    from config import (
        DATABASE_PATH,
        VALIDATION_METRICS_PATH,
        MRTS_BASE_URL,
        MRTS_TIMEOUT,
        RETAIL_CATEGORIES,
        ANOMALY_THRESHOLD_DEFAULT
    )
except ImportError:
    print("ERROR: Could not import config. Please run from repository root.")
    sys.exit(2)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent / 'logs' / 'weekly_validation.log')
    ]
)
logger = logging.getLogger(__name__)


def fetch_actuals_from_mrts(target_date: str) -> Dict[str, float]:
    """
    Fetch actual retail sales values from MRTS API for a given date.

    Args:
        target_date: Date string in YYYY-MM format (MRTS data is monthly)

    Returns:
        Dictionary mapping category keys to actual sales values
    """
    import requests

    actuals = {}

    # Convert target_date to YYYY-MM format for MRTS API
    try:
        dt = datetime.strptime(target_date, '%Y-%m-%d')
        mrts_time = dt.strftime('%Y-%m')
    except ValueError:
        logger.error(f"Invalid date format: {target_date}. Use YYYY-MM-DD.")
        return {}

    logger.info(f"Fetching MRTS data for {mrts_time}")

    # MRTS API seasonally adjusted data codes
    # These correspond to our retail categories
    mrts_codes = {
        'total_retail_sales': '44812',      # Advance Retail Sales: Retail and Food Services, Total
        'automobile_dealers': '44431',      # Gasoline Stations: Sales
        'building_materials_garden': '44435',  # Building Materials and Garden Equipment and Supplies Dealers
        'clothing_accessories': '44837',    # Clothing and Clothing Accessories Dealers
        'electronics_appliances': '44441',  # Electronics and Appliance Stores
        'food_beverage_stores': '44421',    # Food and Beverage Stores
        'furniture_home_furnishings': '44433',  # Furniture and Home Furnishings Stores
        'gasoline_stations': '44431',       # Gasoline Stations
        'general_merchandise': '44449',     # General Merchandise Stores
        'health_personal_care': '44443',    # Health and Personal Care Stores
        'sporting_goods_hobby': '44445',    # Sporting Goods, Hobby, Musical Instrument, and Book Stores
        'nonstore_retailers': '44451',      # Nonstore Retailers
    }

    for category_key, mrts_code in mrts_codes.items():
        try:
            url = f"{MRTS_BASE_URL}"
            params = {
                'time': mrts_time,
                'data_type_code': mrts_code,
                'seasonally_adj': 'true'
            }

            response = requests.get(url, params=params, timeout=MRTS_TIMEOUT)

            if response.status_code == 200:
                data = response.json()
                if len(data) > 1:
                    # MRTS returns [ [col1, col2, ...], [value1, value2, ...], ... ]
                    value = float(data[1][0])
                    actuals[category_key] = value
                    logger.info(f"  {category_key}: ${value:,.2f}M")
                else:
                    logger.warning(f"  No data available for {category_key}")
            else:
                logger.warning(f"  API error for {category_key}: HTTP {response.status_code}")

        except Exception as e:
            logger.error(f"  Error fetching {category_key}: {e}")

    return actuals


def update_predictions_with_actuals(actuals: Dict[str, float], target_date: str) -> int:
    """
    Update prediction_log table with actual values.

    Args:
        actuals: Dictionary of category_key -> actual_value
        target_date: Date string in YYYY-MM-DD format

    Returns:
        Number of predictions updated
    """
    import sqlite3

    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    updated_count = 0

    for category_key, actual_value in actuals.items():
        try:
            # Find predictions for this category and date
            # Match model names that contain the category
            cursor.execute("""
                UPDATE prediction_log
                SET actual_value = ?,
                    error_absolute = ABS(predicted_value - ?),
                    error_percentage = ABS(predicted_value - ?) / ? * 100,
                    is_validated = 1
                WHERE prediction_date = ?
                AND model_name LIKE ?
                AND actual_value IS NULL
            """, (actual_value, actual_value, actual_value, actual_value,
                  target_date, f'%{category_key}%'))

            rows_updated = cursor.rowcount
            updated_count += rows_updated
            logger.info(f"  Updated {rows_updated} predictions for {category_key}")

        except Exception as e:
            logger.error(f"  Error updating {category_key}: {e}")

    conn.commit()
    conn.close()

    return updated_count


def calculate_metrics_for_week(target_date: str) -> Dict[str, Any]:
    """
    Calculate validation metrics for predictions around target date.

    Args:
        target_date: Date string in YYYY-MM-DD format

    Returns:
        Dictionary of metrics by model
    """
    import sqlite3
    import json

    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    # Get all validated predictions from the past week
    cursor.execute("""
        SELECT
            model_name,
            COUNT(*) as total_predictions,
            SUM(CASE WHEN actual_value IS NOT NULL THEN 1 ELSE 0 END) as validated_count,
            AVG(CASE WHEN actual_value IS NOT NULL AND error_percentage IS NOT NULL
                THEN error_percentage ELSE NULL END) as avg_mape,
            AVG(CASE WHEN actual_value IS NOT NULL AND error_absolute IS NOT NULL
                THEN error_absolute ELSE NULL END) as avg_mae
        FROM prediction_log
        WHERE prediction_date <= ?
        AND actual_value IS NOT NULL
        GROUP BY model_name
        ORDER BY model_name
    """, (target_date,))

    results = cursor.fetchall()

    metrics = {
        "target_date": target_date,
        "generated_at": datetime.now().isoformat(),
        "models": {}
    }

    for row in results:
        model_name = row[0]
        total_preds = row[1]
        validated = row[2]
        avg_mape = row[3]
        avg_mae = row[4]

        if validated > 0 and avg_mape is not None:
            metrics["models"][model_name] = {
                "total_predictions": total_preds,
                "validated_predictions": validated,
                "validation_rate": validated / total_preds if total_preds > 0 else 0,
                "metrics": {
                    "MAPE": {
                        "mean": round(avg_mape, 4) if avg_mape else None,
                        "description": "Mean Absolute Percentage Error"
                    },
                    "MAE": {
                        "mean": round(avg_mae, 4) if avg_mae else None,
                        "description": "Mean Absolute Error"
                    }
                }
            }

    conn.close()

    return metrics


def find_anomalies(target_date: str, threshold: float) -> list:
    """
    Find predictions with high error (anomalies).

    Args:
        target_date: Date string in YYYY-MM-DD format
        threshold: Error percentage threshold

    Returns:
        List of anomaly records
    """
    import sqlite3

    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            model_name,
            prediction_date,
            predicted_value,
            actual_value,
            error_percentage,
            error_absolute
        FROM prediction_log
        WHERE prediction_date <= ?
        AND actual_value IS NOT NULL
        AND error_percentage > ?
        ORDER BY error_percentage DESC
        LIMIT 50
    """, (target_date, threshold))

    rows = cursor.fetchall()
    conn.close()

    anomalies = []
    for row in rows:
        anomalies.append({
            "model_name": row[0],
            "prediction_date": row[1],
            "predicted_value": row[2],
            "actual_value": row[3],
            "error_percentage": row[4],
            "error_absolute": row[5]
        })

    return anomalies


def export_metrics_for_dashboard(metrics: Dict[str, Any], anomalies: list) -> str:
    """
    Export metrics to JSON file for dashboard consumption.

    Args:
        metrics: Metrics dictionary
        anomalies: List of anomalies

    Returns:
        Path to exported file
    """
    import json

    output = {
        "validation_metrics": metrics,
        "anomalies": {
            "count": len(anomalies),
            "threshold": ANOMALY_THRESHOLD_DEFAULT,
            "items": anomalies
        },
        "last_updated": datetime.now().isoformat()
    }

    # Ensure directory exists
    VALIDATION_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(VALIDATION_METRICS_PATH, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Exported metrics to: {VALIDATION_METRICS_PATH}")
    return str(VALIDATION_METRICS_PATH)


def run_weekly_validation(target_date: Optional[str] = None,
                          fetch_actuals: bool = True,
                          anomaly_threshold: float = ANOMALY_THRESHOLD_DEFAULT) -> int:
    """
    Run the complete weekly validation workflow.

    Args:
        target_date: Target date for validation (default: last Sunday)
        fetch_actuals: Whether to fetch actuals from MRTS API
        anomaly_threshold: Threshold for anomaly detection

    Returns:
        Exit code (0 = success, 1 = error)
    """
    logger.info("=" * 80)
    logger.info("WEEKLY VALIDATION WORKFLOW")
    logger.info("=" * 80)

    # Determine target date (default to last Sunday)
    if target_date is None:
        today = datetime.now()
        days_since_sunday = (today.weekday() + 1) % 7  # Monday=0, Sunday=6
        last_sunday = today - timedelta(days=days_since_sunday)
        target_date = last_sunday.strftime('%Y-%m-%d')

    logger.info(f"Target Date: {target_date}")
    logger.info(f"Anomaly Threshold: {anomaly_threshold}%")
    logger.info("")

    exit_code = 0

    # Step 1: Fetch actuals from MRTS API
    if fetch_actuals:
        logger.info("Step 1: Fetching actuals from MRTS API")
        logger.info("-" * 80)
        actuals = fetch_actuals_from_mrts(target_date)

        if actuals:
            logger.info(f"✅ Fetched {len(actuals)} category values")
        else:
            logger.warning("⚠️  No actuals fetched from API")
            # Continue anyway - we might have actuals from other sources
    else:
        logger.info("Step 1: Skipping MRTS fetch (--no-fetch flag)")
        actuals = {}

    # Step 2: Update predictions with actuals
    logger.info("")
    logger.info("Step 2: Updating predictions with actuals")
    logger.info("-" * 80)

    if actuals:
        updated = update_predictions_with_actuals(actuals, target_date)
        logger.info(f"✅ Updated {updated} predictions with actuals")
    else:
        logger.info("ℹ️  No actuals to update")

    # Step 3: Calculate metrics
    logger.info("")
    logger.info("Step 3: Calculating validation metrics")
    logger.info("-" * 80)

    metrics = calculate_metrics_for_week(target_date)
    model_count = len(metrics.get("models", {}))
    logger.info(f"✅ Calculated metrics for {model_count} models")

    # Step 4: Find anomalies
    logger.info("")
    logger.info("Step 4: Detecting anomalies")
    logger.info("-" * 80)

    anomalies = find_anomalies(target_date, anomaly_threshold)
    logger.info(f"✅ Found {len(anomalies)} anomalies (threshold: {anomaly_threshold}%)")

    if anomalies:
        logger.info("Top 5 anomalies:")
        for i, anomaly in enumerate(anomalies[:5], 1):
            logger.info(f"  {i}. {anomaly['model_name']}: "
                       f"{anomaly['error_percentage']:.2f}% error "
                       f"({anomaly['prediction_date']})")

    # Step 5: Export for dashboard
    logger.info("")
    logger.info("Step 5: Exporting metrics for dashboard")
    logger.info("-" * 80)

    export_path = export_metrics_for_dashboard(metrics, anomalies)
    logger.info(f"✅ Exported to {export_path}")

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("WEEKLY VALIDATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Target Date: {target_date}")
    logger.info(f"Models Validated: {model_count}")
    logger.info(f"Anomalies Found: {len(anomalies)}")
    logger.info(f"Export Path: {export_path}")

    if len(anomalies) > 10:
        logger.warning(f"⚠️  High number of anomalies: {len(anomalies)}")
        exit_code = 1

    return exit_code


def main():
    """Main entry point for CLI usage"""
    parser = argparse.ArgumentParser(
        description="Run weekly validation for RetailPRED predictions"
    )
    parser.add_argument(
        '--date',
        type=str,
        help='Target date in YYYY-MM-DD format (default: last Sunday)'
    )
    parser.add_argument(
        '--no-fetch',
        action='store_true',
        help='Skip fetching actuals from MRTS API'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=ANOMALY_THRESHOLD_DEFAULT,
        help=f'Anomaly detection threshold (default: {ANOMALY_THRESHOLD_DEFAULT}%%)'
    )

    args = parser.parse_args()

    try:
        exit_code = run_weekly_validation(
            target_date=args.date,
            fetch_actuals=not args.no_fetch,
            anomaly_threshold=args.threshold
        )
        sys.exit(exit_code)

    except KeyboardInterrupt:
        logger.info("Validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Validation failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
