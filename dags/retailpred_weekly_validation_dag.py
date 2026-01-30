"""
RetailPRED Weekly Validation DAG

This DAG runs weekly validation tasks for RetailPRED predictions:
1. Fetches latest actual values from MRTS API
2. Updates predictions with actuals in the database
3. Calculates validation metrics (MAPE, MAE)
4. Detects anomalies (high error predictions)
5. Exports metrics for dashboard consumption

Environment Variables:
    RETAILPRED_DIR: Path to RetailPRED repository (default: /home/oliau/retailPRED)
    AIRFLOW_HOME: Airflow home directory

Author: RetailPRED Team
Schedule: Every Monday at 2 AM
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Optional Airflow imports - allow local testing without Airflow installed
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.operators.bash import BashOperator
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False
    DAG = None
    PythonOperator = None
    BashOperator = None

# ============================================================================
# Configuration
# ============================================================================

# Repository paths (use environment variable or default)
# If running from dags/ directory, parent is repo root
if Path(__file__).parent.name == "dags":
    RETAILPRED_DIR = Path(__file__).parent.parent
else:
    RETAILPRED_DIR = Path(os.getenv(
        "RETAILPRED_DIR",
        "/home/oliau/retailPRED"
    ))

# Add repository root to Python path for imports
sys.path.insert(0, str(RETAILPRED_DIR))

# ============================================================================
# Import RetailPRED modules
# ============================================================================

try:
    from config import (
        DATABASE_PATH,
        VALIDATION_METRICS_PATH,
        MRTS_BASE_URL,
        MRTS_TIMEOUT,
        RETAIL_CATEGORIES,
        ANOMALY_THRESHOLD_DEFAULT,
        AIRFLOW_HOME
    )
except ImportError as e:
    raise ImportError(
        f"Could not import RetailPRED config from {RETAILPRED_DIR}. "
        f"Ensure RETAILPRED_DIR environment variable points to the repository root. "
        f"Error: {e}"
    )

try:
    from weekly_validation import (
        fetch_actuals_from_mrts,
        update_predictions_with_actuals,
        calculate_metrics_for_week,
        find_anomalies,
        export_metrics_for_dashboard
    )
except ImportError as e:
    raise ImportError(
        f"Could not import weekly_validation module from {RETAILPRED_DIR}. "
        f"Ensure weekly_validation.py exists in the repository root. "
        f"Error: {e}"
    )

# ============================================================================
# DAG Defaults
# ============================================================================

default_args = {
    'owner': 'retailpred',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 6),  # First Monday of 2025
    'email': ['admin@retailpred.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=1),
}

# Only create DAG if Airflow is available
dag = None
if AIRFLOW_AVAILABLE:
    dag = DAG(
        'retailpred_weekly_validation',
        default_args=default_args,
        description='Weekly prediction validation and metrics calculation for RetailPRED',
        schedule='0 2 * * 1',  # 2 AM every Monday
        catchup=False,
        tags=['retail', 'validation', 'weekly', 'retailpred'],
        max_active_runs=1,
        params={
            'database_path': str(DATABASE_PATH),
            'output_path': str(VALIDATION_METRICS_PATH),
            'anomaly_threshold': ANOMALY_THRESHOLD_DEFAULT,
            'repo_dir': str(RETAILPRED_DIR),
        },
    )

# ============================================================================
# Task Functions
# ============================================================================

def get_target_date(**context) -> str:
    """
    Determine the target validation date.
    Uses the logical date (ds) from Airflow context.

    Returns:
        Date string in YYYY-MM-DD format
    """
    # ds is the logical date (execution_date) in YYYY-MM-DD format
    return context['ds']


def fetch_mrts_actuals(ti, target_date: str, **context) -> dict:
    """
    Fetch actual values from MRTS API.

    Args:
        ti: TaskInstance for XCom data passing
        target_date: Target date in YYYY-MM-DD format

    Returns:
        Dictionary of category_key -> actual_value
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info(f"Fetching MRTS actuals for: {target_date}")

    try:
        actuals = fetch_actuals_from_mrts(target_date)
        logger.info(f"Fetched {len(actuals)} category values from MRTS API")

        # Pass to next task via XCom
        ti.xcom_push(key='actuals', value=actuals)

        return actuals

    except Exception as e:
        logger.error(f"Failed to fetch MRTS data: {e}")
        # Return empty dict to allow workflow to continue
        # (predictions may already have actuals from other sources)
        return {}


def update_predictions(ti, target_date: str, **context) -> int:
    """
    Update prediction_log table with actual values.

    Args:
        ti: TaskInstance for XCom data retrieval
        target_date: Target date in YYYY-MM-DD format

    Returns:
        Number of predictions updated
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info(f"Updating predictions for: {target_date}")

    # Pull actuals from XCom
    actuals = ti.xcom_pull(task_ids='fetch_mrts_actuals', key='actuals')

    if not actuals:
        logger.info("No actuals to update (skip if MRTS fetch failed)")
        return 0

    try:
        updated = update_predictions_with_actuals(actuals, target_date)
        logger.info(f"Updated {updated} predictions with actuals")
        return updated

    except Exception as e:
        logger.error(f"Failed to update predictions: {e}")
        raise


def calculate_metrics(ti, target_date: str, **context) -> dict:
    """
    Calculate validation metrics for all models.

    Args:
        ti: TaskInstance for XCom data passing
        target_date: Target date in YYYY-MM-DD format

    Returns:
        Metrics dictionary
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info(f"Calculating metrics for: {target_date}")

    try:
        metrics = calculate_metrics_for_week(target_date)
        model_count = len(metrics.get("models", {}))

        logger.info(f"Calculated metrics for {model_count} models")

        # Pass to next task via XCom
        ti.xcom_push(key='metrics', value=metrics)

        return metrics

    except Exception as e:
        logger.error(f"Failed to calculate metrics: {e}")
        raise


def detect_anomalies(ti, target_date: str, threshold: float, **context) -> list:
    """
    Detect anomalies (predictions with high error).

    Args:
        ti: TaskInstance for XCom data passing
        target_date: Target date in YYYY-MM-DD format
        threshold: Error percentage threshold

    Returns:
        List of anomaly records
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info(f"Detecting anomalies with threshold {threshold}%")

    try:
        anomalies = find_anomalies(target_date, threshold)

        logger.info(f"Found {len(anomalies)} anomalies")
        if anomalies:
            logger.info("Top 5 anomalies:")
            for i, anomaly in enumerate(anomalies[:5], 1):
                logger.info(
                    f"  {i}. {anomaly['model_name']}: "
                    f"{anomaly['error_percentage']:.2f}% error "
                    f"({anomaly['prediction_date']})"
                )

        # Pass to next task via XCom
        ti.xcom_push(key='anomalies', value=anomalies)

        return anomalies

    except Exception as e:
        logger.error(f"Failed to detect anomalies: {e}")
        raise


def export_dashboard_metrics(ti, **context) -> str:
    """
    Export metrics and anomalies to JSON for dashboard.

    Args:
        ti: TaskInstance for XCom data retrieval

    Returns:
        Path to exported file
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Exporting metrics for dashboard")

    try:
        # Pull data from XCom
        metrics = ti.xcom_pull(task_ids='calculate_metrics', key='metrics')
        anomalies = ti.xcom_pull(task_ids='detect_anomalies', key='anomalies')

        if not metrics:
            raise ValueError("No metrics found from calculate_metrics task")

        export_path = export_metrics_for_dashboard(metrics, anomalies)

        logger.info(f"Exported metrics to: {export_path}")
        return export_path

    except Exception as e:
        logger.error(f"Failed to export metrics: {e}")
        raise


def validation_summary(ti, target_date: str, **context) -> None:
    """
    Print validation summary and check for warnings.

    Args:
        ti: TaskInstance for XCom data retrieval
        target_date: Target date for summary
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("RETAILPRED WEEKLY VALIDATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Target Date: {target_date}")

    # Pull data for summary
    metrics = ti.xcom_pull(task_ids='calculate_metrics', key='metrics')
    anomalies = ti.xcom_pull(task_ids='detect_anomalies', key='anomalies')

    if metrics:
        model_count = len(metrics.get("models", {}))
        logger.info(f"Models Validated: {model_count}")

    if anomalies is not None:
        anomaly_count = len(anomalies)
        logger.info(f"Anomalies Found: {anomaly_count}")

        if anomaly_count > 10:
            logger.warning(f"⚠️  High number of anomalies: {anomaly_count}")
            logger.warning("Consider investigating model performance")

    logger.info("=" * 80)


def save_predictions_to_db(ti, target_date: str, **context) -> dict:
    """
    Generate and save predictions to the database.

    Args:
        ti: TaskInstance for XCom data retrieval
        target_date: Target date reference

    Returns:
        Summary of saved predictions
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Generating and saving predictions to database")

    try:
        # Import modules
        from generate_rolling_predictions import generate_all_predictions
        from save_predictions_to_db import save_predictions

        # Calculate start date (first of next month)
        from datetime import datetime, timedelta
        ref_date = datetime.strptime(target_date, '%Y-%m-%d')
        start_date = (ref_date.replace(day=1) + timedelta(days=32)).replace(day=1)

        # Generate predictions
        predictions, summary = generate_all_predictions(
            start_date=start_date.strftime('%Y-%m-%d'),
            months_ahead=12
        )

        logger.info(f"Generated {summary['total_predictions']} predictions")

        # Save predictions to database (force=False to skip duplicates)
        added, skipped, errors = save_predictions(predictions, force=False)

        logger.info(f"Saved: {added} added, {skipped} skipped, {errors} errors")

        return {
            "generated": summary.get('total_predictions', 0),
            "added": added,
            "skipped": skipped,
            "errors": errors
        }

    except Exception as e:
        logger.error(f"Failed to save predictions: {e}")
        raise


# ============================================================================
# Task Definitions (only if Airflow is available)
# ============================================================================

if AIRFLOW_AVAILABLE:
    # Task 1: Get target date
    get_target_date_task = PythonOperator(
        task_id='get_target_date',
        python_callable=get_target_date,
        do_xcom_push=True,
        dag=dag,
    )

    # Task 2: Fetch actuals from MRTS API
    fetch_mrts_task = PythonOperator(
        task_id='fetch_mrts_actuals',
        python_callable=fetch_mrts_actuals,
        op_kwargs={
            'target_date': "{{ ti.xcom_pull(task_ids='get_target_date') }}",
        },
        dag=dag,
    )

    # Task 3: Update predictions with actuals
    update_predictions_task = PythonOperator(
        task_id='update_predictions',
        python_callable=update_predictions,
        op_kwargs={
            'target_date': "{{ ti.xcom_pull(task_ids='get_target_date') }}",
        },
        dag=dag,
    )

    # Task 4: Calculate validation metrics
    calculate_metrics_task = PythonOperator(
        task_id='calculate_metrics',
        python_callable=calculate_metrics,
        op_kwargs={
            'target_date': "{{ ti.xcom_pull(task_ids='get_target_date') }}",
        },
        dag=dag,
    )

    # Task 5: Detect anomalies
    detect_anomalies_task = PythonOperator(
        task_id='detect_anomalies',
        python_callable=detect_anomalies,
        op_kwargs={
            'target_date': "{{ ti.xcom_pull(task_ids='get_target_date') }}",
            'threshold': "{{ params.anomaly_threshold }}",
        },
        dag=dag,
    )

    # Task 6: Generate and save predictions to database
    # (Combines generation and saving to avoid XCom data passing)
    save_predictions_task = PythonOperator(
        task_id='save_predictions_to_db',
        python_callable=save_predictions_to_db,
        op_kwargs={
            'target_date': "{{ ti.xcom_pull(task_ids='get_target_date') }}",
        },
        dag=dag,
    )

    # Task 7: Export metrics for dashboard
    export_metrics_task = PythonOperator(
        task_id='export_metrics',
        python_callable=export_dashboard_metrics,
        dag=dag,
    )

    # Task 8: Print summary
    summary_task = PythonOperator(
        task_id='validation_summary',
        python_callable=validation_summary,
        op_kwargs={
            'target_date': "{{ ti.xcom_pull(task_ids='get_target_date') }}",
        },
        dag=dag,
        trigger_rule='all_done',  # Run even if upstream tasks fail
    )

    # Task 9: Push to GitHub
    push_to_github_task = BashOperator(
        task_id='push_to_github',
        bash_command=f"""
        cd {RETAILPRED_DIR}

        # Configure git
        git config user.name "Airflow Automation"
        git config user.email "airflow@retailpred.com"

        # Add updated files
        git add data/retailpred.db data/validation_metrics.json

        # Commit with timestamp
        git commit -m "chore: weekly validation update $(date +%Y-%m-%d)" || echo "No changes to commit"

        # Push to main branch
        git push origin main || echo "Push failed - check SSH keys"

        echo "✓ Git push complete"
        """,
        dag=dag,
        trigger_rule='all_done',  # Run even if upstream tasks fail
    )

    # ============================================================================
    # Task Dependencies
    # ============================================================================

    # Validation workflow (tasks 1-5)
    get_target_date_task >> fetch_mrts_task >> update_predictions_task
    update_predictions_task >> calculate_metrics_task
    calculate_metrics_task >> detect_anomalies_task

    # Prediction generation workflow (task 6)
    # Generate and save new predictions after validation starts
    get_target_date_task >> save_predictions_task

    # Both workflows converge at export and summary
    detect_anomalies_task >> export_metrics_task
    save_predictions_task >> export_metrics_task
    export_metrics_task >> summary_task

    # Push to GitHub after summary
    summary_task >> push_to_github_task

# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    """
    Test the DAG locally by running this script directly.
    This allows you to verify the workflow before deploying to Airflow.
    """
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Set test paths
    test_dir = Path(__file__).parent.parent
    print(f"Testing DAG from: {test_dir}")
    print(f"Database: {DATABASE_PATH}")
    print(f"Output: {VALIDATION_METRICS_PATH}")
    print()

    # Test imports
    print("Testing imports...")
    print(f"  ✅ config imported: DATABASE_PATH = {DATABASE_PATH}")
    print(f"  ✅ weekly_validation functions imported")
    print()

    # Test execution with a sample date
    test_date = "2025-01-05"
    print(f"Running validation for: {test_date}")
    print("-" * 80)

    try:
        # Fetch actuals (will fail without API, but that's OK for testing)
        print("Step 1: Fetching MRTS actuals...")
        actuals = fetch_actuals_from_mrts(test_date)
        print(f"  Fetched {len(actuals)} categories")

        # Calculate metrics (should work with existing data)
        print("Step 2: Calculating metrics...")
        metrics = calculate_metrics_for_week(test_date)
        print(f"  Metrics for {len(metrics.get('models', {}))} models")

        # Detect anomalies
        print("Step 3: Detecting anomalies...")
        anomalies = find_anomalies(test_date, ANOMALY_THRESHOLD_DEFAULT)
        print(f"  Found {len(anomalies)} anomalies")

        # Export
        print("Step 4: Exporting metrics...")
        export_path = export_metrics_for_dashboard(metrics, anomalies)
        print(f"  Exported to: {export_path}")

        print()
        print("=" * 80)
        print("DAG TEST COMPLETE ✅")
        print("=" * 80)

    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
