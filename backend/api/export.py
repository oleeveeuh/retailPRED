"""
Export API endpoints for RetailPRED
Provides CSV exports optimized for BI tools like Tableau
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import sqlite3
import csv
from io import StringIO
from pathlib import Path
from datetime import datetime

router = APIRouter()

# Database path
DB_PATH = Path(__file__).parent.parent.parent / "data" / "retailpred.db"


@router.get("/predictions-csv")
async def export_predictions_csv(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    category: Optional[str] = None,
    model_name: Optional[str] = None
):
    """
    Export predictions to CSV format optimized for Tableau

    Query Parameters:
    - start_date: Filter by prediction start date (YYYY-MM-DD)
    - end_date: Filter by prediction end date (YYYY-MM-DD)
    - category: Filter by category
    - model_name: Filter by model name

    Returns:
    - CSV file with columns: date, store, product, predicted_sales,
      actual_sales, model_name, error_pct (MAPE), error_abs (MAE),
      mase (Mean Absolute Scaled Error), confidence_lower, confidence_upper

    Error Metrics:
    - error_pct: MAPE = |actual - predicted| / actual * 100 (standard MAPE formula)
    - error_abs: MAE = |actual - predicted| (absolute error in dollars)
    - mase: MASE = MAE_model / MAE_naive (scaled error, <1 means better than naive)
    """

    try:
        # Connect to database
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Build query
        query = """
            SELECT
                prediction_date,
                store_id,
                product_id,
                predicted_value,
                actual_value,
                model_name,
                confidence_interval_lower,
                confidence_interval_upper,
                created_at
            FROM prediction_log
            WHERE 1=1
        """
        params = []

        # Add filters
        if start_date:
            query += " AND prediction_date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND prediction_date <= ?"
            params.append(end_date)

        if model_name:
            query += " AND model_name LIKE ?"
            params.append(f"%{model_name}%")

        query += " ORDER BY prediction_date DESC, model_name"

        # Execute query
        cursor.execute(query, params)
        rows = cursor.fetchall()

        if not rows:
            raise HTTPException(status_code=404, detail="No predictions found")

        # Create CSV in memory
        output = StringIO()
        writer = csv.writer(output)

        # Calculate MASE scaling factor (naive forecast errors)
        # MASE = MAE / Naive_MAE
        # where naive forecast uses the previous period's actual value
        mase_query = """
            SELECT
                model_name,
                AVG(ABS(actual_value - predicted_value)) as model_mae
            FROM prediction_log
            WHERE actual_value IS NOT NULL
            GROUP BY model_name
        """
        cursor.execute(mase_query)
        model_maes = {row['model_name']: row['model_mae'] for row in cursor.fetchall()}

        # Calculate naive MAE (using seasonal naive: same month from previous year)
        naive_query = """
            SELECT
                AVG(ABS(t1.value - (
                    SELECT t2.value
                    FROM time_series_data t2
                    JOIN categories c ON t2.category_id = c.id
                    WHERE c.name = 'total_sales'
                    AND t2.date = date(t1.date, '-1 year')
                    LIMIT 1
                ))) as naive_mae
            FROM time_series_data t1
            JOIN categories c ON t1.category_id = c.id
            WHERE c.name = 'total_sales'
            AND t1.date >= date('now', '-2 years')
        """
        cursor.execute(naive_query)
        naive_result = cursor.fetchone()
        naive_mae = naive_result['naive_mae'] if naive_result and naive_result['naive_mae'] else 100000

        # Write header
        writer.writerow([
            'date',
            'store',
            'product',
            'predicted_sales',
            'actual_sales',
            'model_name',
            'error_pct',
            'error_abs',
            'mase',
            'confidence_lower',
            'confidence_upper',
            'is_validated'
        ])

        # Write data rows
        for row in rows:
            prediction_date = row['prediction_date']
            store_id = row['store_id'] or 'All Stores'
            product_id = row['product_id'] or 'All Products'
            predicted_value = row['predicted_value']
            actual_value = row['actual_value']
            model_name = row['model_name']
            conf_lower = row['confidence_interval_lower']
            conf_upper = row['confidence_interval_upper']

            # Calculate error metrics if actual value exists
            if actual_value is not None:
                # MAPE: Mean Absolute Percentage Error (using actual as denominator)
                error_pct = abs((actual_value - predicted_value) / actual_value * 100)

                # MAE: Mean Absolute Error
                error_abs = abs(actual_value - predicted_value)

                # MASE: Mean Absolute Scaled Error
                model_mae = model_maes.get(model_name, 0)
                mase = (model_mae / naive_mae) if naive_mae > 0 else 0

                is_validated = 'Yes'
            else:
                error_pct = ''
                error_abs = ''
                mase = ''
                is_validated = 'No'

            # Write row
            writer.writerow([
                prediction_date,
                store_id,
                product_id,
                f"{predicted_value:.2f}",
                f"{actual_value:.2f}" if actual_value else '',
                model_name,
                f"{error_pct:.2f}" if error_pct else '',
                f"{error_abs:.2f}" if error_abs else '',
                f"{mase:.4f}" if mase else '',
                f"{conf_lower:.2f}" if conf_lower else '',
                f"{conf_upper:.2f}" if conf_upper else '',
                is_validated
            ])

        # Close connection
        conn.close()

        # Prepare response
        csv_content = output.getvalue()
        output.close()

        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"retail_predictions_{timestamp}.csv"

        # Return CSV file
        return StreamingResponse(
            iter([csv_content]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Type": "text/csv"
            }
        )

    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export error: {str(e)}")


@router.get("/historical-csv")
async def export_historical_csv(
    category: str = "total_sales",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    Export historical sales data to CSV format

    Query Parameters:
    - category: Category to export (default: total_sales)
    - start_date: Filter by start date (YYYY-MM-DD)
    - end_date: Filter by end date (YYYY-MM-DD)

    Returns:
    - CSV file with historical sales data
    """

    try:
        # Connect to database
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Build query
        query = """
            SELECT
                t.date,
                c.name as category,
                t.value,
                t.created_at
            FROM time_series_data t
            JOIN categories c ON t.category_id = c.id
            WHERE c.id = (SELECT id FROM categories WHERE name = ? LIMIT 1)
        """
        params = [category]

        if start_date:
            query += " AND t.date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND t.date <= ?"
            params.append(end_date)

        query += " ORDER BY t.date ASC"

        # Execute query
        cursor.execute(query, params)
        rows = cursor.fetchall()

        if not rows:
            raise HTTPException(status_code=404, detail="No historical data found")

        # Create CSV in memory
        output = StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow(['date', 'category', 'sales', 'created_at'])

        # Write data rows
        for row in rows:
            writer.writerow([
                row['date'],
                row['category'],
                f"{row['value']:.2f}",
                row['created_at']
            ])

        # Close connection
        conn.close()

        # Prepare response
        csv_content = output.getvalue()
        output.close()

        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"retail_historical_{category}_{timestamp}.csv"

        # Return CSV file
        return StreamingResponse(
            iter([csv_content]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Type": "text/csv"
            }
        )

    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export error: {str(e)}")


@router.get("/model-performance-csv")
async def export_model_performance_csv():
    """
    Export model performance metrics to CSV format

    Returns:
    - CSV file with model performance data including MAPE, RMSE, MAE, R²
    """

    try:
        # Connect to database
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Query model metadata
        query = """
            SELECT
                model_name,
                model_type,
                category,
                mape,
                rmse,
                mae,
                r2,
                is_active,
                created_at
            FROM model_metadata
            ORDER BY category, model_type, mape ASC
        """

        cursor.execute(query)
        rows = cursor.fetchall()

        if not rows:
            raise HTTPException(status_code=404, detail="No models found")

        # Create CSV in memory
        output = StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow([
            'model_name',
            'model_type',
            'category',
            'mape_percentage',
            'rmse',
            'mae',
            'r_squared',
            'is_active',
            'created_at'
        ])

        # Write data rows
        for row in rows:
            writer.writerow([
                row['model_name'],
                row['model_type'],
                row['category'],
                f"{row['mape']:.2f}" if row['mape'] else '',
                f"{row['rmse']:.2f}" if row['rmse'] else '',
                f"{row['mae']:.2f}" if row['mae'] else '',
                f"{row['r2']:.4f}" if row['r2'] else '',
                'Yes' if row['is_active'] else 'No',
                row['created_at']
            ])

        # Close connection
        conn.close()

        # Prepare response
        csv_content = output.getvalue()
        output.close()

        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"model_performance_{timestamp}.csv"

        # Return CSV file
        return StreamingResponse(
            iter([csv_content]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Type": "text/csv"
            }
        )

    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export error: {str(e)}")
