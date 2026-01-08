"""
RetailPRED Database Helper Module
Provides utilities for interacting with the prediction tracking database
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class RetailPREDDatabase:
    """Helper class for RetailPRED database operations"""

    def __init__(self, db_path: str = "data/retailpred.db"):
        """
        Initialize database connection

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path

    def get_connection(self) -> sqlite3.Connection:
        """
        Get a database connection with row factory enabled

        Returns:
            SQLite connection object
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # Prediction Log Operations

    def log_prediction(
        self,
        model_name: str,
        prediction_date: str,
        predicted_value: float,
        features: Dict[str, Any],
        confidence_interval_lower: Optional[float] = None,
        confidence_interval_upper: Optional[float] = None,
        shap_values: Optional[Dict[str, float]] = None,
        store_id: Optional[int] = None,
        product_id: Optional[int] = None,
        actual_value: Optional[float] = None,
    ) -> int:
        """
        Log a prediction to the database

        Args:
            model_name: Name of the model used for prediction
            prediction_date: Date of the prediction (YYYY-MM-DD format)
            predicted_value: Predicted sales value
            features: Dictionary of feature values used
            confidence_interval_lower: Lower bound of confidence interval
            confidence_interval_upper: Upper bound of confidence interval
            shap_values: Dictionary of SHAP values for explainability
            store_id: Optional store ID
            product_id: Optional product ID
            actual_value: Optional actual value (for tracking accuracy later)

        Returns:
            ID of the inserted prediction log
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                INSERT INTO prediction_log (
                    model_name, store_id, product_id, prediction_date,
                    predicted_value, actual_value, confidence_interval_lower,
                    confidence_interval_upper, features, shap_values, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model_name,
                    store_id,
                    product_id,
                    prediction_date,
                    predicted_value,
                    actual_value,
                    confidence_interval_lower,
                    confidence_interval_upper,
                    json.dumps(features),
                    json.dumps(shap_values) if shap_values else None,
                    datetime.now().isoformat(),
                ),
            )

            conn.commit()
            prediction_id = cursor.lastrowid
            logger.info(f"Logged prediction {prediction_id} for {model_name}")
            return prediction_id

        except Exception as e:
            conn.rollback()
            logger.error(f"Error logging prediction: {e}")
            raise
        finally:
            conn.close()

    def get_predictions(
        self,
        model_name: Optional[str] = None,
        store_id: Optional[int] = None,
        product_id: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve predictions from the database with optional filters

        Args:
            model_name: Filter by model name
            store_id: Filter by store ID
            product_id: Filter by product ID
            start_date: Filter predictions after this date
            end_date: Filter predictions before this date
            limit: Maximum number of records to return

        Returns:
            List of prediction dictionaries
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM prediction_log WHERE 1=1"
        params = []

        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)

        if store_id:
            query += " AND store_id = ?"
            params.append(store_id)

        if product_id:
            query += " AND product_id = ?"
            params.append(product_id)

        if start_date:
            query += " AND prediction_date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND prediction_date <= ?"
            params.append(end_date)

        query += " ORDER BY prediction_date ASC LIMIT ?"
        params.append(limit)

        try:
            cursor.execute(query, params)
            rows = cursor.fetchall()

            predictions = []
            for row in rows:
                prediction = dict(row)
                # Parse JSON fields
                if prediction.get("features"):
                    prediction["features"] = json.loads(prediction["features"])
                if prediction.get("shap_values"):
                    prediction["shap_values"] = json.loads(prediction["shap_values"])
                predictions.append(prediction)

            return predictions

        except Exception as e:
            logger.error(f"Error retrieving predictions: {e}")
            raise
        finally:
            conn.close()

    def update_actual_value(self, prediction_id: int, actual_value: float) -> None:
        """
        Update the actual value for a prediction (for tracking accuracy)

        Args:
            prediction_id: ID of the prediction to update
            actual_value: Actual sales value
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                "UPDATE prediction_log SET actual_value = ? WHERE id = ?",
                (actual_value, prediction_id),
            )
            conn.commit()
            logger.info(f"Updated actual value for prediction {prediction_id}")

        except Exception as e:
            conn.rollback()
            logger.error(f"Error updating actual value: {e}")
            raise
        finally:
            conn.close()

    # Model Metadata Operations

    def register_model(
        self,
        model_name: str,
        model_type: str,
        file_path: str,
        metrics: Dict[str, float],
        hyperparameters: Optional[Dict[str, Any]] = None,
        is_active: bool = True,
    ) -> int:
        """
        Register a trained model in the database

        Args:
            model_name: Unique name/identifier for the model
            model_type: Type of model (e.g., 'RandomForest', 'XGBoost')
            file_path: Path to the serialized model file
            metrics: Dictionary of accuracy metrics (RMSE, MAE, R2, etc.)
            hyperparameters: Optional dictionary of hyperparameters
            is_active: Whether this model is active for use

        Returns:
            ID of the inserted model metadata
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                INSERT INTO model_metadata (
                    model_name, model_type, training_date, metrics,
                    hyperparameters, file_path, is_active, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model_name,
                    model_type,
                    datetime.now().isoformat(),
                    json.dumps(metrics),
                    json.dumps(hyperparameters) if hyperparameters else None,
                    file_path,
                    1 if is_active else 0,
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                ),
            )

            conn.commit()
            model_id = cursor.lastrowid
            logger.info(f"Registered model {model_name} (ID: {model_id})")
            return model_id

        except sqlite3.IntegrityError:
            # Model with this name already exists, update instead
            return self.update_model(
                model_name=model_name,
                model_type=model_type,
                file_path=file_path,
                metrics=metrics,
                hyperparameters=hyperparameters,
                is_active=is_active,
            )
        except Exception as e:
            conn.rollback()
            logger.error(f"Error registering model: {e}")
            raise
        finally:
            conn.close()

    def update_model(
        self,
        model_name: str,
        model_type: Optional[str] = None,
        file_path: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        is_active: Optional[bool] = None,
    ) -> int:
        """
        Update an existing model's metadata

        Args:
            model_name: Name of the model to update
            model_type: New model type
            file_path: New file path
            metrics: New metrics dictionary
            hyperparameters: New hyperparameters dictionary
            is_active: New active status

        Returns:
            ID of the updated model
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        updates = []
        params = []

        if model_type:
            updates.append("model_type = ?")
            params.append(model_type)

        if file_path:
            updates.append("file_path = ?")
            params.append(file_path)

        if metrics:
            updates.append("metrics = ?")
            params.append(json.dumps(metrics))

        if hyperparameters is not None:
            updates.append("hyperparameters = ?")
            params.append(json.dumps(hyperparameters))

        if is_active is not None:
            updates.append("is_active = ?")
            params.append(1 if is_active else 0)

        updates.append("updated_at = ?")
        params.append(datetime.now().isoformat())

        params.append(model_name)

        try:
            query = f"UPDATE model_metadata SET {', '.join(updates)} WHERE model_name = ?"
            cursor.execute(query, params)
            conn.commit()

            cursor.execute("SELECT id FROM model_metadata WHERE model_name = ?", (model_name,))
            row = cursor.fetchone()
            model_id = row["id"] if row else None

            logger.info(f"Updated model {model_name} (ID: {model_id})")
            return model_id

        except Exception as e:
            conn.rollback()
            logger.error(f"Error updating model: {e}")
            raise
        finally:
            conn.close()

    def get_active_models(self) -> List[Dict[str, Any]]:
        """
        Get all active models from the database

        Returns:
            List of active model dictionaries
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT * FROM model_metadata WHERE is_active = 1")
            rows = cursor.fetchall()

            models = []
            for row in rows:
                model = dict(row)
                # Parse JSON fields
                if model.get("metrics"):
                    model["metrics"] = json.loads(model["metrics"])
                if model.get("hyperparameters"):
                    model["hyperparameters"] = json.loads(model["hyperparameters"])
                models.append(model)

            return models

        except Exception as e:
            logger.error(f"Error retrieving active models: {e}")
            raise
        finally:
            conn.close()

    def get_model_by_name(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get model metadata by name

        Args:
            model_name: Name of the model

        Returns:
            Model dictionary or None if not found
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT * FROM model_metadata WHERE model_name = ?", (model_name,))
            row = cursor.fetchone()

            if row:
                model = dict(row)
                # Parse JSON fields
                if model.get("metrics"):
                    model["metrics"] = json.loads(model["metrics"])
                if model.get("hyperparameters"):
                    model["hyperparameters"] = json.loads(model["hyperparameters"])
                return model

            return None

        except Exception as e:
            logger.error(f"Error retrieving model: {e}")
            raise
        finally:
            conn.close()

    def get_historical_sales(
        self, category_id: str, days_back: int = 90, data_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get historical sales data for a category

        Args:
            category_id: Category ID (e.g., '4400' for Total_Retail_Sales)
            days_back: Number of days of historical data to retrieve (default: 90)
            data_type: Optional filter by data_type (e.g., 'daily', 'monthly')

        Returns:
            List of historical sales data with date and value
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        query = """
            SELECT date, value
            FROM time_series_data
            WHERE category_id = ?
            AND value > 100  -- Filter out very small values which are likely not main sales data
        """

        params = [category_id]

        if data_type:
            query += " AND data_type = ?"
            params.append(data_type)

        query += " ORDER BY date DESC LIMIT ?"
        params.append(days_back)

        try:
            cursor.execute(query, params)
            rows = cursor.fetchall()

            # Return in chronological order (oldest first)
            result = [dict(row) for row in reversed(rows)]
            logger.info(f"Retrieved {len(result)} historical data points for category {category_id}")
            return result

        except Exception as e:
            logger.error(f"Error retrieving historical sales: {e}")
            raise
        finally:
            conn.close()

    # Analytics and Reporting

    def get_prediction_accuracy(
        self, model_name: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get prediction accuracy metrics

        Args:
            model_name: Optional filter by model name
            limit: Maximum number of records

        Returns:
            List of accuracy statistics
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        query = """
            SELECT
                model_name,
                COUNT(*) as total_predictions,
                COUNT(actual_value) as predictions_with_actuals,
                AVG(predicted_value) as avg_predicted,
                AVG(actual_value) as avg_actual,
                SUM(CASE WHEN actual_value IS NOT NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as coverage_pct
            FROM prediction_log
            WHERE actual_value IS NOT NULL
        """

        params = []
        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)

        query += " GROUP BY model_name ORDER BY model_name LIMIT ?"
        params.append(limit)

        try:
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Error calculating accuracy: {e}")
            raise
        finally:
            conn.close()


# Convenience instance for quick access
def get_db(db_path: str = "data/retailpred.db") -> RetailPREDDatabase:
    """
    Get a database instance

    Args:
        db_path: Path to the database file

    Returns:
        RetailPREDDatabase instance
    """
    return RetailPREDDatabase(db_path)
