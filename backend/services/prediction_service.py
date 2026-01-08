"""
Prediction Service
High-level service for model inference with automatic logging, validation, and performance tracking

This service wraps your existing ML models and provides:
- Automatic prediction logging to database
- Validation logic with error metrics calculation
- Model performance analytics
- Time-based accuracy tracking
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
import json

import numpy as np
from db.database import RetailPREDDatabase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionService:
    """
    High-level prediction service with logging and validation

    Wraps ML model inference and provides automatic database logging,
    validation tracking, and performance analytics.
    """

    def __init__(self, db_path: str = "data/retailpred.db", ml_base_path: str = None):
        """
        Initialize the prediction service

        Args:
            db_path: Path to SQLite database
            ml_base_path: Path to ML models directory
        """
        self.db = RetailPREDDatabase(db_path)

        # Import ML integrator
        import sys
        if ml_base_path and ml_base_path not in sys.path:
            sys.path.insert(0, ml_base_path)

        try:
            from ml.model_loader import get_ml_integrator
            self.ml_integrator = get_ml_integrator()
        except ImportError:
            logger.warning("ML integrator not available, using mock predictions")
            self.ml_integrator = None

    # --- Core Prediction Functions ---

    def predict(
        self,
        model_name: str,
        features: Dict[str, Any],
        store_id: Optional[int] = None,
        product_id: Optional[int] = None,
        prediction_date: Optional[str] = None,
        log_to_db: bool = True,
    ) -> Dict[str, Any]:
        """
        Make a prediction with automatic logging to database

        Args:
            model_name: Name of the model to use
            features: Dictionary of features for prediction
            store_id: Optional store ID
            product_id: Optional product ID
            prediction_date: Optional prediction date (default: today)
            log_to_db: Whether to log prediction to database

        Returns:
            Dictionary with prediction results:
            {
                'prediction_id': int,
                'predicted_value': float,
                'shap_values': dict,
                'confidence_interval': (lower, upper),
                'logged': bool
            }
        """
        if prediction_date is None:
            prediction_date = datetime.now().strftime("%Y-%m-%d")

        logger.info(
            f"Making prediction for model={model_name}, "
            f"store={store_id}, product={product_id}, date={prediction_date}"
        )

        try:
            # Get prediction from ML model
            if self.ml_integrator:
                predicted_value, shap_dict, metadata = self.ml_integrator.get_prediction(
                    features_dict=features,
                    model_name=model_name,
                    store_id=store_id,
                    product_id=product_id,
                )
            else:
                # Fallback to mock prediction
                logger.warning("Using mock prediction")
                predicted_value, shap_dict, metadata = self._mock_prediction(features)

            # Calculate confidence interval (±15% by default)
            confidence_lower = predicted_value * 0.85
            confidence_upper = predicted_value * 1.15

            # Log to database if requested
            prediction_id = None
            logged = False

            if log_to_db:
                try:
                    prediction_id = self.db.log_prediction(
                        model_name=model_name,
                        prediction_date=prediction_date,
                        predicted_value=predicted_value,
                        features=features,
                        shap_values=shap_dict,
                        confidence_interval_lower=confidence_lower,
                        confidence_interval_upper=confidence_upper,
                        store_id=store_id,
                        product_id=product_id,
                    )
                    logged = True
                    logger.info(f"✓ Prediction logged to database with ID: {prediction_id}")
                except Exception as e:
                    logger.error(f"Failed to log prediction: {e}")

            return {
                "prediction_id": prediction_id,
                "predicted_value": predicted_value,
                "shap_values": shap_dict,
                "confidence_interval": (confidence_lower, confidence_upper),
                "logged": logged,
                "metadata": metadata,
            }

        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise

    def predict_batch(
        self,
        model_name: str,
        features_list: List[Dict[str, Any]],
        store_ids: Optional[List[int]] = None,
        product_ids: Optional[List[int]] = None,
        prediction_dates: Optional[List[str]] = None,
        log_to_db: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Make batch predictions with logging

        Args:
            model_name: Name of the model to use
            features_list: List of feature dictionaries
            store_ids: Optional list of store IDs
            product_ids: Optional list of product IDs
            prediction_dates: Optional list of prediction dates
            log_to_db: Whether to log predictions to database

        Returns:
            List of prediction result dictionaries
        """
        logger.info(f"Making batch predictions: {len(features_list)} predictions")

        results = []
        n = len(features_list)

        for i, features in enumerate(features_list):
            store_id = store_ids[i] if store_ids and i < len(store_ids) else None
            product_id = product_ids[i] if product_ids and i < len(product_ids) else None
            pred_date = prediction_dates[i] if prediction_dates and i < len(prediction_dates) else None

            result = self.predict(
                model_name=model_name,
                features=features,
                store_id=store_id,
                product_id=product_id,
                prediction_date=pred_date,
                log_to_db=log_to_db,
            )
            results.append(result)

        logged_count = sum(1 for r in results if r["logged"])
        logger.info(f"✓ Batch complete: {logged_count}/{n} predictions logged")

        return results

    # --- Validation Functions ---

    def validate_prediction(
        self,
        prediction_id: int,
        actual_value: float,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Validate a prediction with actual value and calculate error metrics

        Args:
            prediction_id: ID of the prediction to validate
            actual_value: Actual sales value
            notes: Optional validation notes

        Returns:
            Dictionary with validation results:
            {
                'prediction_id': int,
                'predicted_value': float,
                'actual_value': float,
                'error_absolute': float,
                'error_percentage': float,
                'is_accurate': bool,
                'metrics': dict
            }
        """
        logger.info(f"Validating prediction {prediction_id} with actual value: {actual_value}")

        try:
            # Get prediction from database
            conn = self.db.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM prediction_log WHERE id = ?", (prediction_id,))
            row = cursor.fetchone()
            conn.close()

            if not row:
                raise ValueError(f"Prediction {prediction_id} not found")

            prediction = dict(row)
            predicted_value = prediction["predicted_value"]

            # Calculate error metrics
            error_absolute = abs(actual_value - predicted_value)
            error_percentage = self._calculate_percentage_error(predicted_value, actual_value)
            error_squared = (actual_value - predicted_value) ** 2

            # Determine if prediction is accurate (within 10% error)
            is_accurate = error_percentage <= 10.0

            # Update database with actual value
            self.db.update_actual_value(prediction_id, actual_value)

            validation_result = {
                "prediction_id": prediction_id,
                "model_name": prediction["model_name"],
                "predicted_value": predicted_value,
                "actual_value": actual_value,
                "error_absolute": round(error_absolute, 2),
                "error_percentage": round(error_percentage, 2),
                "error_squared": round(error_squared, 2),
                "is_accurate": is_accurate,
                "validation_date": datetime.now().isoformat(),
                "metrics": {
                    "mae_contribution": error_absolute,
                    "rmse_contribution": error_squared,
                    "mape_contribution": error_percentage,
                },
            }

            if notes:
                validation_result["notes"] = notes

            logger.info(
                f"✓ Prediction validated: error={error_percentage:.2f}%, "
                f"accurate={is_accurate}"
            )

            return validation_result

        except Exception as e:
            logger.error(f"Error validating prediction: {e}")
            raise

    def validate_batch(
        self,
        predictions: List[Tuple[int, float]],
    ) -> List[Dict[str, Any]]:
        """
        Validate multiple predictions

        Args:
            predictions: List of (prediction_id, actual_value) tuples

        Returns:
            List of validation result dictionaries
        """
        logger.info(f"Validating batch of {len(predictions)} predictions")

        results = []
        for prediction_id, actual_value in predictions:
            try:
                result = self.validate_prediction(prediction_id, actual_value)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to validate prediction {prediction_id}: {e}")
                results.append(
                    {"prediction_id": prediction_id, "error": str(e), "validated": False}
                )

        successful = sum(1 for r in results if r.get("validated", True))
        logger.info(f"✓ Batch validation complete: {successful}/{len(predictions)} successful")

        return results

    # --- Model Performance Functions ---

    def get_model_performance(
        self,
        model_name: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        store_id: Optional[int] = None,
        product_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive model performance metrics across all validated predictions

        Args:
            model_name: Optional model name filter
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)
            store_id: Optional store ID filter
            product_id: Optional product ID filter

        Returns:
            Dictionary with performance metrics:
            {
                'model_name': str,
                'total_predictions': int,
                'validated_predictions': int,
                'validation_rate': float,
                'metrics': {
                    'mae': float,
                    'rmse': float,
                    'mape': float,
                    'accuracy_within_5pct': int,
                    'accuracy_within_10pct': int,
                    'accuracy_within_15pct': int,
                },
                'error_distribution': dict,
                'period': str
            }
        """
        logger.info(f"Calculating model performance for model={model_name}")

        try:
            # Get validated predictions from database
            predictions = self.db.get_predictions(
                model_name=model_name,
                store_id=store_id,
                product_id=product_id,
                start_date=start_date,
                end_date=end_date,
                limit=10000,  # Get all predictions
            )

            # Filter only validated predictions
            validated = [p for p in predictions if p.get("actual_value") is not None]

            if not validated:
                return {
                    "model_name": model_name or "all",
                    "total_predictions": len(predictions),
                    "validated_predictions": 0,
                    "validation_rate": 0.0,
                    "message": "No validated predictions found",
                }

            # Calculate error metrics
            errors_abs = []
            errors_pct = []
            errors_sq = []

            for pred in validated:
                predicted = pred["predicted_value"]
                actual = pred["actual_value"]

                error_abs = abs(actual - predicted)
                error_pct = self._calculate_percentage_error(predicted, actual)
                error_sq = (actual - predicted) ** 2

                errors_abs.append(error_abs)
                errors_pct.append(error_pct)
                errors_sq.append(error_sq)

            # Calculate aggregate metrics
            mae = np.mean(errors_abs)
            rmse = np.sqrt(np.mean(errors_sq))
            mape = np.mean(errors_pct)

            # Calculate accuracy at different thresholds
            within_5pct = sum(1 for e in errors_pct if e <= 5.0)
            within_10pct = sum(1 for e in errors_pct if e <= 10.0)
            within_15pct = sum(1 for e in errors_pct if e <= 15.0)

            # Calculate error distribution
            error_distribution = {
                "min_error_pct": float(np.min(errors_pct)),
                "max_error_pct": float(np.max(errors_pct)),
                "median_error_pct": float(np.median(errors_pct)),
                "std_error_pct": float(np.std(errors_pct)),
                "p25_error_pct": float(np.percentile(errors_pct, 25)),
                "p75_error_pct": float(np.percentile(errors_pct, 75)),
            }

            validation_rate = len(validated) / len(predictions) * 100 if predictions else 0

            performance = {
                "model_name": model_name or "all",
                "total_predictions": len(predictions),
                "validated_predictions": len(validated),
                "validation_rate": round(validation_rate, 2),
                "metrics": {
                    "mae": round(mae, 2),
                    "rmse": round(rmse, 2),
                    "mape": round(mape, 2),
                    "accuracy_within_5pct": within_5pct,
                    "accuracy_within_10pct": within_10pct,
                    "accuracy_within_15pct": within_15pct,
                    "accuracy_5pct_pct": round(within_5pct / len(validated) * 100, 2),
                    "accuracy_10pct_pct": round(within_10pct / len(validated) * 100, 2),
                    "accuracy_15pct_pct": round(within_15pct / len(validated) * 100, 2),
                },
                "error_distribution": error_distribution,
                "period": f"{start_date or 'all'} to {end_date or 'present'}",
                "calculated_at": datetime.now().isoformat(),
            }

            logger.info(
                f"✓ Performance calculated: MAE={mae:.2f}, RMSE={rmse:.2f}, "
                f"MAPE={mape:.2f}%, n={len(validated)}"
            )

            return performance

        except Exception as e:
            logger.error(f"Error calculating model performance: {e}")
            raise

    def get_prediction_accuracy_over_time(
        self,
        model_name: str,
        days: int = 30,
        granularity: str = "daily",
        store_id: Optional[int] = None,
        product_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Calculate prediction accuracy trends over time

        Args:
            model_name: Name of the model
            days: Number of days to look back (default: 30)
            granularity: Time granularity - 'daily' or 'weekly'
            store_id: Optional store ID filter
            product_id: Optional product ID filter

        Returns:
            Dictionary with time-series accuracy metrics:
            {
                'model_name': str,
                'period_days': int,
                'granularity': str,
                'time_series': [
                    {
                        'period': str,
                        'mae': float,
                        'rmse': float,
                        'mape': float,
                        'predictions_count': int
                    },
                ],
                'trend': str,
                'overall_metrics': dict
            }
        """
        logger.info(
            f"Calculating accuracy over time: model={model_name}, days={days}, "
            f"granularity={granularity}"
        )

        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # Get all validated predictions in range
            predictions = self.db.get_predictions(
                model_name=model_name,
                store_id=store_id,
                product_id=product_id,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                limit=10000,
            )

            validated = [p for p in predictions if p.get("actual_value") is not None]

            if not validated:
                return {
                    "model_name": model_name,
                    "period_days": days,
                    "granularity": granularity,
                    "time_series": [],
                    "message": "No validated predictions found in period",
                }

            # Group predictions by time period
            time_groups = self._group_predictions_by_time(validated, granularity)

            # Calculate metrics for each period
            time_series = []
            all_mae = []
            all_rmse = []
            all_mape = []

            for period, preds in sorted(time_groups.items()):
                mae, rmse, mape = self._calculate_metrics_for_predictions(preds)

                all_mae.append(mae)
                all_rmse.append(rmse)
                all_mape.append(mape)

                time_series.append(
                    {
                        "period": period,
                        "mae": round(mae, 2),
                        "rmse": round(rmse, 2),
                        "mape": round(mape, 2),
                        "predictions_count": len(preds),
                    }
                )

            # Calculate trend
            if len(all_mape) >= 2:
                recent_avg = np.mean(all_mape[-3:])
                earlier_avg = np.mean(all_mape[:3]) if len(all_mape) >= 6 else np.mean(all_mape[:-3])

                if recent_avg < earlier_avg * 0.95:
                    trend = "improving"
                elif recent_avg > earlier_avg * 1.05:
                    trend = "degrading"
                else:
                    trend = "stable"
            else:
                trend = "insufficient_data"

            # Overall metrics
            overall_mae = np.mean(all_mae)
            overall_rmse = np.mean(all_rmse)
            overall_mape = np.mean(all_mape)

            result = {
                "model_name": model_name,
                "period_days": days,
                "granularity": granularity,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "time_series": time_series,
                "trend": trend,
                "overall_metrics": {
                    "avg_mae": round(overall_mae, 2),
                    "avg_rmse": round(overall_rmse, 2),
                    "avg_mape": round(overall_mape, 2),
                },
                "total_periods": len(time_series),
                "calculated_at": datetime.now().isoformat(),
            }

            logger.info(
                f"✓ Accuracy over time calculated: {len(time_series)} periods, "
                f"trend={trend}"
            )

            return result

        except Exception as e:
            logger.error(f"Error calculating accuracy over time: {e}")
            raise

    # --- Helper Functions ---

    @staticmethod
    def _calculate_percentage_error(predicted: float, actual: float) -> float:
        """Calculate percentage error"""
        if actual == 0:
            return 0.0
        return abs((actual - predicted) / actual) * 100

    @staticmethod
    def _calculate_metrics_for_predictions(
        predictions: List[Dict[str, Any]]
    ) -> Tuple[float, float, float]:
        """Calculate MAE, RMSE, MAPE for a list of predictions"""
        errors_abs = []
        errors_sq = []
        errors_pct = []

        for pred in predictions:
            predicted = pred["predicted_value"]
            actual = pred["actual_value"]

            errors_abs.append(abs(actual - predicted))
            errors_sq.append((actual - predicted) ** 2)
            errors_pct.append(PredictionService._calculate_percentage_error(predicted, actual))

        mae = np.mean(errors_abs)
        rmse = np.sqrt(np.mean(errors_sq))
        mape = np.mean(errors_pct)

        return mae, rmse, mape

    @staticmethod
    def _group_predictions_by_time(
        predictions: List[Dict[str, Any]], granularity: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group predictions by time period"""
        groups = {}

        for pred in predictions:
            pred_date = datetime.strptime(pred["prediction_date"], "%Y-%m-%d")

            if granularity == "daily":
                period = pred_date.strftime("%Y-%m-%d")
            elif granularity == "weekly":
                # Get Monday of the week
                monday = pred_date - timedelta(days=pred_date.weekday())
                period = monday.strftime("%Y-%m-%d")
            else:  # monthly
                period = pred_date.strftime("%Y-%m")

            if period not in groups:
                groups[period] = []
            groups[period].append(pred)

        return groups

    @staticmethod
    def _mock_prediction(features: Dict[str, Any]) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
        """Generate mock prediction for testing"""
        import random

        base_value = features.get("lag_1", 1000)
        promotion = features.get("promotion", 0)

        prediction = base_value * (1 + 0.15 * promotion)
        prediction += random.uniform(-50, 50)

        shap_values = {
            "lag_1": 0.35 * base_value,
            "promotion": 0.25 * prediction if promotion else 0,
            "moving_avg_7": 0.15 * base_value,
            "month": 0.10 * (prediction / 12),
        }

        metadata = {"model_used": "mock", "warning": "Using mock prediction"}

        return prediction, shap_values, metadata


# --- Convenience Functions ---

def log_prediction(
    model_name: str,
    features: Dict[str, Any],
    prediction: float,
    shap_values: Dict[str, float],
    store_id: Optional[int] = None,
    product_id: Optional[int] = None,
    prediction_date: Optional[str] = None,
    db_path: str = "data/retailpred.db",
) -> int:
    """
    Quick helper to log a prediction to the database

    Args:
        model_name: Name of the model
        features: Feature dictionary used for prediction
        prediction: Predicted value
        shap_values: SHAP value dictionary
        store_id: Optional store ID
        product_id: Optional product ID
        prediction_date: Optional prediction date (default: today)
        db_path: Path to database

    Returns:
        Prediction ID
    """
    db = RetailPREDDatabase(db_path)

    if prediction_date is None:
        prediction_date = datetime.now().strftime("%Y-%m-%d")

    # Calculate confidence intervals
    conf_lower = prediction * 0.85
    conf_upper = prediction * 1.15

    prediction_id = db.log_prediction(
        model_name=model_name,
        prediction_date=prediction_date,
        predicted_value=prediction,
        features=features,
        shap_values=shap_values,
        confidence_interval_lower=conf_lower,
        confidence_interval_upper=conf_upper,
        store_id=store_id,
        product_id=product_id,
    )

    return prediction_id


def validate_prediction(
    prediction_id: int,
    actual_value: float,
    notes: Optional[str] = None,
    db_path: str = "data/retailpred.db",
) -> Dict[str, Any]:
    """
    Quick helper to validate a prediction

    Args:
        prediction_id: ID of prediction to validate
        actual_value: Actual sales value
        notes: Optional validation notes
        db_path: Path to database

    Returns:
        Dictionary with validation results
    """
    service = PredictionService(db_path=db_path)
    return service.validate_prediction(prediction_id, actual_value, notes)


def get_model_performance(
    model_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db_path: str = "data/retailpred.db",
) -> Dict[str, Any]:
    """
    Quick helper to get model performance metrics

    Args:
        model_name: Optional model name filter
        start_date: Optional start date (YYYY-MM-DD)
        end_date: Optional end date (YYYY-MM-DD)
        db_path: Path to database

    Returns:
        Dictionary with performance metrics
    """
    service = PredictionService(db_path=db_path)
    return service.get_model_performance(model_name, start_date, end_date)


def get_prediction_accuracy_over_time(
    model_name: str,
    days: int = 30,
    granularity: str = "daily",
    db_path: str = "data/retailpred.db",
) -> Dict[str, Any]:
    """
    Quick helper to get prediction accuracy trends over time

    Args:
        model_name: Name of the model
        days: Number of days to look back
        granularity: 'daily' or 'weekly'
        db_path: Path to database

    Returns:
        Dictionary with time-series accuracy data
    """
    service = PredictionService(db_path=db_path)
    return service.get_prediction_accuracy_over_time(model_name, days, granularity)
