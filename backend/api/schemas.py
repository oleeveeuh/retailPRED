"""
Pydantic models for API request/response validation
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class ModelType(str, Enum):
    """Supported model types"""
    RANDOM_FOREST = "RandomForest"
    LIGHTGBM = "LGBM"
    AUTOARIMA = "AutoARIMA"
    AUTOETS = "AutoETS"
    SEASONAL_NAIVE = "SeasonalNaive"
    PATCHTST = "PatchTST"
    TIMESNET = "TimesNet"


class Granularity(str, Enum):
    """Forecast granularity options"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


# --- Prediction Request/Response ---

class PredictionRequest(BaseModel):
    """Request model for making predictions"""
    store_id: Optional[int] = Field(None, description="Store ID (optional)")
    product_id: Optional[int] = Field(None, description="Product ID (optional)")
    weeks_ahead: int = Field(..., ge=1, le=52, description="Number of weeks to forecast (1-52)")
    model_name: Optional[str] = Field(None, description="Specific model to use (default: active model)")
    granularity: Granularity = Field(Granularity.WEEKLY, description="Forecast granularity")

    class Config:
        json_schema_extra = {
            "example": {
                "store_id": 1,
                "product_id": 101,
                "weeks_ahead": 4,
                "model_name": "xgboost_retail_v1",
                "granularity": "weekly"
            }
        }


class SHAPValue(BaseModel):
    """Individual SHAP value explanation"""
    feature: str
    value: float
    importance: float


class ForecastPoint(BaseModel):
    """Single forecast data point"""
    date: str
    predicted_value: float
    confidence_lower: Optional[float] = None
    confidence_upper: Optional[float] = None


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction_id: int
    model_name: str
    model_type: str
    store_id: Optional[int]
    product_id: Optional[int]
    forecasts: List[ForecastPoint]
    shap_values: List[SHAPValue]
    features_used: Dict[str, Any]
    created_at: str
    metadata: Dict[str, Any] = {}

    class Config:
        json_schema_extra = {
            "example": {
                "prediction_id": 123,
                "model_name": "xgboost_retail_v1",
                "model_type": "XGBoost",
                "store_id": 1,
                "product_id": 101,
                "forecasts": [
                    {"date": "2024-01-15", "predicted_value": 1500.50, "confidence_lower": 1400.0, "confidence_upper": 1600.0}
                ],
                "shap_values": [
                    {"feature": "promotion", "value": 1.0, "importance": 0.35},
                    {"feature": "lag_1", "value": 1450.0, "importance": 0.25}
                ],
                "features_used": {"promotion": 1, "lag_1": 1450.0, "month": 1},
                "created_at": "2024-01-10T10:30:00",
                "metadata": {"training_data_period": "2020-2024"}
            }
        }


# --- Data Refresh ---

class DataRefreshResponse(BaseModel):
    """Response model for data refresh operation"""
    status: str
    message: str
    records_updated: int
    new_categories: Optional[int] = None
    last_fetch_time: str
    sources_updated: List[str]

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Data refreshed successfully",
                "records_updated": 1523,
                "new_categories": 2,
                "last_fetch_time": "2024-01-10T10:30:00",
                "sources_updated": ["FRED", "MRTS"]
            }
        }


# --- Model Metadata ---

class ModelMetrics(BaseModel):
    """Model accuracy metrics"""
    rmse: float = Field(..., ge=0, description="Root Mean Square Error")
    mae: float = Field(..., ge=0, description="Mean Absolute Error")
    r2: float = Field(..., ge=0, le=1, description="R-squared score")
    mape: Optional[float] = Field(None, ge=0, description="Mean Absolute Percentage Error")
    training_samples: int = Field(..., ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "rmse": 0.75,
                "mae": 0.50,
                "r2": 0.94,
                "mape": 5.2,
                "training_samples": 10593
            }
        }


class ModelInfo(BaseModel):
    """Model metadata information"""
    id: int
    model_name: str
    model_type: ModelType
    training_date: str
    metrics: ModelMetrics
    hyperparameters: Optional[Dict[str, Any]] = None
    file_path: str
    is_active: bool
    created_at: str
    updated_at: str

    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "model_name": "xgboost_retail_v1",
                "model_type": "XGBoost",
                "training_date": "2024-01-01T00:00:00",
                "metrics": {
                    "rmse": 0.75,
                    "mae": 0.50,
                    "r2": 0.94,
                    "mape": 5.2,
                    "training_samples": 10593
                },
                "hyperparameters": {"n_estimators": 200, "max_depth": 6},
                "file_path": "models/xgboost_v1.pkl",
                "is_active": True,
                "created_at": "2024-01-01T10:00:00",
                "updated_at": "2024-01-01T10:00:00"
            }
        }


class ModelsListResponse(BaseModel):
    """Response with list of models"""
    models: List[ModelInfo]
    total_count: int
    active_count: int


# --- Prediction History ---

class PredictionHistoryFilter(BaseModel):
    """Filters for prediction history query"""
    model_name: Optional[str] = None
    store_id: Optional[int] = None
    product_id: Optional[int] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    include_validated_only: bool = False
    limit: int = Field(100, ge=1, le=1000)


class PredictionHistoryItem(BaseModel):
    """Single prediction history item"""
    id: int
    model_name: str
    store_id: Optional[int]
    product_id: Optional[int]
    prediction_date: str
    predicted_value: float
    actual_value: Optional[float] = None
    confidence_interval_lower: Optional[float] = None
    confidence_interval_upper: Optional[float] = None
    is_validated: bool = False
    error_percentage: Optional[float] = None
    error_absolute: Optional[float] = None
    confidence_score: Optional[float] = None
    shap_values: Optional[Dict[str, float]] = None
    created_at: str


class PredictionHistoryResponse(BaseModel):
    """Response with prediction history"""
    predictions: List[PredictionHistoryItem]
    total_count: int
    filters_applied: Dict[str, Any]
    accuracy_summary: Optional[Dict[str, float]] = None


# --- Validation ---

class ValidationRequest(BaseModel):
    """Request to validate a prediction with actual value"""
    prediction_id: int = Field(..., description="Prediction ID to validate")
    actual_value: float = Field(..., gt=0, description="Actual sales value")
    notes: Optional[str] = Field(None, description="Optional notes about validation")

    class Config:
        json_schema_extra = {
            "example": {
                "prediction_id": 123,
                "actual_value": 1525.75,
                "notes": "Actual sales data from POS system"
            }
        }


class ValidationResponse(BaseModel):
    """Response after validation"""
    prediction_id: int
    previous_predicted_value: float
    new_actual_value: float
    error_absolute: float
    error_percentage: float
    is_validated: bool
    message: str


# --- SHAP Explanation ---

class SHAPExplanationRequest(BaseModel):
    """Request for detailed SHAP explanation"""
    prediction_id: int = Field(..., description="Prediction ID to explain")
    top_n: Optional[int] = Field(10, ge=1, le=50, description="Number of top features to return")

    class Config:
        json_schema_extra = {
            "example": {
                "prediction_id": 123,
                "top_n": 10
            }
        }


class SHAPExplanationResponse(BaseModel):
    """Detailed SHAP explanation response"""
    prediction_id: int
    model_name: str
    prediction_date: str
    predicted_value: float
    base_value: float
    feature_contributions: List[SHAPValue]
    total_shap_value: float
    summary: str


# --- Training ---

class TrainingRequest(BaseModel):
    """Request to train models"""
    model_types: List[ModelType] = Field(
        default=[ModelType.LIGHTGBM, ModelType.RANDOM_FOREST],
        description="List of model types to train"
    )
    force_retrain: bool = Field(False, description="Force retraining even if models exist")
    test_size: float = Field(0.2, ge=0.1, le=0.5, description="Test set proportion")
    hyperparameters: Optional[Dict[str, Dict[str, Any]]] = Field(
        None,
        description="Custom hyperparameters for each model type"
    )


class TrainingResponse(BaseModel):
    """Response after training"""
    status: str
    models_trained: List[str]
    training_time_seconds: float
    metrics: Dict[str, ModelMetrics]
    message: str


# --- Error Responses ---

class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str
    detail: str
    status_code: int
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ValidationError(BaseModel):
    """Validation error detail"""
    field: str
    message: str
    accepted_values: Optional[List[str]] = None
