"""
Category-based prediction routes for RetailPRED
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import logging
import sys
from pathlib import Path

# Add app directory to Python path
app_path = Path(__file__).parent.parent
if str(app_path) not in sys.path:
    sys.path.insert(0, str(app_path))

# Import lightweight categories module (fast!)
from ml.categories import get_available_categories, RETAIL_CATEGORIES

# Import heavy modules only when needed
from ml.multi_resolution_inference import (
    get_available_models_for_category,
    generate_forecast,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/categories", tags=["categories"])


# --- Schemas ---

class RetailCategoryOption(BaseModel):
    """Retail category option"""
    key: str
    display_name: str


class CategoriesListResponse(BaseModel):
    """Response with list of available categories"""
    categories: List[RetailCategoryOption]
    total_count: int


class CategoryPredictionRequest(BaseModel):
    """Request for category-based prediction"""
    category: str = Field(..., description="Retail category key (e.g., 'total_sales', 'automobile_dealers')")
    model_type: str = Field(None, description="Model type: 'lightgbm' or 'randomforest' (default: best model)")
    features: Dict[str, Any] = Field(..., description="Feature values for prediction")

    class Config:
        json_schema_extra = {
            "example": {
                "category": "total_retail_sales",
                "model_type": "xgboost",
                "features": {
                    "month": 6,
                    "day_of_week": 2,
                    "is_weekend": 0,
                    "lag_1": 25000,
                    "lag_2": 24500,
                    "lag_3": 24800,
                    "lag_4": 25200,
                    "lag_8": 24000,
                    "lag_12": 26000,
                    "rolling_mean_3": 24700,
                    "rolling_std_3": 500,
                    "rolling_mean_6": 24800,
                    "rolling_std_6": 600,
                    "rolling_mean_12": 25000,
                    "rolling_std_12": 800,
                    "diff_1": 500,
                    "diff_12": 2000,
                    "pct_change_1": 0.02,
                    "pct_change_12": 0.08,
                    "month_sin": 0.866,
                    "month_cos": 0.5,
                    "quarter_sin": -1.0,
                    "quarter_cos": 0.0
                }
            }
        }


class SHAPValue(BaseModel):
    """SHAP value explanation"""
    feature: str
    value: float
    importance: float


class CategoryPredictionResponse(BaseModel):
    """Response for category-based prediction"""
    category: str
    category_display_name: str
    model_name: str
    model_type: str
    predicted_value: float
    shap_values: List[SHAPValue]
    features_used: Dict[str, Any]
    metadata: Dict[str, Any]

    class Config:
        json_schema_extra = {
            "example": {
                "category": "total_retail_sales",
                "category_display_name": "Total Retail Sales",
                "model_name": "total_retail_sales_xgboost_forecaster",
                "model_type": "xgboost",
                "predicted_value": 26786.50,
                "shap_values": [
                    {"feature": "rolling_mean_3", "value": -558.32, "importance": 0.15},
                    {"feature": "lag_12", "value": 448.21, "importance": 0.12}
                ],
                "features_used": {"month": 6, "lag_1": 25000},
                "metadata": {"feature_count": 23}
            }
        }


# --- Routes ---

@router.get("/list", response_model=CategoriesListResponse)
async def list_categories():
    """
    Get list of all available retail categories

    Returns:
        List of retail categories with keys and display names
    """
    try:
        categories = get_available_categories()

        return CategoriesListResponse(
            categories=categories,
            total_count=len(categories)
        )
    except Exception as e:
        logger.error(f"Error listing categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict", response_model=CategoryPredictionResponse)
async def predict_category(request: CategoryPredictionRequest):
    """
    Make a prediction for a specific retail category

    Args:
        request: Prediction request with category, model type, and features

    Returns:
        Prediction with SHAP explanations

    Raises:
        HTTPException: If category or model not found
    """
    try:
        # Validate category
        available_categories = get_available_categories()
        if request.category not in available_categories:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid category '{request.category}'. Available categories: {available_categories}"
            )

        # Get model name
        model_name = get_model_name_for_category(request.category, request.model_type)

        # Get predictor and generate prediction
        predictor = get_predictor(model_name)

        if predictor.model is None:
            raise HTTPException(
                status_code=404,
                detail=f"Model not found: {model_name}. Please train the model first."
            )

        # Generate prediction
        predicted_value, shap_values, metadata = predictor.predict(
            request.features,
            return_shap=True
        )

        # Convert SHAP values to response format
        shap_list = []
        if shap_values:
            for feature, value in shap_values.items():
                shap_list.append(SHAPValue(
                    feature=feature,
                    value=value,
                    importance=abs(value)
                ))

        # Sort by importance
        shap_list.sort(key=lambda x: x.importance, reverse=True)

        return CategoryPredictionResponse(
            category=request.category,
            category_display_name=get_category_display_name(request.category),
            model_name=model_name,
            model_type=request.model_type,
            predicted_value=predicted_value,
            shap_values=shap_list[:10],  # Top 10 features
            features_used=request.features,
            metadata=metadata
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{category}/models")
async def list_category_models(category: str):
    """
    List available models for a specific category

    Args:
        category: Retail category key

    Returns:
        List of available model types for this category
    """
    try:
        models_info = get_available_models_for_category(category)

        if not models_info["available_models"]:
            raise HTTPException(
                status_code=404,
                detail=f"No models found for category '{category}'"
            )

        return models_info

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing category models: {e}")
        raise HTTPException(status_code=500, detail=str(e))
