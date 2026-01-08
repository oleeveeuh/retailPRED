"""
Training Metrics API Routes
Serves model training metrics from training summary files
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/training-metrics", tags=["training-metrics"])

# Path to training outputs
TRAINING_OUTPUTS = Path(__file__).parent.parent.parent / "training_outputs"


@router.get("/summary")
async def get_training_summary():
    """
    Get complete training summary with all model metrics

    Returns metrics for all trained models across all categories
    """
    try:
        summary_file = TRAINING_OUTPUTS / "robust_training_summary.json"

        if not summary_file.exists():
            raise HTTPException(
                status_code=404,
                detail="Training summary not found. Please ensure models have been trained."
            )

        with open(summary_file, 'r') as f:
            summary = json.load(f)

        return summary

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading training summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def get_all_model_metrics():
    """
    Get flattened list of all models with their metrics

    Returns array of model objects with metrics for easy frontend consumption
    """
    try:
        summary_file = TRAINING_OUTPUTS / "robust_training_summary.json"

        if not summary_file.exists():
            raise HTTPException(
                status_code=404,
                detail="Training summary not found. Please ensure models have been trained."
            )

        with open(summary_file, 'r') as f:
            summary = json.load(f)

        # Flatten the nested structure into a list of models
        models = []

        # Check if summary has detailed_results structure
        if 'detailed_results' in summary:
            for category_name, category_data in summary['detailed_results'].items():
                if 'models' in category_data:
                    for model_name, model_metrics in category_data['models'].items():
                        # Convert lowercase metric names to uppercase for frontend
                        formatted_metrics = {}
                        for key, value in model_metrics.items():
                            if key in ['mape', 'smape', 'mase', 'rmse', 'mae']:
                                formatted_metrics[key.upper()] = {'mean': value}
                            else:
                                formatted_metrics[key] = value

                        models.append({
                            'id': f"{category_name}_{model_name}",
                            'model_name': f"{category_name}_{model_name}",
                            'model_type': model_name,
                            'category': category_name,
                            'metrics': formatted_metrics,
                            'training_date': summary.get('training_completed', '2026-01-04'),
                            'is_active': model_name in ['LGBM', 'RandomForest'],  # Mark best models as active
                        })

        return {
            'models': models,
            'total_count': len(models),
            'active_count': sum(1 for m in models if m['is_active'])
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading model metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/best-models")
async def get_best_models():
    """
    Get the best performing model for each category

    Returns model with lowest MASE for each retail category
    """
    try:
        summary_file = TRAINING_OUTPUTS / "robust_training_summary.json"

        if not summary_file.exists():
            raise HTTPException(
                status_code=404,
                detail="Training summary not found. Please ensure models have been trained."
            )

        with open(summary_file, 'r') as f:
            summary = json.load(f)

        best_models = []

        # Check if summary has detailed_results structure
        if 'detailed_results' in summary:
            for category_name, category_data in summary['detailed_results'].items():
                if 'models' in category_data:
                    # Find model with best (lowest) MASE
                    models_sorted = sorted(
                        category_data['models'].items(),
                        key=lambda x: x[1].get('mase', float('inf'))
                    )

                    if models_sorted:
                        best_model_name, best_metrics = models_sorted[0]
                        # Convert lowercase metrics to uppercase
                        formatted_metrics = {
                            'MASE': best_metrics.get('mase'),
                            'MAPE': best_metrics.get('mape'),
                            'sMAPE': best_metrics.get('smape'),
                        }
                        best_models.append({
                            'category': category_name,
                            'model_name': best_model_name,
                            'model_type': best_model_name,
                            'metrics': formatted_metrics,
                            'is_active': True
                        })

        return {
            'best_models': best_models,
            'total_count': len(best_models)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading best models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/categories")
async def get_categories_with_models():
    """
    Get list of all categories that have trained models

    Returns category names and metadata
    """
    try:
        summary_file = TRAINING_OUTPUTS / "robust_training_summary.json"

        if not summary_file.exists():
            raise HTTPException(
                status_code=404,
                detail="Training summary not found. Please ensure models have been trained."
            )

        with open(summary_file, 'r') as f:
            summary = json.load(f)

        categories = []

        if 'detailed_results' in summary:
            for category_name, category_data in summary['detailed_results'].items():
                # Find best model
                models_sorted = sorted(
                    category_data.get('models', {}).items(),
                    key=lambda x: x[1].get('mase', float('inf'))
                )

                best_model = models_sorted[0] if models_sorted else (None, {})

                categories.append({
                    'name': category_name,
                    'display_name': category_name.replace('_', ' '),
                    'best_model': best_model[0],
                    'best_mase': best_model[1].get('mase'),
                    'best_mape': best_model[1].get('mape'),
                    'models_count': len(category_data.get('models', {})),
                    'training_date': summary.get('training_completed'),
                    'data_points': category_data.get('data_points')
                })

        return {
            'categories': categories,
            'total_count': len(categories)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))
