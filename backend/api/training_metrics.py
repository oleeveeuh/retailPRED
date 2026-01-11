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
    Get flattened list of all models with their VALIDATION metrics

    Returns array of model objects with VALIDATION metrics from actual predictions (not training metrics)
    """
    try:
        # Load validation metrics from prediction_log table
        validation_file = TRAINING_OUTPUTS / "validation_metrics.json"

        if not validation_file.exists():
            raise HTTPException(
                status_code=404,
                detail="Validation metrics not found. Please run backend/calculate_validation_metrics.py"
            )

        with open(validation_file, 'r') as f:
            validation_data = json.load(f)

        models = []
        validation_models = validation_data.get('models', {})

        # Convert validation metrics to frontend format
        for model_name, model_data in validation_models.items():
            # Extract category and model type from model_name
            # Format: "category_model_type_model"
            parts = model_name.rsplit('_', 2)  # Split from right into max 3 parts
            if len(parts) >= 3:
                category = '_'.join(parts[:-2])
                model_type = parts[-2] + '_' + parts[-1]  # e.g., "LGBM_model" -> "LGBM"
            elif len(parts) == 2:
                category = parts[0]
                model_type = parts[1]
            else:
                category = model_name
                model_type = "Unknown"

            # Clean up model_type (remove "_model" suffix if present)
            model_type = model_type.replace('_model', '').replace('_', ' ')

            models.append({
                'id': model_name,
                'model_name': model_name,
                'model_type': model_type,
                'category': category,
                'metrics': model_data['metrics'],
                'training_date': validation_data.get('generated_at', '2026-01-04'),
                'is_active': model_type in ['LGBM', 'RandomForest', 'TimesNet', 'PatchTST'],
                'validated_predictions': model_data.get('validated_predictions', 0),
                'total_predictions': model_data.get('total_predictions', 0),
            })

        return {
            'models': models,
            'total_count': len(models),
            'active_count': sum(1 for m in models if m['is_active'])
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading validation metrics: {e}")
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
