#!/usr/bin/env python
"""
Unified Training Pipeline for RetailPRED
Uses original sophisticated algorithms and saves models for backend API compatibility
"""

import sys
from pathlib import Path
import logging
import json
import joblib
from datetime import datetime

# Add paths
project_root = Path(__file__).parent.parent / "project_root"
backend_path = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(backend_path))

# Import original trainer
from models.robust_timecopilot_trainer import RobustTimeCopilotTrainer, logger
from ml.data_loader import RETAIL_CATEGORIES, CATEGORY_DISPLAY_NAMES

def main():
    """Main training function using original pipeline"""

    print("=" * 80)
    print("UNIFIED TRAINING PIPELINE")
    print("Using original sophisticated algorithms + backend API compatibility")
    print("=" * 80)

    # Initialize original trainer
    # Using statistical-only for fast training with best models
    trainer = RobustTimeCopilotTrainer(
        results_dir=str(project_root / "results"),
        data_dir=str(project_root / "data_processed"),
        output_dir=str(project_root.parent / "training_outputs"),
        check_data=True,  # Check data exists
        disable_ai_agents=True,  # Disable AI agents (avoid API costs)
        statistical_only=False,  # Include traditional ML
        traditional_only=True,  # Use only traditional (no neural nets for speed)
    )

    # Get category mappings
    category_map = {
        'Total_Retail_Sales': 'total_retail_sales',
        'Automobile_Dealers': 'automobile_dealers',
        'Building_Materials_Garden': 'building_materials_garden',
        'Clothing_Accessories': 'clothing_accessories',
        'Electronics_and_Appliances': 'electronics_appliances',
        'Food_Beverage_Stores': 'food_beverage_stores',
        'Furniture_Home_Furnishings': 'furniture_home_furnishings',
        'Gasoline_Stations': 'gasoline_stations',
        'General_Merchandise': 'general_merchandise',
        'Health_Personal_Care': 'health_personal_care',
        'Nonstore_Retailers': 'nonstore_retailers',
        'Sporting_Goods_Hobby': 'sporting_goods_hobby',
    }

    # Reverse mapping
    key_to_display = {v: k for k, v in category_map.items()}

    # Train all categories
    logger.info(f"\nTraining {len(category_map)} retail categories...")

    results = trainer.train_categories(
        categories=list(category_map.keys())
    )

    # Save models for backend API
    models_dir = backend_path / "ml" / "models"
    models_dir.mkdir(exist_ok=True, parents=True)

    logger.info(f"\n{'='*80}")
    logger.info("Saving models for backend API...")
    logger.info(f"{'='*80}")

    backend_results = {}

    for category_display, category_key in category_map.items():
        if category_display not in results:
            logger.warning(f"No results for {category_display}")
            continue

        logger.info(f"\n{category_display}:")

        category_results = results[category_display]
        backend_results[category_key] = {}

        # Save each trained model
        for model_name, model_obj in trainer.trained_models.items():
            try:
                if category_display not in trainer.trained_models:
                    continue

                # Skip if this model wasn't trained for this category
                if not hasattr(model_obj, 'model') or model_obj.model is None:
                    continue

                # Backend-compatible model name
                backend_model_name = f"{category_key}_{model_name.lower()}_forecaster"

                # Save model
                model_path = models_dir / f"{backend_model_name}.pkl"
                joblib.dump(model_obj, model_path)

                logger.info(f"  ✓ Saved: {backend_model_name}")

                # Save metadata
                if category_display in category_results:
                    metrics = category_results[model_name]

                    metadata = {
                        "model_name": backend_model_name,
                        "category": category_key,
                        "category_display": category_display,
                        "model_type": model_name,
                        "metrics": metrics,
                        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }

                    metadata_path = models_dir / f"{backend_model_name}_metadata.json"
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)

                    backend_results[category_key][model_name] = {
                        "status": "success",
                        "metrics": metrics
                    }

            except Exception as e:
                logger.error(f"  ✗ Failed to save {model_name}: {e}")

    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = backend_path.parent / "training_outputs" / f"unified_pipeline_summary_{timestamp}.json"
    summary_path.parent.mkdir(exist_ok=True, parents=True)

    with open(summary_path, 'w') as f:
        json.dump(backend_results, f, indent=2)

    # Print summary
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"\nModels saved to: {models_dir}")
    print(f"Summary saved to: {summary_path}")
    print(f"\nAvailable models per category:")

    for category_key, models in backend_results.items():
        display_name = CATEGORY_DISPLAY_NAMES.get(category_key, category_key)
        print(f"\n{display_name}:")
        for model_type, result in models.items():
            if result['status'] == 'success':
                metrics = result['metrics']
                print(f"  {model_type}: MAPE={metrics.get('mape', 0):.2f}%")

    print(f"\n{'='*80}")
    print("Models can now be used via:")
    print("  - Backend API: POST /api/categories/predict")
    print("  - Python: ml.inference.get_predictor()")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
