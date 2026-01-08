"""
Register all trained models in the database
Populates the model_metadata table with trained model information
"""

import sys
from pathlib import Path
import json
import logging

# Add paths
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from db.database import RetailPREDDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Map model types to ModelType enum values
MODEL_TYPE_MAP = {
    "LGBM": "LGBM",
    "RandomForest": "RandomForest",
    "AutoARIMA": "AutoARIMA",
    "AutoETS": "AutoETS",
    "SeasonalNaive": "SeasonalNaive",
    "PatchTST": "PatchTST",
    "TimesNet": "TimesNet",
}

# Training base path
TRAINING_OUTPUTS = Path(__file__).parent.parent.parent / "training_outputs"
MODELS_DIR = TRAINING_OUTPUTS / "models"

# Load training summary
summary_path = TRAINING_OUTPUTS / "robust_training_summary.json"
with open(summary_path) as f:
    training_summary = json.load(f)

# Extract per-model per-category metrics from summary
# We'll need to read the individual category results
category_metrics = {}

# Find all category result files
for category_dir in MODELS_DIR.iterdir():
    if not category_dir.is_dir():
        continue

    category = category_dir.name
    category_metrics[category] = {}

    # Check for model_predictions.json
    predictions_file = category_dir / "../model_predictions.json"
    if predictions_file.exists():
        with open(predictions_file) as f:
            pred_data = json.load(f)

        if category in pred_data:
            for model_name, model_results in pred_data[category].items():
                if "test_mape" in model_results:
                    category_metrics[category][model_name] = {
                        "mape": model_results["test_mape"],
                        "rmse": model_results.get("test_rmse", 0),
                        "mae": model_results.get("test_mae", 0),
                        "r2": model_results.get("test_r2", 0),
                    }

# Also check in training_outputs root
root_predictions = TRAINING_OUTPUTS / "model_predictions.json"
if root_predictions.exists():
    with open(root_predictions) as f:
        pred_data = json.load(f)

    for category, models in pred_data.items():
        if category not in category_metrics:
            category_metrics[category] = {}
        for model_name, model_results in models.items():
            if "test_mape" in model_results:
                category_metrics[category][model_name] = {
                    "mape": model_results["test_mape"],
                    "rmse": model_results.get("test_rmse", 0),
                    "mae": model_results.get("test_mae", 0),
                    "r2": model_results.get("test_r2", 0),
                }

# Initialize database
db_path = Path(__file__).parent.parent.parent / "data" / "retailpred.db"
db = RetailPREDDatabase(db_path=str(db_path))

# Register all models
registered_count = 0
skipped_count = 0

for category_dir in MODELS_DIR.iterdir():
    if not category_dir.is_dir():
        continue

    category = category_dir.name

    # Map display name to key
    category_map = {
        "Total_Retail_Sales": "total_sales",
        "General_Merchandise": "general_merchandise_stores",
        "Sporting_Goods_Hobby": "sporting_goods_hobby_and_musical_instrument_stores",
        "Building_Materials_Garden": "building_material_and_garden_equipment",
        "Furniture_Home_Furnishings": "furniture_and_home_furnishings_stores",
    }

    category_key = category_map.get(category, category.lower().replace(" ", "_"))

    for model_file in category_dir.glob("*.pkl"):
        model_type = model_file.stem.replace("_model", "")

        # Skip if not in our model type map
        if model_type not in MODEL_TYPE_MAP:
            logger.warning(f"Skipping unknown model type: {model_type}")
            continue

        # Get metrics from training results
        metrics = {
            "mape": 0.0,
            "rmse": 0.0,
            "mae": 0.0,
            "r2": 0.0,
            "training_samples": 1000,  # Default value
        }

        if category in category_metrics and model_type in category_metrics[category]:
            metrics.update(category_metrics[category][model_type])

        # Use model type statistics from summary as fallback
        if metrics["mape"] == 0 and model_type in training_summary["models"]["statistics"]:
            stats = training_summary["models"]["statistics"][model_type]
            metrics["mape"] = stats.get("avg_mape", 0)

        # Create model name
        model_name = f"{category_key}_{model_type.lower()}_model"

        # File path
        file_path = str(model_file.absolute())

        try:
            model_id = db.register_model(
                model_name=model_name,
                model_type=MODEL_TYPE_MAP[model_type],
                file_path=file_path,
                metrics=metrics,
                is_active=True,
            )
            logger.info(f"✓ Registered {model_name} (ID: {model_id}) - MAPE: {metrics['mape']:.2f}%")
            registered_count += 1
        except Exception as e:
            logger.warning(f"✗ Failed to register {model_name}: {e}")
            skipped_count += 1

logger.info(f"\n{'='*60}")
logger.info(f"Registration Complete")
logger.info(f"  Registered: {registered_count} models")
logger.info(f"  Skipped: {skipped_count} models")
logger.info(f"{'='*60}")

# Verify
conn = db.get_connection()
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM model_metadata")
total = cursor.fetchone()[0]
cursor.execute("SELECT COUNT(*) FROM model_metadata WHERE is_active = 1")
active = cursor.fetchone()[0]
conn.close()

logger.info(f"\nDatabase Verification:")
logger.info(f"  Total models in database: {total}")
logger.info(f"  Active models: {active}")
