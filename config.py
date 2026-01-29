"""
RetailPRED Configuration Module

Centralized configuration for paths and settings.
This is the single source of truth for all file locations.
"""

import os
from pathlib import Path

# Project root - automatically detected
PROJECT_ROOT = Path(__file__).parent

# Database path - single canonical location
# All database access should use this path
DATABASE_PATH = PROJECT_ROOT / "data" / "retailpred.db"

# Model storage paths
MODELS_DIR = PROJECT_ROOT / "models"
# Legacy path for backward compatibility (symlink to backend/ml/models)
BACKEND_MODELS_DIR = PROJECT_ROOT / "backend" / "ml" / "models"
# Training outputs directory
TRAINING_OUTPUTS_DIR = PROJECT_ROOT / "training_outputs"

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = PROJECT_ROOT / "project_root" / "data_processed"
MULTI_RESOLUTION_DATA_DIR = PROJECT_ROOT / "project_root" / "data_multi_resolution"

# Model files
MODEL_LATEST_PATH = MODELS_DIR / "model_latest.pkl"
METRICS_LATEST_PATH = MODELS_DIR / "latest_metrics.json"


def get_db_path() -> str:
    """Get the database path as a string"""
    return str(DATABASE_PATH)


def get_models_dir() -> Path:
    """Get the models directory"""
    return MODELS_DIR


def get_model_path(model_name: str) -> Path:
    """Get the full path for a model file"""
    return BACKEND_MODELS_DIR / model_name


def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        MODELS_DIR,
        DATA_DIR,
        TRAINING_OUTPUTS_DIR / "models",
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Print configuration for debugging
    print("RetailPRED Configuration:")
    print(f"  Project Root: {PROJECT_ROOT}")
    print(f"  Database: {DATABASE_PATH}")
    print(f"  Database exists: {DATABASE_PATH.exists()}")
    print(f"  Models Dir: {MODELS_DIR}")
    print(f"  Backend Models Dir: {BACKEND_MODELS_DIR}")
    print(f"  Processed Data: {PROCESSED_DATA_DIR}")
