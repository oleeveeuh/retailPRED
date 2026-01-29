"""
RetailPRED Configuration Module

Centralized configuration for paths, API settings, and external services.
This is the single source of truth for all configuration.

Environment variables override defaults (set in .env or system environment).
"""

import os
from pathlib import Path
from typing import Optional

# ============================================================================
# Project Structure
# ============================================================================

# Project root - automatically detected
PROJECT_ROOT = Path(__file__).parent

# ============================================================================
# Database Configuration
# ============================================================================

# Database path - single canonical location
DATABASE_PATH = PROJECT_ROOT / "data" / "retailpred.db"

# ============================================================================
# Model Storage Paths
# ============================================================================

MODELS_DIR = PROJECT_ROOT / "models"
BACKEND_MODELS_DIR = PROJECT_ROOT / "backend" / "ml" / "models"
TRAINING_OUTPUTS_DIR = PROJECT_ROOT / "training_outputs"

# Model files
MODEL_LATEST_PATH = MODELS_DIR / "model_latest.pkl"
METRICS_LATEST_PATH = MODELS_DIR / "latest_metrics.json"

# ============================================================================
# Data Paths
# ============================================================================

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = PROJECT_ROOT / "project_root" / "data_processed"
MULTI_RESOLUTION_DATA_DIR = PROJECT_ROOT / "project_root" / "data_multi_resolution"

# ============================================================================
# API Configuration (for deployment)
# ============================================================================

# Frontend URL (Vercel deployment)
VERCEL_URL = os.getenv("VERCEL_URL", "https://retailpred.vercel.app")
# Backend API URL (for Docker/production)
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")
# Frontend local development URL
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# ============================================================================
# External Data Sources
# ============================================================================

# FRED API (Federal Reserve Economic Data)
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
FRED_BASE_URL = "https://api.stlouisfed.org/fred"

# MRTS API (Monthly Retail Trade Survey)
MRTS_BASE_URL = "https://api.census.gov/data/timeseries/eits/mrts"
MRTS_TIMEOUT = 60

# Yahoo Finance (for market data features)
YFINANCE_ENABLED = True
YFINANCE_RATE_LIMIT = 2000  # requests per hour

# ============================================================================
# Retail Categories
# ============================================================================

RETAIL_CATEGORIES = {
    "total_retail_sales": "Total_Retail_Sales",
    "automobile_dealers": "Automobile_Dealers",
    "building_materials_garden": "Building_Materials_Garden",
    "clothing_accessories": "Clothing_Accessories",
    "electronics_appliances": "Electronics_and_Appliances",
    "food_beverage_stores": "Food_Beverage_Stores",
    "furniture_home_furnishings": "Furniture_Home_Furnishings",
    "gasoline_stations": "Gasoline_Stations",
    "general_merchandise": "General_Merchandise",
    "health_personal_care": "Health_Personal_Care",
    "nonstore_retailers": "Nonstore_Retailers",
    "sporting_goods_hobby": "Sporting_Goods_Hobby",
}

# ============================================================================
# Airflow Configuration
# ============================================================================

# Airflow home directory
AIRFLOW_HOME = os.getenv("AIRFLOW_HOME", "/home/oliau/airflow")
AIRFLOW_DAGS_FOLDER = os.getenv("AIRFLOW_DAGS_FOLDER", PROJECT_ROOT / "dags")

# Default_conn_id for SQLite
AIRFLOW_CONN_ID = "retailpred_sqlite"

# ============================================================================
# Validation Settings
# ============================================================================

# Anomaly detection threshold (percentage error)
ANOMALY_THRESHOLD_DEFAULT = 10.0

# Validation export path
VALIDATION_METRICS_PATH = PROJECT_ROOT / "data" / "validation_metrics.json"

# ============================================================================
# Logging Configuration
# ============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = PROJECT_ROOT / "logs" / "validation.log"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ============================================================================
# Helper Functions
# ============================================================================

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
        PROJECT_ROOT / "logs",
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_retail_category_key(category_name: str) -> str:
    """
    Convert display name to category key

    Args:
        category_name: Display name (e.g., "Total_Retail_Sales")

    Returns:
        Category key (e.g., "total_retail_sales")
    """
    # Reverse mapping from display name to key
    for key, value in RETAIL_CATEGORIES.items():
        if value == category_name:
            return key
    # Fallback: convert to lowercase with underscores
    return category_name.lower().replace(" ", "_")


def get_category_display_name(category_key: str) -> str:
    """
    Get display name for a category key

    Args:
        category_key: Category key (e.g., "total_retail_sales")

    Returns:
        Display name (e.g., "Total_Retail_Sales")
    """
    return RETAIL_CATEGORIES.get(category_key, category_key)
