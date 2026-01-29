#!/usr/bin/env python3
"""
RetailPRED Database Update Script

This script updates the database with new predictions and metrics.
Intended to be run after model training.

Usage:
    python update_db.py --model-path models/model_latest.pkl
    python update_db.py --category total_sales
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATABASE_PATH, MODELS_DIR


def update_database(model_path: str, category: str = "total_sales") -> dict:
    """
    Update the database with model information

    Args:
        model_path: Path to the trained model file
        category: Category name

    Returns:
        Dictionary with update results
    """
    import sqlite3

    result = {
        "success": False,
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "category": category,
    }

    # Check if model exists
    model_file = Path(model_path)
    if not model_file.exists():
        # Try relative to models directory
        model_file = MODELS_DIR / Path(model_path).name
        if not model_file.exists():
            result["error"] = f"Model file not found: {model_path}"
            return result

    # Load metrics if available
    metrics_path = MODELS_DIR / "latest_metrics.json"
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics_data = json.load(f)
            metrics = metrics_data.get("metrics", {})

    # Connect to database
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        # Check if model_metadata table exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='model_metadata'
        """)

        if cursor.fetchone():
            # Update model metadata
            cursor.execute("""
                INSERT OR REPLACE INTO model_metadata
                (model_name, model_type, training_date, metrics, file_path, is_active, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                f"{category}_latest",
                "ensemble",
                datetime.now().isoformat(),
                json.dumps(metrics),
                str(model_file),
                1,
                datetime.now().isoformat(),
                datetime.now().isoformat(),
            ))
            conn.commit()
            result["success"] = True
            result["message"] = "Database updated successfully"
        else:
            result["warning"] = "model_metadata table not found in database"

        conn.close()
    except Exception as e:
        result["error"] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Update RetailPRED database with model information"
    )
    parser.add_argument(
        "--model-path",
        default=str(MODELS_DIR / "model_latest.pkl"),
        help="Path to trained model file"
    )
    parser.add_argument(
        "--category",
        default="total_sales",
        help="Category name"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("RetailPRED Database Update")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Category: {args.category}")
    print(f"Database: {DATABASE_PATH}")
    print("=" * 60)
    print()

    result = update_database(args.model_path, args.category)

    if result.get("success"):
        print(f"✓ {result.get('message', 'Update successful')}")
        return 0
    else:
        error = result.get("error", result.get("warning", "Unknown error"))
        print(f"✗ {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
