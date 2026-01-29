#!/usr/bin/env python3
"""
Save Generated Predictions to Database

This script takes predictions from generate_rolling_predictions.py
and inserts them into the prediction_log table.

Features:
- Checks for duplicates (skips if model_name + prediction_date already exists)
- Marks new predictions as is_validated = FALSE
- Logs summary of predictions added

Usage:
    python save_predictions_to_db.py [--input predictions.json] [--force]
"""

import sys
import os
import argparse
import logging
import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATABASE_PATH

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent.parent / 'logs' / 'save_predictions.log')
    ]
)
logger = logging.getLogger(__name__)


def check_existing_predictions(predictions: List[Dict[str, Any]]) -> Dict[str, set]:
    """
    Check which predictions already exist in the database

    Args:
        predictions: List of prediction dictionaries

    Returns:
        Dictionary of {model_name: set of existing dates}
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    # Get unique model names
    model_names = list(set(p.get('model_name', '') for p in predictions))
    model_names = [m for m in model_names if m]

    if not model_names:
        conn.close()
        return {}

    placeholders = ','.join('?' * len(model_names))
    query = f"""
        SELECT DISTINCT model_name, prediction_date
        FROM prediction_log
        WHERE model_name IN ({placeholders})
    """

    cursor.execute(query, model_names)
    existing = cursor.fetchall()
    conn.close()

    # Build dictionary of existing predictions
    existing_dict = {}
    for model_name, pred_date in existing:
        if model_name not in existing_dict:
            existing_dict[model_name] = set()
        existing_dict[model_name].add(pred_date)

    return existing_dict


def save_predictions(
    predictions: List[Dict[str, Any]],
    force: bool = False
) -> Tuple[int, int, int]:
    """
    Save predictions to the database

    Args:
        predictions: List of prediction dictionaries
        force: If True, overwrite existing predictions

    Returns:
        Tuple of (added_count, skipped_count, error_count)
    """
    if not predictions:
        logger.warning("No predictions to save")
        return 0, 0, 0

    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    # Check existing predictions if not forcing
    existing_dict = {} if force else check_existing_predictions(predictions)

    added_count = 0
    skipped_count = 0
    error_count = 0

    for pred in predictions:
        model_name = pred.get('model_name', '')
        prediction_date = pred.get('prediction_date', '')
        predicted_value = pred.get('predicted_value')
        ci_lower = pred.get('confidence_interval_lower')
        ci_upper = pred.get('confidence_interval_upper')

        if not model_name or not prediction_date or predicted_value is None:
            error_count += 1
            continue

        # Check if already exists
        if not force and model_name in existing_dict:
            if prediction_date in existing_dict[model_name]:
                skipped_count += 1
                continue

        try:
            cursor.execute("""
                INSERT INTO prediction_log (
                    model_name,
                    prediction_date,
                    predicted_value,
                    confidence_interval_lower,
                    confidence_interval_upper,
                    actual_value,
                    error_absolute,
                    error_percentage,
                    is_validated,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL, 0, ?)
            """, (
                model_name,
                prediction_date,
                predicted_value,
                ci_lower,
                ci_upper,
                datetime.now().isoformat()
            ))

            added_count += 1

        except sqlite3.IntegrityError:
            # Duplicate key (shouldn't happen with check above, but just in case)
            skipped_count += 1
        except Exception as e:
            logger.error(f"Error inserting {model_name}/{prediction_date}: {e}")
            error_count += 1

    conn.commit()
    conn.close()

    return added_count, skipped_count, error_count


def load_predictions_from_file(input_path: str) -> List[Dict[str, Any]]:
    """
    Load predictions from JSON file

    Args:
        input_path: Path to predictions JSON file

    Returns:
        List of prediction dictionaries
    """
    with open(input_path, 'r') as f:
        data = json.load(f)

    if 'predictions' in data:
        return data['predictions']
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("Invalid predictions file format")


def print_summary(added: int, skipped: int, errors: int):
    """Print summary of save operation"""
    total = added + skipped + errors

    logger.info("")
    logger.info("=" * 80)
    logger.info("SAVE SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total predictions processed: {total}")
    logger.info(f"✓ Added to database:    {added}")
    logger.info(f"⊘ Skipped (duplicate):  {skipped}")
    logger.info(f"✗ Errors:              {errors}")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Save generated predictions to database"
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Path to predictions JSON file'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing predictions'
    )

    args = parser.parse_args()

    if not args.input:
        # Try to find the most recent predictions file
        predictions_dir = Path(__file__).parent / "predictions"
        if predictions_dir.exists():
            files = list(predictions_dir.glob("predictions_*.json"))
            if files:
                args.input = str(max(files, key=lambda p: p.stat().st_mtime))
                logger.info(f"Using latest predictions file: {args.input}")

    if not args.input:
        logger.error("No input file specified. Use --input predictions.json")
        return 1

    try:
        predictions = load_predictions_from_file(args.input)
        logger.info(f"Loaded {len(predictions)} predictions from {args.input}")

        added, skipped, errors = save_predictions(predictions, force=args.force)

        print_summary(added, skipped, errors)

        if errors > 0:
            return 1

        return 0

    except Exception as e:
        logger.error(f"Save failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
