"""
Generate Simple Time Series Comparison Plots for All Models

Creates only time series comparison (Actual vs Predicted) for each model
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = "/Users/olivialiau/retailPRED/data/retailpred.db"
OUTPUT_DIR = Path("/Users/olivialiau/retailPRED/training_outputs/visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORY_DISPLAY = {
    "total_sales": "Total Retail Sales",
    "building_materials": "Building Materials & Garden",
    "automobile_dealers": "Automobile Dealers",
    "gasoline_stations": "Gasoline Stations",
    "food_beverage": "Food & Beverage Stores",
    "health_personal_care": "Health & Personal Care",
    "general_merchandise": "General Merchandise",
    "furniture_home_furnishings": "Furniture & Home Furnishings",
    "clothing_accessories": "Clothing & Accessories",
    "sporting_goods_hobby": "Sporting Goods & Hobby",
    "electronics_and_appliances": "Electronics & Appliances",
}

MODEL_TYPES = ["LGBM", "RandomForest", "AutoARIMA", "AutoETS", "SeasonalNaive", "PatchTST", "TimesNet"]


def get_predictions(category: str, model_type: str) -> pd.DataFrame:
    """Get predictions from database"""
    conn = sqlite3.connect(DB_PATH)
    model_name = f"{category}_{model_type}_model"

    query = """
        SELECT prediction_date, predicted_value, actual_value
        FROM prediction_log
        WHERE model_name = ?
        AND actual_value IS NOT NULL
        ORDER BY prediction_date ASC
    """

    df = pd.read_sql_query(query, conn, params=(model_name,))
    conn.close()
    return df


def create_timeseries_plot(category: str, model_type: str, df: pd.DataFrame):
    """Create simple time series comparison plot"""
    if len(df) == 0:
        logger.info(f"  ✗ {model_type}: No data with actual values")
        return

    category_display = CATEGORY_DISPLAY.get(category, category)
    output_subdir = OUTPUT_DIR / category_display.replace(" ", "_").replace("&", "and")
    output_subdir.mkdir(parents=True, exist_ok=True)

    # Calculate metrics
    y_true = df['actual_value'].values
    y_pred = df['predicted_value'].values
    mape = round(abs((y_true - y_pred) / y_true).mean() * 100, 2)

    # Create plot
    plt.figure(figsize=(14, 6))

    # Plot actual vs predicted
    plt.plot(df['prediction_date'], df['actual_value'],
             label='Actual', linewidth=2.5, marker='o', markersize=5, color='#2E86AB')
    plt.plot(df['prediction_date'], df['predicted_value'],
             label='Predicted', linewidth=2.5, marker='s', markersize=5, color='#A23B72', linestyle='--')

    # Formatting
    plt.xlabel('Date', fontsize=12, fontweight='bold')
    plt.ylabel('Sales Value ($)', fontsize=12, fontweight='bold')
    plt.title(f'{category_display} - {model_type} Performance\nMAPE: {mape}%',
              fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save
    filename = f"{category_display.replace(' ', '_')}_{model_type}_performance.png"
    filepath = output_subdir / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"  ✓ {model_type}: MAPE={mape}% ({len(df)} points)")


def main():
    logger.info("=" * 80)
    logger.info("Generating Time Series Plots")
    logger.info("=" * 80)

    categories = list(CATEGORY_DISPLAY.keys())
    total = 0
    skipped = 0

    for category in categories:
        category_display = CATEGORY_DISPLAY.get(category)
        logger.info(f"\n{category_display}")

        for model_type in MODEL_TYPES:
            try:
                df = get_predictions(category, model_type)
                create_timeseries_plot(category, model_type, df)
                if len(df) > 0:
                    total += 1
                else:
                    skipped += 1
            except Exception as e:
                logger.error(f"  ✗ {model_type}: {str(e)[:50]}")
                skipped += 1

    logger.info("\n" + "=" * 80)
    logger.info(f"✓ Generated {total} plots")
    logger.info(f"✗ Skipped {skipped}")
    logger.info(f"📁 {OUTPUT_DIR}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
