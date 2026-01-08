"""
Export RetailPRED database to JSON for static demo deployment

Exports predictions, economic indicators, and summary statistics
for use in Vercel/static deployments without a backend database.
"""

import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database path
DB_PATH = "/Users/olivialiau/retailPRED/data/retailpred.db"

# Output directory
OUTPUT_DIR = Path("/Users/olivialiau/retailPRED/frontend/public/demo-data")


def inspect_database_schema(conn: sqlite3.Connection) -> None:
    """Inspect and print database schema"""
    logger.info("=" * 80)
    logger.info("Inspecting database schema...")
    logger.info("=" * 80)

    cursor = conn.cursor()

    # Get all tables
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table'
        ORDER BY name
    """)
    tables = [row[0] for row in cursor.fetchall()]

    logger.info(f"\nFound {len(tables)} tables:")
    for table in tables:
        logger.info(f"\n  Table: {table}")

        # Get schema
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()

        logger.info(f"  Columns ({len(columns)}):")
        for col in columns:
            col_id, name, type_name, not_null, default_val, primary_key = col
            pk = " [PK]" if primary_key else ""
            null = " NOT NULL" if not_null else ""
            logger.info(f"    - {name}: {type_name}{pk}{null}")

        # Get row count
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            logger.info(f"  Rows: {count:,}")
        except Exception as e:
            logger.info(f"  Rows: Error counting - {e}")

    logger.info("\n" + "=" * 80 + "\n")


def export_predictions(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Export ALL predictions with actual values for validation"""
    logger.info("Exporting predictions...")

    cursor = conn.cursor()

    # Check if table exists
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='prediction_log'
    """)
    if not cursor.fetchone():
        logger.warning("  ⚠ prediction_log table not found")
        return {"data": [], "metadata": {"error": "Table not found"}}

    # Get ALL predictions (not just 100) for full validation display
    cursor.execute("""
        SELECT
            id,
            model_name,
            prediction_date,
            predicted_value,
            actual_value,
            confidence_interval_lower,
            confidence_interval_upper,
            shap_values,
            features,
            created_at,
            error_percentage,
            error_absolute,
            confidence_score
        FROM prediction_log
        ORDER BY prediction_date DESC, created_at DESC
    """)

    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]

    predictions = []
    for row in rows:
        pred = dict(zip(columns, row))

        # Parse JSON fields
        if pred.get('shap_values'):
            try:
                pred['shap_values'] = json.loads(pred['shap_values'])
            except:
                pred['shap_values'] = None

        if pred.get('features'):
            try:
                pred['features'] = json.loads(pred['features'])
            except:
                pred['features'] = None

        predictions.append(pred)

    # Get statistics
    cursor.execute("SELECT COUNT(*) FROM prediction_log")
    total_count = cursor.fetchone()[0]

    cursor.execute("SELECT MIN(prediction_date), MAX(prediction_date) FROM prediction_log")
    min_date, max_date = cursor.fetchone()

    cursor.execute("""
        SELECT DISTINCT SUBSTR(model_name, 1, INSTR(model_name, '_model') - 1)
        FROM prediction_log
        ORDER BY 1
    """)
    models = [row[0] for row in cursor.fetchall()]

    metadata = {
        "export_timestamp": datetime.now().isoformat(),
        "row_count": len(predictions),
        "total_predictions_in_db": total_count,
        "date_range": {
            "start": min_date,
            "end": max_date
        },
        "models": models,
        "note": f"All {len(predictions)} predictions from database"
    }

    logger.info(f"  ✓ Exported {len(predictions)} predictions")
    logger.info(f"    Total in database: {total_count:,}")
    logger.info(f"    Date range: {min_date} to {max_date}")
    logger.info(f"    Models: {len(models)}")

    return {"data": predictions, "metadata": metadata}


def export_economic_indicators(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Export economic indicators if available"""
    logger.info("Exporting economic data...")

    cursor = conn.cursor()

    # Check for economic data tables
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND (
            name LIKE '%economic%' OR
            name LIKE '%indicator%' OR
            name LIKE '%external%'
        )
        ORDER BY name
    """)
    tables = [row[0] for row in cursor.fetchall()]

    if not tables:
        logger.warning("  ⚠ No economic indicator tables found")
        logger.info("    Creating sample economic indicators for demo...")

        # Create sample data for demo
        import random
        from datetime import timedelta

        base_date = datetime.now() - timedelta(days=500)
        indicators = []

        for i in range(500):
            date = base_date + timedelta(days=i)
            indicators.append({
                "date": date.strftime("%Y-%m-%d"),
                "cpi": round(300 + i * 0.1 + random.uniform(-2, 2), 2),
                "interest_rates": round(5.25 + random.uniform(-0.5, 0.5), 2),
                "unemployment": round(3.7 + random.uniform(-0.3, 0.3), 2),
                "consumer_sentiment": round(70 + random.uniform(-5, 5), 2),
                "money_supply": round(20000 + i * 10 + random.uniform(-100, 100), 2),
                "industrial_production": round(105 + random.uniform(-2, 2), 2)
            })

        metadata = {
            "export_timestamp": datetime.now().isoformat(),
            "row_count": len(indicators),
            "source": "synthetic_demo_data",
            "note": "Generated sample data for demo purposes"
        }

        logger.info(f"  ✓ Generated {len(indicators)} sample indicators")
        return {"data": indicators, "metadata": metadata}

    # Export from actual table
    table_name = tables[0]
    logger.info(f"  Found table: {table_name}")

    cursor.execute(f"""
        SELECT * FROM {table_name}
        ORDER BY date DESC
        LIMIT 500
    """)

    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]

    indicators = [dict(zip(columns, row)) for row in rows]

    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    total_count = cursor.fetchone()[0]

    metadata = {
        "export_timestamp": datetime.now().isoformat(),
        "row_count": len(indicators),
        "total_in_db": total_count,
        "source": table_name,
        "note": "Most recent 500 records"
    }

    logger.info(f"  ✓ Exported {len(indicators)} indicators")
    logger.info(f"    Total in database: {total_count:,}")

    return {"data": indicators, "metadata": metadata}


def create_summary_stats(conn: sqlite3.Connection, predictions_data: Dict, economic_data: Dict) -> Dict[str, Any]:
    """Create summary statistics"""
    logger.info("Creating summary...")

    cursor = conn.cursor()

    # Get prediction counts by model
    cursor.execute("""
        SELECT
            CASE
                WHEN model_name LIKE '%LGBM%' THEN 'LGBM'
                WHEN model_name LIKE '%RandomForest%' THEN 'RandomForest'
                WHEN model_name LIKE '%AutoARIMA%' THEN 'AutoARIMA'
                WHEN model_name LIKE '%AutoETS%' THEN 'AutoETS'
                WHEN model_name LIKE '%SeasonalNaive%' THEN 'SeasonalNaive'
                WHEN model_name LIKE '%PatchTST%' THEN 'PatchTST'
                WHEN model_name LIKE '%TimesNet%' THEN 'TimesNet'
                ELSE 'Other'
            END as model_type,
            COUNT(*) as count
        FROM prediction_log
        GROUP BY model_type
        ORDER BY count DESC
    """)

    model_counts = {row[0]: row[1] for row in cursor.fetchall()}

    # Get prediction counts by year
    cursor.execute("""
        SELECT
            SUBSTR(prediction_date, 1, 4) as year,
            COUNT(*) as count
        FROM prediction_log
        GROUP BY year
        ORDER BY year DESC
    """)

    year_counts = {row[0]: row[1] for row in cursor.fetchall()}

    # Get SHAP coverage
    cursor.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN shap_values IS NOT NULL THEN 1 ELSE 0 END) as with_shap
        FROM prediction_log
    """)

    total, with_shap = cursor.fetchone()
    shap_coverage = round((with_shap / total * 100) if total > 0 else 0, 2)

    summary = {
        "export_timestamp": datetime.now().isoformat(),
        "database_path": DB_PATH,
        "predictions": {
            "total_count": total,
            "by_year": year_counts,
            "by_model_type": model_counts,
            "shap_coverage": {
                "with_shap": with_shap,
                "total": total,
                "percentage": shap_coverage
            }
        },
        "economic_indicators": {
            "count": economic_data.get("metadata", {}).get("row_count", 0),
            "source": economic_data.get("metadata", {}).get("source", "unknown")
        },
        "demo_data": {
            "predictions_sample": predictions_data.get("metadata", {}).get("row_count", 0),
            "indicators_sample": economic_data.get("metadata", {}).get("row_count", 0)
        },
        "models_available": {
            "with_shap": ["LGBM", "RandomForest"],
            "without_shap": ["AutoARIMA", "AutoETS", "SeasonalNaive"]
        }
    }

    logger.info("  ✓ Summary created:")
    logger.info(f"    Total predictions: {total:,}")
    logger.info(f"    SHAP coverage: {shap_coverage}% ({with_shap:,}/{total:,})")
    logger.info(f"    Years covered: {list(year_counts.keys())}")

    return summary


def main():
    """Main export function"""
    logger.info("=" * 80)
    logger.info("RetailPRED Demo Data Export")
    logger.info("=" * 80)
    logger.info("")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info("")

    try:
        # Connect to database
        logger.info(f"Connecting to database: {DB_PATH}")
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row

        # Step 1: Inspect schema
        inspect_database_schema(conn)

        # Step 2: Export predictions
        predictions_data = export_predictions(conn)
        predictions_file = OUTPUT_DIR / "predictions.json"
        with open(predictions_file, 'w') as f:
            json.dump(predictions_data, f, indent=2, default=str)
        logger.info(f"  → Saved to: {predictions_file}")
        logger.info("")

        # Step 3: Export economic indicators
        economic_data = export_economic_indicators(conn)
        economic_file = OUTPUT_DIR / "economic-indicators.json"
        with open(economic_file, 'w') as f:
            json.dump(economic_data, f, indent=2, default=str)
        logger.info(f"  → Saved to: {economic_file}")
        logger.info("")

        # Step 4: Create summary
        summary = create_summary_stats(conn, predictions_data, economic_data)
        summary_file = OUTPUT_DIR / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"  → Saved to: {summary_file}")
        logger.info("")

        conn.close()

        # Final summary
        logger.info("=" * 80)
        logger.info("Export complete! Files saved to frontend/public/demo-data/")
        logger.info("=" * 80)
        logger.info("")
        logger.info(f"1. predictions.json ({predictions_data['metadata']['row_count']} predictions)")
        logger.info(f"2. economic-indicators.json ({economic_data['metadata']['row_count']} indicators)")
        logger.info(f"3. summary.json (database statistics)")
        logger.info("")
        logger.info("Ready for static demo deployment!")

    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise


if __name__ == "__main__":
    main()
