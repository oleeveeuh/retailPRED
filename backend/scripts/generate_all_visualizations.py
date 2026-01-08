"""
Generate Performance Visualizations for All Trained Models

Creates HTML and PNG visualizations showing:
- Actual vs Predicted values
- Residuals plot
- Error distribution
- Model comparison
- Performance metrics
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sqlite3
import json
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database path
DB_PATH = "/Users/olivialiau/retailPRED/data/retailpred.db"

# Output directory
OUTPUT_DIR = Path("/Users/olivialiau/retailPRED/training_outputs/visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Category display names
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

# Model types
MODEL_TYPES = ["LGBM", "RandomForest", "AutoARIMA", "AutoETS", "SeasonalNaive"]


def get_predictions_from_db(category: str, model_type: str) -> pd.DataFrame:
    """Get predictions from database for a specific category and model"""
    conn = sqlite3.connect(DB_PATH)

    model_name = f"{category}_{model_type}_model"

    query = """
        SELECT
            prediction_date,
            predicted_value,
            actual_value,
            confidence_interval_lower,
            confidence_interval_upper,
            created_at
        FROM prediction_log
        WHERE model_name = ?
        AND actual_value IS NOT NULL
        ORDER BY prediction_date ASC
    """

    df = pd.read_sql_query(query, conn, params=(model_name,))
    conn.close()

    return df


def calculate_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate performance metrics"""
    if len(df) == 0:
        return {}

    y_true = df['actual_value'].values
    y_pred = df['predicted_value'].values

    # Calculate metrics
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # R² score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        'RMSE': round(rmse, 2),
        'MAE': round(mae, 2),
        'MAPE': round(mape, 2),
        'R²': round(r2, 4),
        'Sample_Size': len(df)
    }


def create_performance_plots(category: str, model_type: str, df: pd.DataFrame, metrics: Dict[str, float]):
    """Create performance visualization plots"""
    if len(df) == 0:
        logger.warning(f"  No data for {category} - {model_type}")
        return

    category_display = CATEGORY_DISPLAY.get(category, category.replace("_", " ").title())
    output_subdir = OUTPUT_DIR / category_display.replace(" ", "_").replace("&", "and")
    output_subdir.mkdir(parents=True, exist_ok=True)

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'{category_display} - {model_type} Model Performance',
                 fontsize=16, fontweight='bold', y=0.995)

    # 1. Actual vs Predicted scatter plot
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(df['actual_value'], df['predicted_value'],
               alpha=0.6, s=50, edgecolors='k', linewidths=0.5)
    min_val = min(df['actual_value'].min(), df['predicted_value'].min())
    max_val = max(df['actual_value'].max(), df['predicted_value'].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual Value', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Predicted Value', fontsize=11, fontweight='bold')
    ax1.set_title('Actual vs Predicted', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Time series plot
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(df.index, df['actual_value'], label='Actual',
            linewidth=2, color='#2E86AB', marker='o', markersize=4)
    ax2.plot(df.index, df['predicted_value'], label='Predicted',
            linewidth=2, color='#A23B72', marker='s', markersize=4, linestyle='--')
    ax2.set_xlabel('Time', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Value', fontsize=11, fontweight='bold')
    ax2.set_title('Time Series Comparison', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    # 3. Residuals plot
    ax3 = plt.subplot(2, 3, 3)
    residuals = df['actual_value'] - df['predicted_value']
    ax3.scatter(df['predicted_value'], residuals,
               alpha=0.6, s=50, edgecolors='k', linewidths=0.5, c=residuals, cmap='RdYlBu_r')
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax3.set_xlabel('Predicted Value', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Residuals (Actual - Predicted)', fontsize=11, fontweight='bold')
    ax3.set_title('Residual Plot', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    cb = plt.colorbar(ax3.collections[0], ax=ax3)
    cb.set_label('Residual Value', fontsize=10)

    # 4. Error distribution histogram
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='#18A558')
    ax4.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    ax4.set_xlabel('Prediction Error', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax4.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Percentage error over time
    ax5 = plt.subplot(2, 3, 5)
    pct_errors = (residuals / df['actual_value']) * 100
    ax5.plot(df.index, pct_errors, linewidth=2, color='#F18F01', marker='o', markersize=4)
    ax5.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax5.fill_between(df.index, pct_errors, 0, alpha=0.3, color='#F18F01')
    ax5.set_xlabel('Time', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Percentage Error (%)', fontsize=11, fontweight='bold')
    ax5.set_title('Percentage Error Over Time', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)

    # 6. Metrics text display
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    metrics_text = f"""
    Performance Metrics

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    RMSE:     ${metrics['RMSE']:,.2f}
    MAE:      ${metrics['MAE']:,.2f}
    MAPE:     {metrics['MAPE']:.2f}%
    R²:       {metrics['R²']:.4f}
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Samples:  {metrics['Sample_Size']:,}
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Best Performance:
    • Lowest RMSE
    • Highest R²
    • MAPE < 5%: {'✓ Excellent' if metrics['MAPE'] < 5 else '✗ Needs improvement'}
    """

    ax6.text(0.1, 0.5, metrics_text,
             transform=ax6.transAxes,
             fontsize=11,
             verticalalignment='center',
             fontfamily='monospace',
             bbox=dict(boxstyle='round',
                      facecolor='wheat',
                      alpha=0.5))

    plt.tight_layout()

    # Save PNG
    png_file = output_subdir / f"{category_display.replace(' ', '_')}_{model_type}_performance.png"
    plt.savefig(png_file, dpi=150, bbox_inches='tight')
    plt.close()

    # Save HTML (using plotly for interactivity)
    create_interactive_plot(category_display, model_type, df, metrics, output_subdir)


def create_interactive_plot(category_display: str, model_type: str, df: pd.DataFrame,
                          metrics: Dict[str, float], output_dir: Path):
    """Create interactive HTML plot using plotly"""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logger.warning("  Plotly not installed, skipping HTML export")
        return

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Actual vs Predicted', 'Time Series', 'Residuals', 'Error Distribution'),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'scatter'}, {'type': 'histogram'}]]
    )

    # 1. Actual vs Predicted
    fig.add_trace(
        go.Scatter(x=df['actual_value'], y=df['predicted_value'],
                   mode='markers',
                   name='Predictions',
                   marker=dict(size=8, color='blue', opacity=0.6)),
        row=1, col=1
    )

    # Add perfect prediction line
    min_val = min(df['actual_value'].min(), df['predicted_value'].min())
    max_val = max(df['actual_value'].max(), df['predicted_value'].max())
    fig.add_trace(
        go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                   mode='lines',
                   name='Perfect Prediction',
                   line=dict(color='red', dash='dash')),
        row=1, col=1
    )

    # 2. Time series
    fig.add_trace(
        go.Scatter(x=df['prediction_date'], y=df['actual_value'],
                   mode='lines+markers',
                   name='Actual',
                   line=dict(color='blue')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=df['prediction_date'], y=df['predicted_value'],
                   mode='lines+markers',
                   name='Predicted',
                   line=dict(color='red', dash='dash')),
        row=1, col=2
    )

    # 3. Residuals
    residuals = df['actual_value'] - df['predicted_value']
    fig.add_trace(
        go.Scatter(x=df['predicted_value'], y=residuals,
                   mode='markers',
                   name='Residuals',
                   marker=dict(size=8, color=residuals, colorscale='RdYlBu_r', opacity=0.6)),
        row=2, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)

    # 4. Error distribution
    fig.add_trace(
        go.Histogram(x=residuals,
                    name='Error Distribution',
                    marker=dict(color='lightblue')),
        row=2, col=2
    )

    # Update layout
    fig.update_layout(
        height=800,
        title_text=f'{category_display} - {model_type} Model Performance<br>' +
                  f'<sub>RMSE: ${metrics["RMSE"]:,.2f} | MAE: ${metrics["MAE"]:,.2f} | MAPE: {metrics["MAPE"]:.2f}% | R²: {metrics["R²"]:.4f}</sub>',
        showlegend=True
    )

    fig.update_xaxes(title_text="Actual Value", row=1, col=1)
    fig.update_yaxes(title_text="Predicted Value", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_yaxes(title_text="Value", row=1, col=2)
    fig.update_xaxes(title_text="Predicted Value", row=2, col=1)
    fig.update_yaxes(title_text="Residuals", row=2, col=1)
    fig.update_xaxes(title_text="Residual", row=2, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=2)

    # Save HTML
    html_file = output_dir / f"{category_display.replace(' ', '_')}_{model_type}_performance.html"
    fig.write_html(html_file)


def create_model_comparison(category: str, all_metrics: Dict[str, Dict[str, float]]):
    """Create comparison plot for all models"""
    if not all_metrics:
        return

    category_display = CATEGORY_DISPLAY.get(category, category.replace("_", " ").title())
    output_subdir = OUTPUT_DIR / category_display.replace(" ", "_").replace("&", "and")
    output_subdir.mkdir(parents=True, exist_ok=True)

    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{category_display} - Model Comparison',
                 fontsize=16, fontweight='bold')

    models = list(all_metrics.keys())
    metrics_names = ['RMSE', 'MAE', 'MAPE', 'R²']

    # Prepare data
    rmse_vals = [all_metrics[m]['RMSE'] for m in models]
    mae_vals = [all_metrics[m]['MAE'] for m in models]
    mape_vals = [all_metrics[m]['MAPE'] for m in models]
    r2_vals = [all_metrics[m]['R²'] for m in models]

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']

    # RMSE comparison
    axes[0, 0].bar(models, rmse_vals, color=colors, edgecolor='black', alpha=0.7)
    axes[0, 0].set_ylabel('RMSE ($)', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('RMSE Comparison (Lower is Better)', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')

    # MAE comparison
    axes[0, 1].bar(models, mae_vals, color=colors, edgecolor='black', alpha=0.7)
    axes[0, 1].set_ylabel('MAE ($)', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('MAE Comparison (Lower is Better)', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')

    # MAPE comparison
    axes[1, 0].bar(models, mape_vals, color=colors, edgecolor='black', alpha=0.7)
    axes[1, 0].set_ylabel('MAPE (%)', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('MAPE Comparison (Lower is Better)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].axhline(y=5, color='r', linestyle='--', linewidth=2, label='5% Threshold')
    axes[1, 0].legend()
    plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')

    # R² comparison
    axes[1, 1].bar(models, r2_vals, color=colors, edgecolor='black', alpha=0.7)
    axes[1, 1].set_ylabel('R² Score', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('R² Comparison (Higher is Better)', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].set_ylim([0, 1])
    plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    # Save
    png_file = output_subdir / f"{category_display.replace(' ', '_')}_all_models_comparison.png"
    plt.savefig(png_file, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"  Saved comparison: {png_file.name}")


def main():
    logger.info("=" * 80)
    logger.info("Generating Performance Visualizations for All Models")
    logger.info("=" * 80)
    logger.info("")

    categories = list(CATEGORY_DISPLAY.keys())

    total_generated = 0
    total_skipped = 0

    for category in categories:
        category_display = CATEGORY_DISPLAY.get(category, category)
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Category: {category_display}")
        logger.info(f"{'=' * 80}")

        all_metrics = {}

        for model_type in MODEL_TYPES:
            try:
                # Get predictions
                df = get_predictions_from_db(category, model_type)

                if len(df) == 0:
                    logger.info(f"  ✗ {model_type}: No data")
                    total_skipped += 1
                    continue

                # Calculate metrics
                metrics = calculate_metrics(df)
                all_metrics[model_type] = metrics

                # Create visualizations
                create_performance_plots(category, model_type, df, metrics)

                logger.info(f"  ✓ {model_type}: RMSE=${metrics['RMSE']}, MAPE={metrics['MAPE']}%, R²={metrics['R²']}")
                total_generated += 1

            except Exception as e:
                logger.error(f"  ✗ {model_type}: {str(e)[:100]}")
                total_skipped += 1

        # Create comparison plot
        if all_metrics:
            create_model_comparison(category, all_metrics)

    logger.info("\n" + "=" * 80)
    logger.info(f"✓ Generated {total_generated} model visualizations")
    logger.info(f"✗ Skipped {total_skipped} (no data)")
    logger.info(f"📁 Output: {OUTPUT_DIR}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
