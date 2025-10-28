# Retail Market Dynamics - Project Setup

This repository contains a setup script for initializing the "retail_market_dynamics" project in Google Colab.

## Quick Start

### Option 1: Run in Google Colab (Recommended)

1. Open Google Colab: https://colab.research.google.com/
2. Upload or clone this repository
3. Run the setup script in a new Colab cell:

```python
# Copy and paste the entire setup_retail_market_dynamics.py script here
# Or use exec() to run it from a file
```

### Option 2: Copy-paste the Setup Code

You can also copy the entire contents of `setup_retail_market_dynamics.py` and run it directly in a Colab notebook cell.

## What the Script Does

1. **Creates Folder Structure**
   - `/content/retail_market_dynamics/data/raw/` - For raw data files
   - `/content/retail_market_dynamics/data/processed/` - For cleaned/processed data
   - `/content/retail_market_dynamics/scripts/` - For utility scripts
   - `/content/retail_market_dynamics/models/` - For trained models
   - `/content/retail_market_dynamics/visuals/` - For visualizations
   - `/content/retail_market_dynamics/notebooks/` - For Jupyter notebooks

2. **Mounts Google Drive**
   - Automatically mounts your Google Drive
   - Creates project directory at `/content/drive/MyDrive/retail_market_dynamics`
   - Sets `DRIVE_PATH` variable for easy access

3. **Installs Required Libraries**
   - pandas - Data manipulation
   - numpy - Numerical computing
   - requests - HTTP library
   - matplotlib, seaborn, plotly - Visualization
   - yfinance - Financial data
   - fredapi - Economic data from FRED
   - scikit-learn - Machine learning
   - statsmodels - Statistical modeling
   - prophet - Time series forecasting

4. **Verifies Installation**
   - Imports all libraries
   - Prints version numbers
   - Reports any installation failures

## Project Structure

```
/content/retail_market_dynamics/
├── data/
│   ├── raw/           # Raw data files
│   └── processed/      # Cleaned and processed data
├── scripts/            # Utility and data processing scripts
├── models/             # Trained ML models
├── visuals/            # Charts, graphs, and visualizations
└── notebooks/          # Jupyter notebooks for analysis

/content/drive/MyDrive/retail_market_dynamics/
└── (Mirrored structure for persistent storage)
```

## Complete Workflow

Follow these steps to set up and run the complete pipeline:

```python
# Step 1: Setup (run once)
exec(open('setup_retail_market_dynamics.py').read())

# Step 2: Download raw data
exec(open('data_ingestion.py').read())

# Step 3: Clean and merge data
exec(open('data_cleaning.py').read())

# Step 4: Create engineered features
exec(open('feature_engineering.py').read())

# Step 5: Train ML models
exec(open('modeling.py').read())

# Step 6: Ready for analysis!
import pandas as pd
df_results = pd.read_csv('/content/retail_market_dynamics/models/results_summary.csv')
```

## Usage After Setup

Once the setup is complete:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from fredapi import Fred

# Your project path
PROJECT_PATH = "/content/retail_market_dynamics"
DRIVE_PATH = "/content/drive/MyDrive/retail_market_dynamics"

# Load processed features
df_features = pd.read_csv('data/processed/features.csv')

# Start your analysis!
```

## Data Ingestion Script

After setup, use `data_ingestion.py` to download raw data:

```python
# In a Colab notebook or Python script
exec(open('data_ingestion.py').read())

# Or run the main function
from data_ingestion import main
main()
```

**What it does:**
- Downloads MRTS (Monthly Retail Trade Survey) data from U.S. Census Bureau
- Downloads economic indicators (CPI, Consumer Confidence) from FRED
- Downloads stock data for SPY, XRT, AMZN, WMT using yfinance
- Automatically handles errors and creates sample data when APIs are unavailable

**Requirements for full functionality:**
- FRED API key (get from https://fred.stlouisfed.org/docs/api/api_key.html)
- Internet connection for data downloads

## Data Cleaning Script

After downloading raw data, use `data_cleaning.py` to process and unify all datasets:

```python
# In a Colab notebook or Python script
exec(open('data_cleaning.py').read())

# Or run the main function
from data_cleaning import main
main()
```

**What it does:**
- Cleans MRTS data: aggregates sales by month, computes total sales and growth
- Cleans stock data: converts daily to monthly, calculates returns for SPY, XRT, AMZN, WMT
- Cleans FRED data: aligns to month-end, computes CPI monthly changes
- Merges all datasets into unified time series with columns:
  - `date` - Month-end date
  - `retail_sales` - Total retail sales (millions)
  - `retail_growth` - Month-over-month growth (%)
  - `sp500_close` - S&P 500 closing price
  - `sp500_return` - S&P 500 monthly return (%)
  - `cpi` - Consumer Price Index
  - `consumer_confidence` - Consumer Confidence Index
- Saves final dataset to `data/processed/combined.csv`

**Output files:**
- `data/processed/retail_cleaned.csv` - Cleaned retail data
- `data/processed/stocks_cleaned.csv` - Cleaned stock data
- `data/processed/fred_cleaned.csv` - Cleaned economic data
- `data/processed/combined.csv` - Final unified dataset

## Feature Engineering Script

After cleaning the data, use `feature_engineering.py` to create derived features:

```python
# In a Colab notebook or Python script
exec(open('feature_engineering.py').read())

# Or run the main function
from feature_engineering import main
main()
```

**What it does:**
- **Lagged Features**: Creates 1-month and 3-month lags for `retail_growth` and `cpi`
- **Rolling Averages**: Creates 3-month and 6-month rolling averages for `retail_growth` and `sp500_return`
- **Normalized Features**: Applies MinMax or Z-score normalization to numerical features
- **Correlation Analysis**: Creates heatmap showing relationships between features
- **Output**: Saves engineered features to `data/processed/features.csv`

**Output files:**
- `data/processed/features.csv` - Engineered features with lags, rolling stats, and normalized values
- `visuals/correlation_heatmap.png` - Correlation heatmap visualization

**Key features added:**
- `retail_growth_lag1`, `retail_growth_lag3` - Lagged retail growth
- `cpi_lag1`, `cpi_lag3` - Lagged CPI values
- `retail_growth_rolling3m`, `retail_growth_rolling6m` - Rolling averages
- `sp500_return_rolling3m`, `sp500_return_rolling6m` - Rolling averages
- `*_norm` - Normalized versions of all numerical features

## Modeling Script

After feature engineering, use `modeling.py` to train and evaluate ML models:

```python
# In a Colab notebook or Python script
exec(open('modeling.py').read())

# Or run the main function
from modeling import main
main()
```

**What it does:**
- **Statistical Analysis**: Runs correlation tests and Granger causality tests to assess predictive relationships
- **Linear Regression**: Trains a linear model to predict retail dynamics
- **Random Forest**: Trains an ensemble model with feature importance analysis
- **Prophet**: Time series forecasting using Facebook Prophet
- **Model Evaluation**: Calculates R², MAE, and RMSE for each model
- **Saves Results**: Models and evaluation metrics to `/models/`

**Output files:**
- `models/results_summary.csv` - Model evaluation metrics (R², MAE, RMSE)
- `models/linear_regression.joblib` - Trained linear regression model
- `models/random_forest.joblib` - Trained random forest model
- `models/prophet.joblib` - Trained Prophet model
- `visuals/linear_regression_actual_vs_predicted.png` - Actual vs predicted plot
- `visuals/random_forest_actual_vs_predicted.png` - Actual vs predicted plot
- `visuals/random_forest_feature_importances.png` - Feature importance plot
- `visuals/prophet_forecast.html` - Interactive forecast visualization

**Model evaluation metrics:**
- **R² Score**: Coefficient of determination (higher is better)
- **MAE**: Mean Absolute Error (lower is better)
- **RMSE**: Root Mean Squared Error (lower is better)

## Streamlit Dashboard

After completing the analysis pipeline, launch an interactive dashboard:

```bash
streamlit run dashboard.py
```

**Dashboard features:**
- **Time Series Charts**: Interactive line charts showing retail sales and S&P 500 over time
- **Scatter Plots**: Compare retail growth vs market returns with correlation analysis
- **Correlation Heatmap**: Visualize relationships between all features
- **Model Forecasts**: View Prophet forecasts with confidence intervals
- **Interactive Controls**: Date range selector and ticker dropdown (SPY, AMZN, WMT, XRT)
- **Key Metrics Display**: Real-time statistics for retail and market indicators
- **Data Export**: Download processed data as CSV

**To run:**
```bash
# Install Streamlit if not already installed
pip install streamlit

# Run the dashboard
streamlit run dashboard.py
```

The dashboard automatically loads data from:
- `data/processed/features.csv` - Engineered features
- `models/` - Trained models
- `models/results_summary.csv` - Model evaluation metrics

## Monthly Update Automation

Automate monthly data refresh with `monthly_update.py`:

```bash
# Run monthly update
python monthly_update.py
```

**What it does:**
- **Smart Update Detection**: Only runs if data is older than threshold (default: 1 month)
- **Automated Pipeline**: Runs data_ingestion.py → data_cleaning.py → feature_engineering.py
- **Data Merging**: Appends new data to existing features.csv without duplicates
- **Model Retraining**: Automatically retrains models if new data is available
- **Notifications**: Sends success/failure notifications (Slack/Email placeholders)
- **Logging**: Comprehensive logging to `monthly_update.log`

**To automate with cron (Linux/Mac):**
```bash
# Add to crontab to run on 1st of each month at 2 AM
0 2 1 * * cd /path/to/retailPRED && python monthly_update.py >> monthly_update.log 2>&1
```

**To automate with Task Scheduler (Windows):**
1. Open Task Scheduler
2. Create Basic Task → "Monthly Update"
3. Trigger: Monthly → Day 1 → Time 2:00 AM
4. Action: Start a program → `python.exe monthly_update.py`
5. Settings: Run whether user is logged on or not

**Configuration:**
- Edit `monthly_update.py` to adjust update threshold
- Add Slack webhook URL for notifications
- Add email settings for email notifications

## Backup Utility

Safely backup the entire project to Google Drive with `save_to_drive.py`:

```bash
# Create a backup
python save_to_drive.py

# List existing backups
python save_to_drive.py --list

# Custom source and output directories
python save_to_drive.py --source /custom/path --output /backup/location
```

**What it does:**
- **Project Backup**: Zips the entire retail_market_dynamics folder
- **Timestamped Files**: Creates backups with names like `retail_market_dynamics_backup_20240101_143022.zip`
- **Smart Exclusions**: Skips unnecessary files (`.pyc`, `__pycache__`, `.git`, etc.)
- **Progress Tracking**: Shows file-by-file progress during backup
- **Size Information**: Displays source size, backup size, and compression ratio
- **Google Drive Integration**: Automatically saves to `/content/drive/MyDrive/retail_market_dynamics/model_backups/`

**Features:**
- Automatic Drive mounting if not already mounted
- Directory size calculation and reporting
- Compression efficiency reporting
- Can run in both Colab and local environments
- Excludes temporary and cache files

**Integration with monthly updates:**
You can integrate this into your monthly update automation:

```python
# In monthly_update.py, add at the end:
import save_to_drive
save_to_drive.backup_to_drive()
```

## Notes

- The script runs quietly and only shows important messages
- If library installation fails, the script will continue and report errors
- All directories are created with `exist_ok=True`, so it's safe to run multiple times
- Google Drive mounting may require authentication on first run
- The data ingestion script will work with sample data if APIs are unavailable

## Troubleshooting

**Issue**: Libraries fail to install
- **Solution**: The script will continue and report which libraries failed. You can manually install them using `!pip install library_name`

**Issue**: Cannot mount Google Drive
- **Solution**: Make sure you're running in Google Colab. The script will continue without Drive mount.

**Issue**: Permission errors
- **Solution**: Run `!chmod -R 777 /content/retail_market_dynamics` to fix permissions

## Contact

For questions or issues, please refer to the project documentation or contact the project maintainer.

