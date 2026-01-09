# RetailPRED: Macroeconomic Retail Sales Forecasting System

An end-to-end machine learning system for forecasting retail sales across multiple categories using advanced time series models, macroeconomic indicators, and SHAP-based explainability.

[![Live Demo](https://img.shields.io/badge/demo-live_online-brightgreen)](https://retailpred.vercel.app)
[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://python.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue)](https://typescriptlang.org)
[![React](https://img.shields.io/badge/React-19+-blue)](https://react.dev)

**Live Demo:** https://retailpred.vercel.app

---

## Project Overview

This project was developed to explore the application of modern machine learning techniques to retail sales forecasting. It combines data from multiple sources (Federal Reserve Economic Data, U.S. Census Bureau, and Yahoo Finance) to generate accurate forecasts across 11 retail categories using seven different forecasting algorithms.

## System Overview

RetailPRED is an end-to-end retail forecasting platform that combines multi-resolution time series modeling with interactive visualizations and model explainability. The system processes data from Federal Reserve Economic Data (FRED), Monthly Retail Trade Survey (MRTS), and Yahoo Finance to generate accurate forecasts across 11 retail categories.

### Key Capabilities

- **Multi-Model Architecture**: Seven forecasting algorithms (LightGBM, Random Forest, AutoARIMA, AutoETS, Seasonal Naive, PatchTST, TimesNet) with automatic model selection
- **Feature Engineering**: 242 engineered features including lag features, rolling statistics, rate-of-change indicators, and economic variables
- **Model Explainability**: SHAP (SHapley Additive exPlanations) values for tree-based models to interpret feature contributions
- **Economic Scenario Modeling**: What-if analysis with macroeconomic indicators (unemployment, CPI, interest rates, GDP)
- **Historical Validation**: Track prediction accuracy over time with comprehensive metrics
- **Interactive Dashboard**: Real-time visualization of forecasts, confidence intervals, and model performance

### Performance Metrics

**Best Performing Models** (across 4 categories):

| Model | Avg MAPE | Avg MASE | Best For |
|-------|----------|----------|----------|
| **LightGBM** | 1.42% | 0.207 | Most categories (3/4) |
| **Random Forest** | 2.08% | 0.285 | Complex interactions |
| **AutoETS** | 9.60% | 0.991 | Exponential smoothing |
| **AutoARIMA** | 12.69% | 1.303 | Autoregressive patterns |

**Category Champions** (lowest MAPE):
- Building Materials & Garden: **0.16%** (LightGBM)
- Furniture & Home Furnishings: **0.30%** (LightGBM)
- General Merchandise: **2.09%** (LightGBM)
- Sporting Goods & Hobby: **3.13%** (LightGBM)

---

## Table of Contents

1. [Technical Architecture](#technical-architecture)
2. [Data Pipeline](#data-pipeline)
3. [Feature Engineering](#feature-engineering)
4. [Model Training](#model-training)
5. [Model Details](#model-details)
6. [Inference Pipeline](#inference-pipeline)
7. [Project Structure](#project-structure)
8. [Quick Start](#quick-start)
9. [API Reference](#api-reference)
10. [Deployment](#deployment)

---

## Technical Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Ingestion Layer                      │
│  FRED API │ MRTS Census │ Yahoo Finance (stock data)        │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                  ETL Processing Layer                        │
│  Data normalization │ Multi-resolution resampling           │
│  Feature engineering │ Quality validation                   │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                 Model Training Layer                         │
│  7 algorithms │ Cross-validation │ Model selection          │
│  SHAP computation │ Performance evaluation                  │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                   Inference Layer                            │
│  Real-time forecasting │ Confidence intervals               │
│  Feature importance │ Scenario analysis                     │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                   Presentation Layer                         │
│  React frontend │ Interactive visualizations                │
│  Model explanations │ Validation tracking                   │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

**Backend (Python):**
- FastAPI: High-performance async API framework
- LightGBM & scikit-learn: Machine learning models
- Statsmodels: Statistical time series models (ARIMA, ETS)
- SHAP: Model explainability
- SQLite: Prediction logging and validation
- Pandas/NumPy: Data processing

**Frontend (TypeScript):**
- React 19: UI framework
- TanStack Query: Data fetching and caching
- Recharts: Data visualization
- TailwindCSS: Styling
- Framer Motion: Animations
- Vite: Build tool

**Deployment:**
- Vercel: Frontend hosting (demo mode)
- Docker: Full-stack containerization

---

## Data Pipeline

### 1. Data Sources

#### Monthly Retail Trade Survey (MRTS)
- **Provider:** U.S. Census Bureau
- **Frequency:** Monthly
- **Coverage:** 11 retail categories
- **Data Lag:** 1-2 months

**Categories Tracked:**
1. Total Retail Sales (4400A)
2. Automobile Dealers (441)
3. Building Materials & Garden (444)
4. Clothing & Accessories (452)
5. Electronics & Appliances (44X72)
6. Food & Beverage Stores (445)
7. Furniture & Home Furnishings (442)
8. Gasoline Stations (448)
9. General Merchandise (454)
10. Health & Personal Care (447)
11. Sporting Goods & Hobby (453)

#### Federal Reserve Economic Data (FRED)
- **Provider:** Federal Reserve Bank of St. Louis
- **Frequency:** Monthly
- **Indicators:** 9 macroeconomic variables

**Key Indicators:**
- **CPI:** Consumer Price Index (inflation measure)
- **FEDFUNDS:** Federal Funds Rate (interest rate)
- **UNRATE:** Unemployment Rate
- **UMCSENT:** Consumer Sentiment Index
- **INDPRO:** Industrial Production Index
- **PCE:** Personal Consumption Expenditures
- **M2SL:** Money Supply M2
- **PAYEMS:** Nonfarm Payrolls
- **GDP:** Gross Domestic Product

#### Yahoo Finance Stock Data
- **Provider:** Yahoo Finance
- **Frequency:** Daily (aggregated to monthly)
- **Stocks Tracked:** AAPL, WMT, AMZN, COST

**Metrics per Stock:**
- Monthly return
- Monthly volatility
- Average trading volume

### 2. Data Processing Pipeline

**Location:** `project_root/etl/`

#### Stage 1: Data Collection
```bash
# Fetch MRTS retail sales data
python project_root/etl/fetch_mrts.py

# Fetch FRED economic indicators
python project_root/etl/fetch_fred.py

# Fetch Yahoo Finance stock data
python project_root/etl/fetch_yahoo.py
```

**Output:** Raw CSV files in `project_root/data_raw/`

#### Stage 2: Data Normalization
**Script:** `build_dataset.py`

**Process:**
1. Date alignment to month-end
2. Daily to monthly aggregation (Yahoo Finance data)
3. Outer join on date column
4. Forward fill missing values
5. Combine FRED, MRTS, and Yahoo Finance data

**Output:** `project_root/data_processed/combined_dataset.csv`

#### Stage 3: Multi-Resolution Resampling
**Script:** `build_multi_resolution_dataset.py`

**Purpose:** Create data at multiple temporal granularities for feature engineering

**Method:**

**Daily Data Creation (Monthly to Daily Interpolation):**
```python
# Linear interpolation for smooth daily curves
df_daily = df.reindex(daily_date_range)
df_daily = df_daily.interpolate(method='linear')

# Retail-specific day-of-week adjustment factors
dow_factors = {
    0: 0.90,  # Monday
    1: 0.95,  # Tuesday
    2: 0.95,  # Wednesday
    3: 1.00,  # Thursday
    4: 1.05,  # Friday
    5: 1.25,  # Saturday (highest retail activity)
    6: 1.20,  # Sunday
}
```

**Weekly Aggregation:**
- Resample to week-start (Monday)
- Aggregate using mean for continuous variables
- Use last observation for categorical

**Monthly Aggregation:**
- Maintain original monthly data
- Preserves true monthly patterns

**Yearly Aggregation:**
- Resample to year-start
- Capture long-term trends

**Output:** 4 datasets per category (daily, weekly, monthly, yearly)

---

## Feature Engineering

### Feature Architecture

**Total Features:** 242 features per observation
**Feature Categories:** 7 major types

### 1. Temporal Features (16 features)

Capture seasonal patterns and calendar effects through both linear and cyclical encodings.

**Linear Temporal Features:**
- `year`: Calendar year (2010-2025)
- `month`: Month of year (1-12)
- `quarter`: Quarter of year (1-4)
- `day_of_week`: Day of week (0=Monday, 6=Sunday)
- `week_of_year`: ISO week number (1-53)
- `is_weekend`: Binary flag (1 if Sat/Sun, 0 otherwise)
- `day_of_month`: Day of month (1-31)
- `day_of_year`: Day of year (1-366)

**Cyclical Temporal Features** (preserve continuity):
- `month_sin`, `month_cos`: Cyclical month encoding
- `quarter_sin`, `quarter_cos`: Cyclical quarter encoding
- `day_of_year_sin`, `day_of_year_cos`: Cyclical day encoding
- `day_of_week_sin`, `day_of_week_cos`: Cyclical weekday encoding

**Why Cyclical Encoding:** Preserves the cyclical nature of time where December (12) is close to January (1), which linear encoding would disrupt.

### 2. Lag Features (10 features)

Capture autoregressive patterns where past values predict future values.

**Adaptive Lag Selection:** Lags chosen based on available data history (maximum 40% of data length)

| Feature | Period | Use Case |
|---------|--------|----------|
| `lag_1d` | 1 day | Short-term memory |
| `lag_7d` | 7 days (1 week) | Weekly pattern |
| `lag_14d` | 14 days (2 weeks) | Bi-weekly pattern |
| `lag_30d` | 30 days (1 month) | Monthly pattern |
| `lag_4w` | 4 weeks | Monthly comparison (weekly granularity) |
| `lag_8w` | 8 weeks | 2-month comparison |
| `lag_12w` | 12 weeks | Quarterly comparison |
| `lag_3m` | 3 months | Quarterly pattern |
| `lag_6m` | 6 months | Semi-annual pattern |
| `lag_12m` | 12 months | Year-over-year pattern |

**Feature Importance:** Lag features consistently rank in the top 5 most important features across all categories.

### 3. Rolling Statistics (24 features)

Capture moving averages, volatility, and trend strength at multiple time scales.

**Monthly Rolling Windows (6 features):**
- `rolling_mean_3`, `rolling_std_3`: 3-period mean/std
- `rolling_mean_6`, `rolling_std_6`: 6-period mean/std
- `rolling_mean_12`, `rolling_std_12`: 12-period mean/std

**Daily Rolling Windows (6 features):**
- `rolling_mean_7d`, `rolling_std_7d`: 7-day mean/std
- `rolling_mean_14d`, `rolling_std_14d`: 14-day mean/std
- `rolling_mean_30d`, `rolling_std_30d`: 30-day mean/std

**Weekly Rolling Windows (6 features):**
- `rolling_mean_4w`, `rolling_std_4w`: 4-week mean/std
- `rolling_mean_8w`, `rolling_std_8w`: 8-week mean/std
- `rolling_mean_12w`, `rolling_std_12w`: 12-week mean/std

**Monthly Extended Rolling Windows (6 features):**
- `rolling_mean_3m`, `rolling_std_3m`: 3-month mean/std
- `rolling_mean_6m`, `rolling_std_6m`: 6-month mean/std
- `rolling_mean_12m`, `rolling_std_12m`: 12-month mean/std

**Interpretation:**
- **Rolling Means:** Capture trend direction (increasing = uptrend, decreasing = downtrend)
- **Rolling Std:** Capture volatility regime (high = unstable, low = stable)

### 4. Rate of Change Features (10 features)

Capture momentum, acceleration, and growth rates.

| Feature | Formula | Interpretation |
|---------|---------|----------------|
| `diff_1` | y[t] - y[t-1] | Day-over-day change |
| `diff_12` | y[t] - y[t-12] | 12-period change |
| `pct_change_1` | (y[t] - y[t-1]) / y[t-1] × 100 | Daily return % |
| `pct_change_12` | (y[t] - y[t-12]) / y[t-12] × 100 | 12-period return % |
| `pct_change_1w` | (y[t] - y[t-7]) / y[t-7] × 100 | Weekly growth % |
| `diff_1w` | y[t] - y[t-7] | Weekly change |
| `pct_change_1m` | (y[t] - y[t-30]) / y[t-30] × 100 | Monthly growth % |
| `diff_1m` | y[t] - y[t-30] | Monthly change |
| `pct_change_1y` | (y[t] - y[t-365]) / y[t-365] × 100 | Annual growth % |
| `diff_1y` | y[t] - y[t-365] | Annual change |

### 5. Momentum Indicators (2 features)

Capture sustained directional movement.

- `momentum_30d`: 30-day momentum (y[t] - y[t-30])
- `momentum_90d`: 90-day momentum (y[t] - y[t-90])

**Interpretation:**
- Positive momentum = uptrend
- Negative momentum = downtrend
- Large magnitude = strong trend

### 6. Year-over-Year Feature (1 feature)

- `yoy_change`: Normalized annual growth rate (pct_change_1y / 100)

**Purpose:** Compare current performance to same period last year, controlling for seasonality.

### 7. Economic Indicators (9 features)

Macroeconomic variables that influence retail spending:

- `CPI`: Consumer Price Index (inflation)
- `FEDFUNDS`: Federal Funds Rate (interest rates)
- `UNRATE`: Unemployment Rate
- `UMCSENT`: Consumer Sentiment
- `INDPRO`: Industrial Production
- `PCE`: Personal Consumption Expenditures
- `M2SL`: Money Supply
- `PAYEMS`: Nonfarm Payrolls
- `GDP`: Gross Domestic Product

### Feature Importance Analysis

**Top 10 Features** (averaged across all categories):

| Rank | Feature | Avg Importance | Category |
|------|---------|----------------|----------|
| 1 | `lag_1d` | 24% | Lag |
| 2 | `rolling_mean_7d` | 12% | Rolling Statistics |
| 3 | `pct_change_1w` | 9% | Rate of Change |
| 4 | `lag_7d` | 8% | Lag |
| 5 | `month` | 7% | Temporal |
| 6 | `rolling_std_7d` | 6% | Rolling Statistics |
| 7 | `diff_1w` | 5% | Rate of Change |
| 8 | `quarter_sin` | 4% | Temporal |
| 9 | `momentum_30d` | 4% | Momentum |
| 10 | `UNRATE` | 3% | Economic |

**Key Insights:**
- Autoregressive features (lags) dominate with 32% combined importance
- Rolling statistics capture 18% (trend + volatility)
- Rate of change features capture 14% (momentum)
- Temporal features capture 11% (seasonality)

---

## Model Training

### Training Configuration

**Location:** `project_root/models/robust_timecopilot_trainer.py`

### Data Split

**Temporal Train/Test Split** (Critical for time series):

- **Training Set:** January 2010 - December 2024 (15 years)
  - 5,652 daily observations
  - Used for model training
  - Ensures models learn historical patterns

- **Test Set (Holdout):** January 2025 - December 2025 (1 year)
  - 162 daily observations
  - Strict temporal holdout (no data leakage)
  - Used for validation and performance evaluation
  - Mimics real-world forecasting scenario

**Why Temporal Split:** Random split would cause data leakage where future information contaminates training. Temporal split ensures models are evaluated on truly unseen future data.

### Cross-Validation

**Method:** Time Series Split (`TimeSeriesSplit` from sklearn)

**Configuration:**
- 5 folds
- Gap of 7 days between train and validation (prevent leakage)
- Expanding window (not rolling) to maximize training data

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(
    n_splits=5,
    gap=7,  # 7-day gap to prevent leakage
    test_size=162  # 1 year holdout
)
```

### Model Selection Strategy

**Per-Category Best Model Selection:**

After training all models on the training set, performance is evaluated on the 2025 holdout set. The model with lower MAPE is selected as the "best model" for that category.

**Selection Process:**
1. Train all 7 models on 2010-2024 data
2. Evaluate all on 2025 holdout data
3. Select model with lowest validation MAPE
4. Deploy selected model for production forecasting

---

## Model Details

### 1. LightGBM (Gradient Boosting)

**Algorithm:** Gradient boosting framework that uses tree-based learning algorithms

**Hyperparameters:**
```python
{
    'objective': 'regression',
    'metric': 'mape',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'verbosity': -1,
    'random_state': 42
}
```

**Training Configuration:**
- Early stopping with 100 rounds patience
- 1,000 maximum boosting rounds
- Validation set used for early stopping

**Performance:** Best model on 3/4 categories
- Average MAPE: 1.42%
- Average MASE: 0.207
- Best for: Smooth trends, consistent patterns, non-linear relationships

**SHAP Support:** YES - Tree-based model supports SHAP value computation for explainability

**Why It Works Well:**
- Handles non-linear relationships
- Robust to outliers
- Built-in regularization prevents overfitting
- Fast training speed
- Excellent at capturing complex feature interactions

### 2. Random Forest

**Algorithm:** Ensemble learning method operating by constructing a multitude of decision trees

**Hyperparameters:**
```python
{
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 10,
    'min_samples_leaf': 4,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1
}
```

**Performance:** Second-best on most categories
- Average MAPE: 2.08%
- Average MASE: 0.285
- Best for: Volatile patterns, complex interactions, non-linear patterns

**SHAP Support:** YES - Tree-based model supports SHAP value computation

**Why It Works Well:**
- Handles high-dimensional spaces well
- Robust to overfitting (with proper tuning)
- Captures complex feature interactions
- Parallelizable training
- Less sensitive to hyperparameters

### 3. AutoARIMA

**Algorithm:** AutoRegressive Integrated Moving Average with automatic parameter selection

**How It Works:**
- Automatically selects ARIMA parameters (p, d, q)
- Uses AIC (Akaike Information Criterion) for model selection
- Handles trend and seasonality through differencing

**Performance:**
- Average MAPE: 12.69%
- Average MASE: 1.303
- Best for: Autoregressive patterns, clear trend/seasonality

**SHAP Support:** NO - Statistical model without feature-based structure

**Why Lower Performance:**
- Linear model assumes linear relationships
- Cannot capture complex feature interactions
- Struggles with non-linear patterns
- Limited to univariate relationships with external regressors

**Use Case:** Good baseline model, interpretable parameters, fast training

### 4. AutoETS

**Algorithm:** Exponential Smoothing with automatic error/trend/seasonality selection

**How It Works:**
- Automatically selects ETS model type
- Handles additive and multiplicative seasonality
- Smooths data using exponential weights

**Performance:**
- Average MAPE: 9.60%
- Average MASE: 0.991
- Best for: Exponential smoothing trends, seasonal patterns

**SHAP Support:** NO - Statistical model without feature-based structure

**Why Better Than ARIMA:**
- More robust to outliers
- Handles multiple seasonality types
- Better for seasonal data
- Smoother forecasts

**Limitation:** Still linear, cannot capture complex interactions

### 5. Seasonal Naive

**Algorithm:** Naive forecasting method using seasonal lags

**How It Works:**
- Forecast = value from same period last season
- Simple baseline for comparison

**Performance:**
- Average MAPE: 12.72%
- Average MASE: 1.372
- Best for: Strong seasonal patterns, simple baseline

**SHAP Support:** NO - No feature-based structure

**Use Case:** Baseline model for comparison, minimal assumptions

### 6. PatchTST

**Algorithm:** Patch Time Series Transformer (deep learning model)

**Architecture:**
- Transformer-based model for time series
- Patches time series into segments
- Uses self-attention mechanisms

**Performance:**
- Average MAPE: 22.21%
- Average MASE: 2.383
- Best for: Complex temporal patterns (theoretically)

**SHAP Support:** NO - Deep learning model without straightforward SHAP support

**Why Poor Performance:**
- Requires large amounts of training data
- Overfitting on small dataset (5,652 samples)
- Complex architecture for relatively simple patterns
- Long training time for marginal gains

**Use Case:** Research purposes, larger datasets

### 7. TimesNet

**Algorithm:** Deep learning model using temporal 2D convolution

**Architecture:**
- Converts 1D time series to 2D tensors
- Uses CNN for pattern extraction
- Captures multi-scale temporal patterns

**Performance:**
- Average MAPE: 22.47%
- Average MASE: 2.416
- Best for: Complex temporal patterns (theoretically)

**SHAP Support:** NO - Deep learning model without straightforward SHAP support

**Why Poor Performance:**
- Same issues as PatchTST
- Overfitting on small dataset
- Long training time (144+ seconds per category)
- Complex architecture for relatively simple patterns

**Use Case:** Research purposes, larger datasets

### SHAP Explainability

**What is SHAP:** SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output of any machine learning model.

**How It Works:**
- Computes contribution of each feature to prediction
- Based on Shapley values from game theory
- Provides local and global interpretability

**Available For:** LightGBM and Random Forest only

**Why Only These Models:**
- Both are tree-based models
- SHAP has optimized TreeExplainer for tree models
- Fast computation (O(TLD) where T=trees, L=leaves, D=depth)
- Statistical models (ARIMA, ETS) don't have feature-based structure
- Deep learning models (PatchTST, TimesNet) require expensive approximations

**Example Output:**
```json
{
  "prediction": 72345.67,
  "baseline": 70000.00,
  "shap_values": [
    {"feature": "lag_1d", "value": 1500.00, "importance": 0.35},
    {"feature": "rolling_mean_7d", "value": 800.00, "importance": 0.18},
    {"feature": "month", "value": 450.00, "importance": 0.10},
    {"feature": "UNRATE", "value": -200.00, "importance": -0.05}
  ]
}
```

**Interpretation:**
- `lag_1d` increases prediction by $1,500 (35% of deviation from baseline)
- `UNRATE` decreases prediction by $200 (negative impact on sales)

---

## Inference Pipeline

### Real-Time Prediction Generation

**Location:** `backend/ml/inference.py`

**API Endpoint:** `GET /api/predict`

### Request Parameters

```python
{
    'category': 'total_sales',
    'model_name': 'lightgbm',  # optional, defaults to best model
    'weeks_ahead': 4,
    'granularity': 'weekly'
}
```

### Prediction Process

#### Step 1: Model Loading

```python
import joblib

# Load trained model
model_path = f"../training_outputs/models/Total_Retail_Sales/LGBM_model.pkl"
model = joblib.load(model_path)
```

#### Step 2: Historical Data Loading

```python
# Load historical data for feature computation
historical_df = load_historical_data(
    category="Total_Retail_Sales",
    days_back=400  # Enough data for lags
)
```

#### Step 3: Feature Computation

```python
# Get most recent date
last_date = historical_df['date'].max()

# Compute features for prediction date
features_df = compute_features(
    historical_data=historical_df,
    prediction_date=last_date + timedelta(days=7)
)
```

#### Step 4: Multi-Step Forecast

For `weeks_ahead > 1`, use recursive forecasting:

```python
forecasts = []
current_df = historical_df.copy()

for week in range(weeks_ahead):
    # Compute features for next date
    next_date = current_df['date'].max() + timedelta(days=7)
    features = compute_features(current_df, next_date)

    # Predict
    pred = model.predict(features)[0]
    forecasts.append({
        'date': next_date,
        'predicted_value': pred
    })

    # Append prediction to history for next iteration
    new_row = {'date': next_date, 'y': pred}
    current_df = pd.concat([current_df, pd.DataFrame([new_row])])
```

#### Step 5: Confidence Intervals

```python
# Calculate confidence intervals (±15% default)
confidence_lower = prediction * 0.85
confidence_upper = prediction * 1.15
confidence_score = 0.85  # Model confidence score
```

#### Step 6: SHAP Value Computation (Tree-based models only)

```python
if model_type in ['LGBM', 'RandomForest']:
    import shap

    # Compute SHAP values for explainability
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features_df)

    # Get top N features
    top_features = get_top_features(shap_values, feature_names, n=10)
else:
    top_features = None  # Not available for statistical models
```

### Response Format

```json
{
    "prediction_id": 1234,
    "model_name": "Total_Retail_Sales_LGBM_model",
    "model_type": "LGBM",
    "forecasts": [
        {
            "date": "2026-01-10",
            "predicted_value": 72345.67,
            "confidence_lower": 68928.39,
            "confidence_upper": 75762.95
        }
    ],
    "shap_values": [
        {
            "feature": "lag_1d",
            "value": 71_850.00,
            "contribution": 1500.00,
            "importance": 0.35
        }
    ],
    "metrics": {
        "mape": 1.42,
        "mase": 0.207,
        "training_samples": 5652
    }
}
```

---

## Project Structure

```
retailPRED/
├── backend/                          # FastAPI backend service
│   ├── api/                         # API layer
│   │   ├── routes.py               # API endpoints
│   │   └── schemas.py              # Pydantic schemas
│   ├── ml/                         # ML models and inference
│   │   ├── inference.py            # Prediction logic
│   │   └── feature_computer.py     # Feature computation (242 features)
│   ├── services/                   # Business logic layer
│   │   ├── prediction_service.py   # Prediction logging & validation
│   │   └── counterfactual_service.py  # What-if scenarios
│   ├── db/                         # Database layer
│   │   └── database.py             # SQLite database interface
│   ├── main.py                     # FastAPI application entry point
│   └── requirements.txt            # Python dependencies
│
├── frontend/                        # React + TypeScript frontend
│   ├── src/
│   │   ├── api/                    # API client (unifiedApi.ts)
│   │   ├── components/             # UI components
│   │   │   ├── ForecastChart.tsx   # Main forecasting visualization
│   │   │   ├── FeatureImportanceChart.tsx  # SHAP visualization
│   │   │   └── ModelInfoCard.tsx   # Model performance display
│   │   ├── pages/                  # Page components
│   │   │   ├── DashboardPage.tsx   # Main dashboard
│   │   │   ├── ValidationPage.tsx  # Prediction validation tracking
│   │   │   ├── EconomicScenariosPage.tsx  # What-if analysis
│   │   │   └── SensitivityPage.tsx  # Feature sensitivity analysis
│   │   └── services/               # Data services
│   │       └── demoDataService.ts  # Static data loading for demo mode
│   ├── public/
│   │   └── demo-data/              # Pre-generated demo data
│   │       ├── predictions.json    # 7,873 predictions
│   │       ├── summary.json        # Model metadata
│   │       └── economic-indicators.json  # Economic indicators
│   └── package.json
│
├── project_root/                   # Training and data pipeline
│   ├── config/                     # Configuration files
│   ├── data_raw/                   # Raw data from sources
│   ├── data_processed/             # Merged raw data
│   ├── data_multi_resolution/      # Engineered features (OUTPUT)
│   ├── models/                     # Training scripts
│   │   └── robust_timecopilot_trainer.py  # Main training script
│   ├── training_outputs/           # Training results
│   │   ├── models/                 # Trained model files (.pkl)
│   │   ├── visualizations/         # Performance plots
│   │   ├── training_report.md      # Training summary
│   │   └── robust_training_summary.json  # Training metrics
│   ├── etl/                        # Data processing scripts
│   │   ├── build_dataset.py        # Data merging
│   │   ├── build_multi_resolution_dataset.py  # Feature engineering
│   │   ├── fetch_fred.py           # FRED data fetcher
│   │   ├── fetch_mrts.py           # MRTS data fetcher
│   │   └── fetch_yahoo.py          # Yahoo data fetcher
│   └── sqlite/                     # SQLite dataset builder
│       └── sqlite_dataset_builder.py  # Build dataset from raw data
│
├── data/                            # Runtime data directory
│   └── retailpred.db               # SQLite database (predictions, validation)
│
├── scripts/                         # Utility scripts
│   └── export-for-demo.py          # Export database to JSON for demo
│
├── docker-compose.yml              # Docker deployment configuration
├── vercel.json                     # Vercel deployment configuration
├── README.md                       # This file
└── WEBREADME.md                    # Web app deployment guide
```

---

## Quick Start

### Prerequisites

- Python 3.9 or higher
- Node.js 18 or higher
- Git

### 1. Clone Repository

```bash
git clone https://github.com/oleeveeuh/retailPRED.git
cd retailPRED
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Initialize database
python -m db.database init

# Start backend server
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Backend runs on: http://localhost:8000
API Docs: http://localhost:8000/docs

### 3. Frontend Setup

Open new terminal:

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

Frontend runs on: http://localhost:5173

### 4. Train Models (Optional)

If training models from scratch:

```bash
cd project_root/models
python robust_timecopilot_trainer.py
```

**Note:** Pre-trained models are already included in `training_outputs/models/`

---

## API Reference

### Prediction Endpoints

#### Generate Forecast

```
GET /api/predict
```

**Query Parameters:**
- `category` (required): Retail category key
- `model_name` (optional): 'LGBM', 'RandomForest', 'AutoARIMA', 'AutoETS', 'SeasonalNaive', 'PatchTST', 'TimesNet'
- `weeks_ahead` (required): 1-52
- `granularity` (required): 'daily', 'weekly', or 'monthly'

**Example:**
```bash
curl "http://localhost:8000/api/predict?category=total_sales&weeks_ahead=4&granularity=weekly&model_name=LGBM"
```

#### Get Prediction History

```
GET /api/predictions/history
```

**Query Parameters:**
- `model_name` (optional): Filter by model
- `start_date` (optional): Start date (YYYY-MM-DD)
- `end_date` (optional): End date (YYYY-MM-DD)
- `limit` (optional): Max results (default: 100)

#### Validate Prediction

```
POST /api/predictions/validate
```

**Body:**
```json
{
  "prediction_id": 123,
  "actual_value": 1525.75,
  "notes": "Actual sales from POS system"
}
```

### Category Endpoints

#### List Categories

```
GET /api/categories/list
```

Returns all 11 retail categories with keys and display names.

#### Get Category Models

```
GET /api/categories/{category}/models
```

Returns available models for a specific category.

### Model Endpoints

#### List Models

```
GET /api/models?active_only=true
```

Returns all trained models with metadata.

#### Get Training Metrics

```
GET /api/training-metrics/models
```

Returns comprehensive training metrics for all models.

### Interactive Documentation

Visit http://localhost:8000/docs for interactive API documentation (Swagger UI).

---

## Deployment

### Vercel Deployment (Demo Mode)

The live demo uses static JSON files exported from the database for zero-backend deployment.

**Build Configuration:** `vercel.json`

**Demo Mode:** Enabled via `VITE_DEMO_MODE=true` environment variable

**Data Export:**
```bash
# Export database to JSON
python scripts/export-for-demo.py
```

**Output:** `frontend/public/demo-data/` containing:
- `predictions.json`: 7,873 predictions
- `summary.json`: Model metadata
- `economic-indicators.json`: Economic indicators

**Deploy to Vercel:**
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel
```

### Docker Deployment (Full Stack)

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

Services:
- **backend**: FastAPI on port 8000
- **frontend**: React build served by nginx on port 80
- **database**: SQLite volume mounted

### Manual Deployment

**Backend:**
```bash
cd backend
docker build -t retailpred-backend .
docker run -p 8000:8000 retailpred-backend
```

**Frontend:**
```bash
cd frontend
npm run build
docker build -t retailpred-frontend .
docker run -p 80:80 retailpred-frontend
```

---

## Project Background

This project was completed as part of a machine learning portfolio to demonstrate:

- **Full-stack ML engineering**: From data collection to production deployment
- **Time series forecasting**: Using both statistical and deep learning approaches
- **Model explainability**: Implementing SHAP values for interpretability
- **Production deployment**: Static site deployment with Vercel
- **Interactive visualization**: Real-time forecast exploration with React

## Key Learnings

Through this project, I gained experience with:

- **Data engineering**: Building multi-resolution datasets from heterogeneous sources
- **Feature engineering**: Creating 242 features including lag, rolling statistics, and economic indicators
- **Model selection**: Comparing 7 algorithms (LightGBM, Random Forest, AutoARIMA, AutoETS, Seasonal Naive, PatchTST, TimesNet)
- **Model interpretation**: Understanding why SHAP only works with tree-based models
- **Frontend development**: Building responsive React applications with TypeScript
- **Deployment strategies**: Implementing zero-backend deployment with static data

## Technologies Used

**Python Backend:**
- FastAPI, LightGBM, scikit-learn, statsmodels, SHAP, SQLite, Pandas, NumPy

**TypeScript Frontend:**
- React 19, TanStack Query, Recharts, TailwindCSS, Framer Motion, Vite

**Deployment:**
- Vercel (frontend), Docker (full-stack)

---

*Last Updated: January 8, 2026*
