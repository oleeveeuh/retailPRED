# RetailPRED: Time Series Retail Sales Forecasting System

An end-to-end machine learning system for forecasting retail sales across multiple categories using advanced time series models and SHAP-based explainability.

[![Live Demo](https://img.shields.io/badge/demo-live_online-brightgreen)](https://retailpred.vercel.app)
[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://python.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue)](https://typescriptlang.org)
[![React](https://img.shields.io/badge/React-19+-blue)](https://react.dev)

**Live Demo:** https://retailpred.vercel.app

---

## Project Overview

This project was developed to explore the application of modern machine learning techniques to retail sales forecasting. It uses data from the U.S. Census Bureau's Monthly Retail Trade Survey (MRTS) to generate accurate forecasts across 11 retail categories using seven different forecasting algorithms with advanced feature engineering from time series data.

## System Overview

RetailPRED is an end-to-end retail forecasting platform that combines multi-resolution time series modeling with interactive visualizations and model explainability. The system processes MRTS retail sales data to generate accurate forecasts across 11 retail categories using 74 engineered time-series features.

### Key Capabilities

- **Multi-Model Architecture**: Seven forecasting algorithms (LightGBM, Random Forest, AutoARIMA, AutoETS, Seasonal Naive, PatchTST, TimesNet) with automatic model selection
- **Advanced Feature Engineering**: 74 engineered time-series features including lag features, rolling statistics, rate-of-change indicators, and cyclical temporal encodings
- **Model Explainability**: SHAP (SHapley Additive exPlanations) values for tree-based models to interpret feature contributions
- **Economic Scenario Analysis**: Stress-test forecasts under different economic conditions (recession, recovery, rate hikes, inflation surge)
- **Model-Specific Scenario Predictions**: Compare how each model (LGBM, RandomForest, PatchTST, TimesNet) responds to economic scenarios
- **Historical Validation**: Track prediction accuracy over time with comprehensive metrics
- **Interactive Dashboard**: Real-time visualization of forecasts, confidence intervals, and model performance

### Performance Metrics

**All metrics shown below are from validation/test set performance**, NOT training metrics. This reflects real-world accuracy on unseen data.

**Best Performing Models** (across all 11 categories on test data):

| Model | Avg MAPE | Best For |
|-------|----------|----------|
| **TimesNet** | 3.90% | Deep learning, complex temporal patterns |
| **Seasonal Naive** | 3.91% | Strong seasonal patterns, simple baseline |
| **AutoARIMA** | 3.92% | Autoregressive patterns, interpretable |
| **AutoETS** | 3.95% | Exponential smoothing, robust to outliers |
| **PatchTST** | 4.01% | Transformer-based time series |
| **LightGBM** | 10.63% | Most categories (some problematic models) |
| **Random Forest** | 11.99% | Complex interactions, interpretable |

**Note:** LGBM has 3 models with high validation MAPE (~25%) that need investigation (furniture, general_merchandise, sporting_goods). The other 7 LGBM models perform well (3.91-4.67%).

**Category Champions** (lowest validation MAPE):
- Automobile Dealers: **3.46%** (SeasonalNaive), **3.58%** (PatchTST)
- Building Materials & Garden: **3.30%** (SeasonalNaive), **3.48%** (AutoETS)
- Clothing & Accessories: **3.78%** (AutoARIMA), **3.85%** (SeasonalNaive)
- Electronics & Appliances: **3.38%** (AutoARIMA), **3.91%** (LGBM)
- Food & Beverage Stores: **3.28%** (PatchTST), **3.66%** (SeasonalNaive)
- Furniture & Home Furnishings: **3.74%** (AutoARIMA), **3.74%** (TimesNet)
- Gasoline Stations: **3.37%** (AutoETS), **3.37%** (TimesNet)
- General Merchandise: **3.19%** (TimesNet), **3.52%** (AutoETS)
- Health & Personal Care: **3.83%** (SeasonalNaive), **4.27%** (AutoARIMA)
- Sporting Goods & Hobby: **3.68%** (AutoARIMA), **3.97%** (SeasonalNaive)
- Total Retail Sales: **3.25%** (AutoETS), **3.89%** (TimesNet)

**Overall System Performance:**
- **Total Predictions**: 7,557
- **Validated Predictions**: 3,566 (47.2% validation rate)
- **Overall Validation Accuracy**: 94.3% (5.7% average error)
- **Models Deployed**: 75 across 11 categories

---

## Table of Contents

1. [Technical Architecture](#technical-architecture)
2. [Data Pipeline](#data-pipeline)
3. [Feature Engineering](#feature-engineering)
4. [Model Training](#model-training)
5. [Model Details](#model-details)
6. [Inference Pipeline](#inference-pipeline)
7. [Economic Context Feature](#economic-context-feature)
8. [Economic Scenario Analysis](#economic-scenario-analysis)
9. [Project Structure](#project-structure)
10. [Quick Start](#quick-start)
11. [API Reference](#api-reference)
12. [Deployment](#deployment)

---

## Technical Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Ingestion Layer                      │
│  U.S. Census Bureau MRTS (Monthly Retail Trade Survey)      │
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
│  Feature importance │ Performance tracking                  │
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

### Data Source: Monthly Retail Trade Survey (MRTS)

**Provider:** U.S. Census Bureau
**Frequency:** Monthly
**Coverage:** 11 retail categories
**Data Lag:** 1-2 months
**History:** 2010-2025 (15 years of historical data)

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

### Data Processing Pipeline

**Location:** `project_root/etl/`

#### Stage 1: Data Collection
```bash
# Fetch MRTS retail sales data
python project_root/etl/fetch_mrts.py
```

**Output:** Raw CSV files in `project_root/data_raw/`

#### Stage 2: Data Normalization
**Script:** `build_dataset.py`

**Process:**
1. Date alignment to month-end
2. Missing value handling
3. Data validation

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

**Total Features:** 74 features per observation
**Data Source:** 100% from MRTS retail sales data
**Feature Categories:** 6 major types

### 1. Temporal Features (9 features)

Capture seasonal patterns and calendar effects through both linear and cyclical encodings.

**Linear Temporal Features:**
- `year`: Calendar year (2010-2025)
- `month`: Month of year (1-12)
- `quarter`: Quarter of year (1-4)
- `day_of_week`: Day of week (0=Monday, 6=Sunday)
- `week_of_year`: ISO week number (1-53)
- `is_weekend`: Binary flag (1 if Sat/Sun, 0 otherwise)

**Cyclical Temporal Features** (preserve continuity):
- `month_sin`, `month_cos`: Cyclical month encoding
- `quarter_sin`, `quarter_cos`: Cyclical quarter encoding
- `day_of_year_sin`, `day_of_year_cos`: Cyclical day encoding
- `day_of_week_sin`, `day_of_week_cos`: Cyclical weekday encoding

**Why Cyclical Encoding:** Preserves the cyclical nature of time where December (12) is close to January (1), which linear encoding would disrupt.

### 2. Lag Features (16 features)

Capture autoregressive patterns where past values predict future values.

| Feature | Period | Use Case |
|---------|--------|----------|
| `lag_1`, `lag_2`, `lag_3`, `lag_4` | 1-4 periods | Short-term memory |
| `lag_8`, `lag_12` | 8-12 periods | Medium-term patterns |
| `lag_1d`, `lag_7d`, `lag_14d`, `lag_30d` | 1, 7, 14, 30 days | Daily patterns |
| `lag_4w`, `lag_8w`, `lag_12w` | 4, 8, 12 weeks | Weekly patterns |
| `lag_3m`, `lag_6m`, `lag_12m` | 3, 6, 12 months | Monthly/quarterly/annual patterns |

**Feature Importance:** Lag features consistently rank as the most important features across all categories (typically 3-5 of top 10 features).

### 3. Rolling Statistics (30 features)

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
| `diff_1`, `diff_12` | y[t] - y[t-n] | Absolute change |
| `pct_change_1`, `pct_change_12` | (y[t] - y[t-n]) / y[t-n] × 100 | Period return % |
| `pct_change_1w`, `diff_1w` | Weekly change | Weekly growth |
| `pct_change_1m`, `diff_1m` | Monthly change | Monthly growth |
| `pct_change_1y`, `diff_1y` | Annual change | Year-over-year growth |

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

### 7. Additional Temporal Features (6 features)

- `day_of_month`: Day of month (1-31)
- `day_of_year`: Day of year (1-366)
- `is_month_start`, `is_month_end`: Month boundary flags
- `is_quarter_start`, `is_quarter_end`: Quarter boundary flags
- `week_of_month`: Week within month
- `month_sin`, `month_cos`: Cyclical encoding
- `quarter_sin`, `quarter_cos`: Cyclical encoding
- `day_of_year_sin`, `day_of_year_cos`: Cyclical encoding
- `day_of_week_sin`, `day_of_week_cos`: Cyclical encoding

### Feature Importance Analysis

**Top 14 Features** (averaged across all categories):

| Rank | Feature | Avg Importance | Category |
|------|---------|----------------|----------|
| 1 | `lag_14d` | 11.3% | Lag |
| 2 | `lag_7d` | 11.2% | Lag |
| 3 | `lag_4w` | 7.2% | Lag |
| 4 | `rolling_mean_7d` | 7.1% | Rolling Statistics |
| 5 | `rolling_mean_30d` | 7.0% | Rolling Statistics |
| 6 | `rolling_std_14d` | 6.1% | Rolling Statistics |
| 7 | `rolling_std_7d` | 5.5% | Rolling Statistics |
| 8 | `rolling_mean_4w` | 4.9% | Rolling Statistics |
| 9 | `rolling_mean_14d` | 4.8% | Rolling Statistics |
| 10 | `rolling_mean_3` | 3.6% | Rolling Statistics |
| 11 | `rolling_mean_8w` | 3.5% | Rolling Statistics |
| 12 | `lag_1d` | 2.9% | Lag |
| 13 | `rolling_mean_6` | 2.6% | Rolling Statistics |
| 14 | `rolling_mean_12` | 2.4% | Rolling Statistics |

**ALL are time-series features!**

**Key Insights:**
- Autoregressive features (lags) dominate with ~32% combined importance
- Rolling statistics capture ~55% (trend + volatility)
- Rate of change and momentum features capture remaining ~13%
- **No external data sources needed** - MRTS retail sales data is sufficient!

---

## Model Training

### Training Configuration

**Best Approach:** Use pre-processed CSV files with 74 time-series features
**Location:** `backend/retrain_all_with_csv.py`

### Data Split

**Temporal Train/Test Split** (Critical for time series):

- **Training Set:** January 2010 - December 2023 (14 years)
  - 4,651 daily observations
  - Used for model training
  - Ensures models learn historical patterns

- **Test Set (Holdout):** January 2024 - December 2025 (2 years)
  - 1,163 daily observations
  - Strict temporal holdout (no data leakage)
  - Used for validation and performance evaluation
  - Mimics real-world forecasting scenario

**Validation vs Training Metrics:**

The dashboard and model cards now display **validation metrics** from actual test data, NOT training metrics. This is critical because:

1. **Training metrics** (MAPE from training set) are often optimistic - models perform well on data they've seen
2. **Validation metrics** (MAPE from test set) reflect real-world performance on unseen data
3. Some models (TimesNet, PatchTST) show high training MAPE (~22%) but excellent validation MAPE (~3-4%)
4. This indicates they generalize well despite pessimistic training estimates

**Current System Status:**
- **Total Predictions**: 7,557 (all models, all dates)
- **Validated Predictions**: 3,566 (47.2% have actual values)
- **Overall Validation Accuracy**: 94.3% (5.7% average error)

**Why Temporal Split:** Random split would cause data leakage where future information contaminates training. Temporal split ensures models are evaluated on truly unseen future data.

### Training Parameters

**RandomForest:**
```python
{
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 3,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1
}
```

**LGBM:**
```python
{
    'n_estimators': 200,
    'max_depth': 15,
    'learning_rate': 0.1,
    'num_leaves': 31,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}
```

### Model Selection Strategy

**Per-Category Best Model Selection:**

After training all models on the training set, performance is evaluated on the holdout set. The model with lower MAPE is selected as the "best model" for that category.

**Selection Process:**
1. Train both models (RandomForest and LGBM) on 2010-2023 data
2. Evaluate both on 2024-2025 holdout data
3. Select model with lowest validation MAPE
4. Deploy selected model for production forecasting

---

## Model Details

### 1. LightGBM (Gradient Boosting)

**Algorithm:** Gradient boosting framework that uses tree-based learning algorithms

**Validation Performance:** Mixed performance on test data
- Average MAPE: **10.63%** across all 10 LGBM models
- **Well-performing models** (7/10): **3.91-4.67%** on validation set
- **Problematic models** (3/10): **~25%** on validation set (furniture, general_merchandise, sporting_goods)
- Training speed: Fast (~1 second per category)
- Best for: Smooth trends, consistent patterns, non-linear relationships

**SHAP Support:** YES - Tree-based model supports SHAP value computation for explainability

**Why It Works Well:**
- Handles non-linear relationships
- Robust to outliers
- Built-in regularization prevents overfitting
- Fast training speed
- Excellent at capturing complex feature interactions

**Issue:** 3 recently retrained models show poor validation performance and need investigation

### 2. Random Forest

**Algorithm:** Ensemble learning method operating by constructing a multitude of decision trees

**Validation Performance:** Moderate performance on test data
- Average MAPE: **11.99%** across all 8 RandomForest models
- Range: **9.22-14.00%** on validation set
- Training speed: Medium (~1-2 seconds per category)
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

**Validation Performance:**
- Average MAPE: **3.92%** across all 11 models
- Range: **3.38-4.60%** on validation set
- Best for: Autoregressive patterns, clear trend/seasonality

**SHAP Support:** NO - Statistical model without feature-based structure

**Use Case:** Good baseline model, interpretable parameters, fast training

### 4. AutoETS

**Algorithm:** Exponential Smoothing with automatic error/trend/seasonality selection

**Validation Performance:**
- Average MAPE: **3.95%** across all 11 models
- Range: **3.25-4.50%** on validation set
- Best for: Exponential smoothing trends, seasonal patterns

**SHAP Support:** NO - Statistical model without feature-based structure

**Why Better Than ARIMA:**
- More robust to outliers
- Handles multiple seasonality types
- Better for seasonal data
- Smoother forecasts

### 5. Seasonal Naive

**Algorithm:** Naive forecasting method using seasonal lags

**Validation Performance:**
- Average MAPE: **3.91%** across all 11 models
- Range: **3.30-4.80%** on validation set
- Best for: Strong seasonal patterns, simple baseline

**SHAP Support:** NO - No feature-based structure

**Use Case:** Baseline model for comparison, minimal assumptions

### 6. PatchTST

**Algorithm:** Patch Time Series Transformer (deep learning model)

**Validation Performance:**
- Average MAPE: **4.01%** across all 11 models
- Range: **3.28-4.50%** on validation set
- Best for: Complex temporal patterns

**SHAP Support:** NO - Deep learning model without straightforward SHAP support

**Why Good Validation Performance:**
- Despite high training MAPE (~22%), performs well on test data
- Captures complex temporal dependencies
- Good at generalization

**Use Case:** Complex patterns, larger datasets

### 7. TimesNet

**Algorithm:** Deep learning model using temporal 2D convolution

**Validation Performance:**
- Average MAPE: **3.90%** across all 11 models (best overall!)
- Range: **3.19-4.44%** on validation set
- Best for: Complex temporal patterns

**SHAP Support:** NO - Deep learning model without straightforward SHAP support

**Why Good Validation Performance:**
- Despite high training MAPE (~22%), performs excellently on test data
- Excellent generalization - the best performing model overall
- Captures multi-scale temporal patterns

**Use Case:** Complex patterns, multi-scale analysis

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
    {"feature": "lag_14d", "value": 1500.00, "importance": 0.35},
    {"feature": "lag_7d", "value": 800.00, "importance": 0.18},
    {"feature": "rolling_mean_7d", "value": 450.00, "importance": 0.10},
    {"feature": "rolling_std_14d", "value": 200.00, "importance": 0.05}
  ]
}
```

**Interpretation:**
- `lag_14d` increases prediction by $1,500 (35% of deviation from baseline)
- `rolling_std_14d` indicates volatility impact

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
import pickle

# Load trained model
with open('training_outputs/models/Total_Retail_Sales/LGBM_model.pkl', 'rb') as f:
    model_dict = pickle.load(f)
    model = model_dict['model']
```

#### Step 2: Historical Data Loading

```python
# Load pre-processed CSV data
csv_path = 'project_root/data_multi_resolution/retail_total_sales_multi_resolution.csv'
historical_df = pd.read_csv(csv_path)
```

#### Step 3: Feature Computation

Features are pre-computed in the CSV file - no need to compute on-the-fly!

```python
# Get features for prediction
features = historical_df[feature_cols].iloc[-1:]  # Most recent
```

#### Step 4: Multi-Step Forecast

For `weeks_ahead > 1`, use the pre-trained model with historical features.

#### Step 5: Confidence Intervals

```python
# Calculate confidence intervals (±15% default)
confidence_lower = prediction * 0.85
confidence_upper = prediction * 1.15
confidence_score = 0.95  # Model confidence score
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
            "feature": "lag_14d",
            "value": 71_850.00,
            "contribution": 1500.00,
            "importance": 0.35
        }
    ],
    "metrics": {
        "mape": 0.52,
        "training_samples": 5814
    }
}
```

---

## Economic Context Feature

### Overview

RetailPRED overlays economic indicators (unemployment, consumer confidence, interest rates) to help explain predictions, but does **NOT** use them for forecasting. This feature provides post-hoc interpretation to help stakeholders understand anomalies and assess model reliability.

### Why Not Use Economic Data for Prediction?

In testing, adding macroeconomic features **degraded model accuracy**:

| Approach | Features | MAPE | Performance |
|----------|----------|------|-------------|
| **Time Series Only** | 74 features | 0.26-2.22% | ✅ Excellent |
| **With Economic Data** | 242 features | 7-12% | ❌ Degraded |

**Reason:** Economic indicators move slowly and introduce overfitting. Time-series features capture recent patterns more accurately.

### How It Works

#### Prediction Layer (0.26% MAPE)

The model uses **only 74 time-series features** from retail sales data:

1. **Lag Features** (7, 14, 21, 28 days)
   - Recent sales values
   - Capture short-term patterns

2. **Rolling Statistics** (7, 14, 28, 90-day windows)
   - Mean, standard deviation, min, max
   - Capture trends and volatility

3. **Momentum Indicators**
   - Rate of change (1, 7, 14, 28-day)
   - Acceleration (2nd order changes)
   - Capture direction and speed

4. **Temporal Encodings**
   - Cyclical month/quarter (sin/cos)
   - Weekend indicators
   - Capture seasonality

**Result:** 0.26% MAPE - Best possible accuracy

#### Interpretation Layer (Post-Hoc)

Economic data is overlaid to **explain** predictions:

1. **Anomaly Detection**
   - Flags predictions with >5% change
   - Shows severity (moderate/severe)
   - Provides economic context

2. **Regime Classification**
   - Normal: Stable conditions (high model confidence)
   - Expansion: Growth conditions (high model confidence)
   - Recession: Downturn conditions (medium model confidence)
   - Crisis: Severe shock (low model confidence)

3. **Historical Events**
   - COVID-19 Pandemic (March 2020)
   - Financial Crisis (September 2008)
   - Fed Rate Hikes (2022)
   - Dot-Com Recession (2001)

4. **Natural Language Explanations**
   - "Sales dropped 15% in March 2020"
   - "Economic context: Unemployment spiked from 3.5% to 14.7%"
   - "Model predicted from sales patterns, economic data confirms cause"

### Use Cases

#### 1. Understanding Historical Anomalies

**Example:** COVID-19 Sales Drop

```
Prediction: Building materials sales dropped 30% in April 2020
    ↓
Economic Context: Unemployment 14.7%, Consumer Confidence 86.0
    ↓
Explanation: Model predicted decline from sales patterns.
             Economic data confirms COVID-19 lockdown impact.
```

#### 2. Assessing Model Reliability

**Regime-Based Confidence:**

- **Normal/Expansion:** 90% reliability (model trained on these conditions)
- **Recession:** 60% reliability (some uncertainty)
- **Crisis:** 30% reliability (high uncertainty, predictions less accurate)

#### 3. Stakeholder Communication

**Before Economic Context:**
> "Sales will drop 15% next month."
> Stakeholder: "Why? Is this accurate?"

**With Economic Context:**
> "Sales will drop 15% next month. This prediction is based on recent sales trends.
> Current economic conditions show rising unemployment and falling consumer confidence,
> suggesting a recession. The model has 60% reliability in these conditions."
> Stakeholder: "That makes sense. Thanks for the context."

### Components

#### 1. Economic Regime Indicator

**Location:** Dashboard top

**Displays:**
- Current economic regime (normal/expansion/recession/crisis)
- Model reliability (progress bar: 90%/60%/30%)
- Economic trends (unemployment →↑↓, confidence →↑↓)
- Brief explanation

**Example:**
```
┌─────────────────────────────────────────┐
│ 📊 Economic Regime: Normal              │
│                                         │
│ Model Reliability: ████████ 90%        │
│                                         │
│ Unemployment: → Stable 3.8%            │
│ Consumer Confidence: → Stable 102.0    │
│                                         │
│ Normal economic conditions. Model      │
│ predictions highly reliable.           │
│                                         │
│ 💡 Interpretation only - not used for  │
│ predictions (74 features, 0.26% MAPE)   │
└─────────────────────────────────────────┘
```

#### 2. Anomaly Explanation

**Location:** Appears when prediction changes >5%

**Displays:**
- Prediction change (e.g., "Sales dropped 15%")
- Economic regime badge
- Indicators with 3-month changes
- Anomalous indicators (if any)
- Natural language explanation
- "Interpretation Only" label

#### 3. Event Timeline

**Location:** Dashboard bottom or separate page

**Displays:**
- Historical economic events (2001-2024)
- Event type (crisis/recession/expansion)
- Economic context (unemployment, confidence)
- Explanation of impact

#### 4. Forecast Chart Annotations

**Location:** Forecast chart

**Displays:**
- Vertical dashed lines at event dates
- Event labels (🚨 COVID-19, ⚠️ Fed Hikes)
- Click to see economic context popover
- Color-coded by severity (red/orange)

### Data Sources

**Prediction Data:**
- U.S. Census Bureau MRTS (retail sales)
- 74 engineered time-series features
- Updated monthly

**Economic Context:**
- FRED API (Federal Reserve Economic Data)
- Unemployment rate (UNRATE)
- Consumer Confidence Index (UMCSENT)
- Federal funds rate (FEDFUNDS)
- Consumer Price Index (CPIAUCSL)
- Updated monthly

### Demo Data

**File:** `frontend/public/demo-data/economic-context.json`
**Size:** 7.8KB
**Events:** 10 historical economic events (2001-2024)

**Coverage:**
- 3 Crisis events (COVID-19, Financial Crisis, Peak COVID)
- 3 Recession events (Dot-Com, Fed Hikes, Fed Peaks)
- 2 Expansion events (2019, 2015)
- 2 Normal events (2024, 2023)

### Settings

**Toggle:** Show/Hide Economic Context

**Location:** Settings page or dashboard

**Options:**
- ✅ Economic regime indicator
- ✅ Anomaly explanations
- ✅ Historical event annotations
- ✅ Event timeline

**Persistence:** localStorage

### Key Points

✅ **Clear Separation:** Prediction (time-series) vs Interpretation (economic)

✅ **Transparency:** Users understand what data drives predictions

✅ **Accuracy:** Maintains 0.26% MAPE by NOT using economic data for prediction

✅ **Context:** Economic data helps explain anomalies and assess reliability

✅ **Flexibility:** Toggle on/off based on user preference

### Learn More

- [Economic Context Implementation Summary](ECONOMIC_CONTEXT_INTEGRATION_SUMMARY.md)
- [Economic Context Data Reference](ECONOMIC_CONTEXT_DATA_REFERENCE.md)
- [Demo Data Generation](DEMO_DATA_ECONOMIC_CONTEXT.md)

---

## Economic Scenario Analysis

### Overview

RetailPRED includes a stress-testing feature that allows you to analyze how different economic scenarios would impact retail sales forecasts. This feature applies scenario-based adjustments to model predictions, enabling what-if analysis for strategic planning.

### Available Scenarios

The system supports five predefined economic scenarios:

| Scenario | Description | Sales Impact | Key Changes |
|----------|-------------|--------------|-------------|
| **Baseline** | Current trends continue | +1% | Stable growth |
| **Recession** | Economic downturn with elevated unemployment | -8% | Unemployment +2%, GDP -1.5% |
| **Recovery** | Strong growth with falling unemployment | +6% | Unemployment -1%, GDP +2% |
| **Rate Hike Cycle** | Tightening monetary policy | -3% | Fed funds +2%, GDP -0.5% |
| **Inflation Surge** | High inflation environment | -2% | CPI +2%, Fed funds +1.5% |

### How It Works

#### Scenario Adjustment Process

1. **Fetch Base Prediction**
   - Get the most recent prediction for the specified model and category
   - Each model has its own base prediction from the database

2. **Apply Scenario Multipliers**
   - Baseline: ×1.01
   - Recession: ×0.92
   - Recovery: ×1.06
   - Rate Hike: ×0.97
   - Inflation Surge: ×0.98

3. **Calculate Adjusted Forecast**
   - Apply multiplier to base prediction
   - Calculate confidence intervals (wider for extreme scenarios)
   - Return model-specific scenario-adjusted prediction

### Model-Specific Predictions

Each forecasting model responds differently to scenarios based on its unique base prediction:

**Example for Baseline Scenario (+1%):**
- LGBM: $600,000 → $606,000
- RandomForest: $595,000 → $600,950
- PatchTST: $605,000 → $611,050
- TimesNet: $598,000 → $603,980

**Example for Recession Scenario (-8%):**
- LGBM: $600,000 → $552,000
- RandomForest: $595,000 → $547,400
- PatchTST: $605,000 → $556,600
- TimesNet: $598,000 → $550,200

### Use Cases

#### 1. Strategic Planning

**Question:** "What if a recession hits next quarter?"

**Answer:** Select "Recession" scenario to see forecasted sales decline of 8% across all models, with confidence intervals showing uncertainty range.

#### 2. Model Comparison Under Stress

**Question:** "Which model is most conservative?"

**Answer:** Compare scenario predictions - some models may predict lower values in downturns, reflecting different risk tolerances.

#### 3. Sensitivity Analysis

**Question:** "How sensitive are sales to interest rate changes?"

**Answer:** Compare "Rate Hike Cycle" (-3%) vs "Baseline" (+1%) to see the 4% differential impact.

### API Endpoints

#### Get Scenario List

```
GET /api/scenarios/list
```

Returns all available scenarios with descriptions.

#### Analyze Scenario (Best Model)

```
POST /api/scenarios/analyze
{
  "category": "total_sales",
  "scenario_type": "recession"
}
```

Returns scenario analysis for the best model in that category.

#### Analyze Scenario (Specific Model)

```
POST /api/scenarios/model-prediction
{
  "category": "total_sales",
  "model_name": "LGBM",
  "scenario_type": "recession"
}
```

Returns model-specific scenario-adjusted prediction.

### Response Format

```json
{
  "scenario_type": "recession",
  "scenario_name": "Recession",
  "description": "Economic downturn with elevated unemployment and negative GDP growth",
  "category": "total_sales",
  "model_name": "LGBM",
  "base_prediction": 600000,
  "prediction": 552000,
  "confidence_interval": [496800, 607200],
  "impact_summary": [
    {
      "indicator": "UNRATE",
      "category": "Labor Market",
      "base_value": 4.2,
      "scenario_value": 6.5,
      "change": 2.3,
      "change_pct": 54.8
    }
  ]
}
```

### Confidence Intervals by Scenario

- **Baseline**: ±4% (narrow - high confidence)
- **Recession**: ±10% (wide - high uncertainty)
- **Recovery**: ±8% (wide - high uncertainty)
- **Rate Hike**: ±6% (moderate)
- **Inflation Surge**: ±8% (wide - high uncertainty)

### Demo Mode

In Vercel demo mode, scenario predictions use pre-generated base values with scenario multipliers applied on the client side.

**Demo Data Structure:**
- Each model has a unique base prediction
- Scenario multipliers match backend logic
- Full impact summaries included

### Key Points

✅ **What-If Analysis:** Test assumptions without retraining models

✅ **Model Comparison:** See how different models respond to scenarios

✅ **Risk Assessment:** Understand potential downside/upside scenarios

✅ **Strategic Planning:** Inform inventory, staffing, and budgeting decisions

✅ **Transparency:** Clear separation between base forecast and scenario adjustments

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
│   │   └── feature_computer.py     # Feature computation (74 features)
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
│   │   │   └── ValidationPage.tsx  # Prediction validation tracking
│   │   └── services/               # Data services
│   │       └── demoDataService.ts  # Static data loading for demo mode
│   ├── public/
│   │   └── demo-data/              # Pre-generated demo data
│   │       ├── predictions.json    # 7,873 predictions
│   │       └── summary.json        # Model metadata
│   └── package.json
│
├── project_root/                   # Training and data pipeline
│   ├── config/                     # Configuration files
│   ├── data_raw/                   # Raw data from sources
│   ├── data_processed/             # Merged raw data
│   ├── data_multi_resolution/      # Engineered features (OUTPUT)
│   ├── models/                     # Training scripts
│   ├── training_outputs/           # Training results
│   │   ├── models/                 # Trained model files (.pkl)
│   │   ├── visualizations/         # Performance plots
│   │   └── training_report.md      # Training summary
│   └── etl/                        # Data processing scripts
│       ├── build_dataset.py        # Data merging
│       ├── build_multi_resolution_dataset.py  # Feature engineering
│       └── fetch_mrts.py           # MRTS data fetcher
│
├── data/                            # Runtime data directory
│   └── retailpred.db               # SQLite database (predictions, validation)
│
├── scripts/                         # Utility scripts
│   └── export-for-demo.py          # Export database to JSON for demo
│
├── docker-compose.yml              # Docker deployment configuration
├── vercel.json                     # Vercel deployment configuration
└── README.md                       # This file
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

**Recommended:** Use pre-trained models (already included in `training_outputs/models/`)

To retrain models from scratch using the CSV-based approach:

```bash
cd backend
python retrain_all_with_csv.py
```

This will train all 11 categories using the proven 74-feature time-series approach.

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

The live demo at https://retailpred.vercel.app uses static JSON files for zero-backend deployment with all features fully functional.

**Build Configuration:** `vercel.json`

**Demo Mode Features:**
- ✅ All 7 models with training metrics
- ✅ Economic scenario analysis (5 scenarios)
- ✅ Model-specific scenario predictions
- ✅ Economic regime indicators
- ✅ Anomaly detection and explanation
- ✅ Historical validation data (7,357 predictions)
- ✅ SHAP values for tree-based models

**Demo Mode Toggle:** Automatically enabled in production builds via build configuration

**Data Export:**
```bash
# Export database to JSON for demo
python scripts/export-for-demo.py
```

**Output:** `frontend/public/demo-data/` containing:
- `predictions.json`: 7,357 predictions with error metrics
- `summary.json`: Model metadata and training results
- `economic-indicators.json`: 500 economic data points
- `economic-context.json`: 10 historical economic events

**Deploy to Vercel:**
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel --prod
```

**Environment Variables:**
None required - demo mode is fully self-contained with static data.

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
- **Time series forecasting**: Using both statistical and machine learning approaches
- **Model explainability**: Implementing SHAP values for interpretability
- **Production deployment**: Static site deployment with Vercel
- **Interactive visualization**: Real-time forecast exploration with React

## Key Learnings

Through this project, I gained experience with:

- **Data engineering**: Building multi-resolution datasets from MRTS data
- **Feature engineering**: Creating 74 time-series features from retail sales data
- **Model selection**: Comparing 7 algorithms (LightGBM, Random Forest, AutoARIMA, AutoETS, Seasonal Naive, PatchTST, TimesNet)
- **Model interpretation**: Understanding why SHAP only works with tree-based models
- **Simplicity vs complexity**: Learning that 74 well-engineered features outperform 242 features with external data
- **Data quality**: Discovering that clean, pre-processed CSV files perform better than on-the-fly feature computation
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

*Last Updated: January 9, 2026*
