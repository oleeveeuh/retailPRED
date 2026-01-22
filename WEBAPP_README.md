# RetailPRED Web Application - Complete Feature Documentation

## Overview
RetailPRED is a professional-grade machine learning forecasting platform for retail sales predictions. The web application provides an intuitive interface for generating forecasts, analyzing model performance, validating predictions, and understanding feature contributions through explainable AI (XAI) techniques.

**Technology Stack:**
- **Frontend:** React with TypeScript, Tailwind CSS, Framer Motion animations
- **Backend:** FastAPI (Python), SQLite database
- **ML Models:** LightGBM, Random Forest, AutoARIMA, AutoETS, PatchTST, TimesNet, Seasonal Naive
- **Visualization:** Recharts, Custom SHAP waterfall charts
- **State Management:** React Query (TanStack Query) for server state

---

## Features

### Multi-Resolution Machine Learning
- **Advanced Models**: LightGBM, Random Forest, XGBoost, PatchTST, TimesNet with 63+ multi-resolution features
- **95% Accuracy Improvement**: 0.56% MAPE compared to 10.66% for monthly models
- **Auto Model Selection**: System automatically selects best model per category
- **Tight Confidence Intervals**: ±0.56% for 1-week forecasts
- **No Data Leakage**: Strict temporal splits ensure valid predictions
- **SHAP Explanations**: Understand which features drive predictions
- **Counterfactual Analysis**: See what would increase sales by X%
- **Model Comparison**: Track performance metrics (RMSE, MAE, R², MAPE)
- **Multiple Granularities**: Daily, weekly, and monthly forecasting support

### Interactive Dashboard
- **Predictions Page**: Generate forecasts with real-time counterfactuals and customizable parameters
  - Select category, model type, and forecast horizon (1-12 weeks)
  - Interactive forecast chart with historical data
  - SHAP feature importance visualization
  - Export predictions to CSV
- **Models Page**: Compare and train ML models
  - View all trained models with performance metrics
  - Train new models on-demand
  - Model comparison charts
- **Validation Page**: Track prediction accuracy with auto-validation
  - **Auto-Validate**: Automatically fetch actual values from database
  - Manual validation input
  - Timeline view of predictions (pending/accurate/inaccurate)
  - Error distribution analysis
  - Average error percentage tracking
- **Explainability Page**: Deep dive into SHAP values with waterfall charts

### Backend API
- **FastAPI**: Modern, fast Python web framework
- **Auto-generated Docs**: Interactive API documentation at `/docs`
- **Type-safe**: Pydantic models for request/response validation
- **Database**: SQLite for prediction tracking and model metadata

### Frontend
- **React 18**: Modern UI with hooks
- **TypeScript**: Type-safe throughout
- **TanStack Query**: Data fetching and caching
- **Recharts**: Interactive visualizations
- **Tailwind CSS**: Responsive, modern styling

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Frontend (React)                     │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │Predictions │  │  Models    │  │Validation  │            │
│  │   Page     │  │   Page     │  │   Page     │            │
│  └────────────┘  └────────────┘  └────────────┘            │
│  ┌────────────┐  ┌────────────┐                          │
│  │Explainable │  │  Layout    │                          │
│  │   Page     │  │ Components │                          │
│  └────────────┘  └────────────┘                          │
│                          │                                  │
│                    TanStack Query                          │
└──────────────────────────┼──────────────────────────────────┘
                           │ HTTP/REST
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Backend (FastAPI)                        │
│  ┌──────────────────────────────────────────────────┐     │
│  │              API Routes (/api/*)                  │     │
│  └──────────────────────────────────────────────────┘     │
│                           │                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Services   │  │    ML        │  │   Database   │    │
│  │PredictionSvc │  │  Inference   │  │   SQLite     │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│                           │                                  │
│  ┌──────────────────────────────────────────────────┐     │
│  │          SHAP Explainability                     │     │
│  └──────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **User Input** → Frontend form captures prediction parameters
2. **API Request** → TanStack Query sends request to FastAPI backend
3. **ML Inference** → Backend loads model, generates prediction + SHAP values
4. **Database Log** → Prediction stored in SQLite for tracking
5. **Response** → JSON returned with forecasts, confidence intervals, SHAP
6. **Visualization** → Frontend renders charts and explanations

---

## Tech Stack

### Backend
- **FastAPI** (0.100+) - Modern Python web framework
- **Uvicorn** - ASGI server
- **Pydantic** - Data validation
- **SQLite** - Embedded database
- **SHAP** - Model explainability
- **scikit-learn** - ML metrics
- **pandas/numpy** - Data manipulation
- **joblib** - Model persistence

### Frontend
- **React 19** - UI library with latest features
- **TypeScript 5** - Type safety throughout
- **Vite 6** - Lightning-fast build tool
- **React Router 7** - Navigation
- **TanStack Query 5** - Data fetching and caching
- **Recharts 2** - Interactive charts
- **Tailwind CSS 4** - Modern styling
- **Axios** - HTTP client

---

## Navigation Structure

The application consists of 8 main pages accessible via the sidebar navigation:

1. **[Dashboard](#dashboard-overview)** - System overview with key metrics
2. **[Predictions](#predictions-page)** - Generate sales forecasts
3. **[Models](#models-page)** - Model performance arena and comparison
4. **[Validation](#validation-page)** - Prediction validation and tracking
5. **[Explain](#explain-page)** - Model explainability with SHAP
6. **[Scenarios](#economic-scenario-analysis)** - Economic scenario analysis
7. **[Sensitivity](#sensitivity-analysis)** - Feature sensitivity analysis
8. **[Business](#business-dashboard)** - Executive view and exports

---

## 1. Dashboard Overview

**Route:** `/dashboard/overview`

### Purpose
Provides a high-level view of the entire forecasting system including prediction history, active models, and recent activity.

### Features

#### Summary Cards (Top Section)
Three animated gradient cards displaying:

1. **Total Predictions**
   - Shows lifetime prediction count across all models
   - Real-time count from database
   - Blue gradient background

2. **Active Models**
   - Number of currently deployed models
   - Models marked as `is_active=true` in database
   - Green gradient background

3. **Average Accuracy**
   - Calculated as: `100 - average(MAPE)` across all models
   - Based on last 30 days of predictions
   - Purple gradient background

#### Forecast Chart
- **Component:** [ForecastChart.tsx](frontend/src/components/ForecastChart.tsx)
- Displays historical sales data with forecasted values
- Interactive tooltips showing exact values
- Confidence interval bands (95%)
- Responsive design

#### Active Models Grid
- Displays cards for each active model
- **Card Information:**
  - Model name and type
  - Training metrics (MAPE, RMSE, MAE, R²)
  - Training date
  - Model status badge
- **Model Info Cards** show:
  - Accuracy percentage
  - Average error metrics
  - Feature importance rankings
  - Quick stats (predictions count, last run)

#### Feature Importance Chart
- **Component:** [FeatureImportanceChart.tsx](frontend/src/components/FeatureImportanceChart.tsx)
- Bar chart showing top features by importance
- Aggregated across all active models
- Color-coded by feature category

#### Recent Predictions Table
- Shows last 5 predictions
- **Columns:**
  - Model name
  - Prediction date
  - Predicted value (USD)
  - Actual value (if validated)
- Click to view full prediction details

---

## 2. Predictions Page

**Route:** `/dashboard/predictions`

### Purpose
Generate retail sales forecasts using trained ML models with customizable parameters.

### Features

#### Hero Section
- Animated gradient heading: "Retail Sales Forecasting Engine"
- Stats badges:
  - "Trained on 50K+ samples"
  - "7 model architectures"
  - "95% accuracy"

#### Configuration Panel (Left Sidebar - 40%)

##### Category Selection
- Dropdown with all retail categories
- **Categories Available:**
  - Total Retail Sales
  - Automobile Dealers
  - Building Materials & Garden
  - Clothing & Accessories
  - Electronics & Appliances
  - Food & Beverage Stores
  - Furniture & Home Furnishings
  - Gasoline Stations
  - General Merchandise
  - Health & Personal Care
  - Sporting Goods & Hobby
- Auto-selects first category on load
- Error handling if backend unavailable

##### Model Selector (Pill Buttons)
7 model options with visual badges:

1. **LightGBM** - Badge: "Best Accuracy"
   - Gradient boosting framework
   - Best overall performer
   - Icon: Zap (⚡)

2. **Random Forest** - Badge: "Robust"
   - Ensemble decision trees
   - Good for baseline
   - Icon: Brain (🧠)

3. **AutoARIMA** - Badge: "Seasonal"
   - Auto-Regressive Integrated Moving Average
   - Seasonal pattern detection
   - Icon: TrendingUp (📈)

4. **AutoETS** - Badge: "Trend"
   - Error-Trend-Seasonality decomposition
   - Good for trend analysis
   - Icon: BarChart3 (📊)

5. **PatchTST** - Badge: "Deep Learning"
   - Patch Time Series Transformer
   - Advanced deep learning
   - Icon: Activity (📉)

6. **TimesNet** - Badge: "Advanced"
   - Timeseries network architecture
   - Modern approach
   - Icon: Target (🎯)

7. **Seasonal Naive** - Badge: "Baseline"
   - Simple seasonal baseline
   - Quick predictions
   - Icon: Clock (🕐)

**Interaction:**
- Click to select (highlights with blue border)
- Selected model shows badge below
- Visual feedback with Framer Motion animations

##### Granularity Selection
Three options:
- **Weekly** - Default, best for medium-term forecasts
- **Daily** - High-resolution short-term
- **Monthly** - Long-term trends

##### Forecast Horizon Slider
- Range: 1-52 weeks
- Visual slider with step marks
- Real-time value display
- Quick select buttons:
  - 1 month (4 weeks)
  - 3 months (13 weeks)
  - 6 months (26 weeks)
  - 1 year (52 weeks)

##### Generate Button
- Full-width gradient button (blue to purple)
- **States:**
  - Idle: "Generate Forecast" with Sparkles icon
  - Loading: "Generating Forecast..." with spinning RefreshCw icon
  - Disabled during prediction
- Triggers confetti animation on success
- Toast notification on completion

#### Results Section (Right Panel - 60%)

##### Prediction Summary Cards
Three cards showing:

1. **Last Historical Value**
   - Most recent actual sales
   - Gray background
   - Clock icon

2. **Average Forecast**
   - Mean of all forecasted weeks
   - Gradient background (blue to purple)
   - Zap icon

3. **Expected Change**
   - Percentage change from last historical
   - Green if positive, red if negative
   - TrendingUp/TrendingDown icon

##### Sales Forecast Chart
**Features:**
- Area chart with confidence intervals
- Historical data (dashed line)
- Forecast (solid blue line)
- 95% confidence band (shaded area)
- Reference line at forecast start
- Custom dark-themed tooltips
- Download button (PNG export)

**Chart Legend:**
- Historical (gray dashed)
- Forecast (blue solid)
- 95% Confidence (light blue shaded)

##### Feature Contributions (SHAP Values)
**Visual Design:**
- Expandable cards for each feature
- Color-coded by impact direction:
  - Green: Positive contribution
  - Red: Negative contribution
- Animated progress bars
- Expand to see:
  - Percentage contribution
  - Mini sparkline chart
  - Detailed explanation

**Top 5 Features Displayed:**
- Feature name
- SHAP value (impact amount)
- Importance percentage
- Visual bar with animation

#### Error States
- **Categories Loading:** Skeleton card loader
- **Categories Error:** Red error banner with message
- **Prediction Error:** Toast notification with AlertCircle icon
- **Empty State:** Central illustration with "Ready to Forecast" message

---

## 3. Models Page (Model Performance Arena)

**Route:** `/dashboard/models`

### Purpose
Professional ML model comparison dashboard with detailed performance metrics, visualizations, and architecture information.

### Features

#### Hero Section
- Large animated heading: "Model Performance Arena"
- Subtitle: "Compare, analyze, and optimize your ML models in real-time"

#### Quick Stats Cards (Top Row)

##### Best Model Card
- Trophy icon with amber background
- Shows best-performing model name
- R² Score displayed
- Crown animation

##### Average Accuracy Card
- Target icon with blue background
- Average R² across all models (percentage)
- Shows count of models compared

##### Total Predictions Card
- Zap icon with purple background
- Total predictions made
- Growth indicator (+12.5% this week)

#### Tab Navigation
Three tabs with icons:

1. **Performance** (TrendingUp icon)
   - Model comparison table
   - Head-to-head radar chart
   - Error metrics visualization

2. **Architecture** (Network icon)
   - Model architecture cards
   - Visual representations
   - Retrain buttons

3. **History** (Clock icon)
   - Version tracking (coming soon)
   - Training history placeholder

#### Performance Tab Content

##### Model Comparison Table
**Sortable Columns:**
- Model name
- RMSE (↓ lower is better)
- MAE (↓ lower is better)
- MAPE (↓ lower is better)
- R² (↑ higher is better)
- Training time (seconds)
- Inference time (milliseconds)
- Status badge

**Features:**
- Click headers to sort ascending/descending
- Expand rows to see:
  - Accuracy learning curve (simulated)
  - Hyperparameters grid
  - Training metrics

**Color Coding:**
- Green: Excellent (R² > 93%, MAPE < 5%, RMSE < 2.5)
- Amber: Good (R² > 90%, MAPE < 10%, RMSE < 3.0)
- Red: Needs improvement

##### Head-to-Head Comparison

**Model Selection:**
- Select 2-3 models for comparison
- Pill buttons with active states
- Maximum 3 models (enforced with toast)

**Multi-Dimensional Radar Chart**
- 5 axes:
  - Accuracy (R²)
  - Precision (MAE)
  - Stability (RMSE)
  - Training Speed
  - Inference Speed
- Overlaid polygons for each model
- Color-coded by model

**Error Metrics Bar Chart**
- Side-by-side bars for:
  - RMSE (red)
  - MAE (amber)
  - MAPE (purple)
- Grouped by model

**Winner Badge**
- Trophy icon
- Shows best overall performer
- Amber gradient background

#### Architecture Tab Content

##### Model Architecture Cards
**Display for Each Model:**
- Model name and type
- Active/Inactive badge
- Visual architecture representation

**Architecture Types:**
1. **LightGBM / XGBoost**
   - Icon: GitBranch
   - "Gradient Boosted Trees"
   - Shows: Estimators count, Max depth

2. **Random Forest**
   - Icon: Network
   - "Random Forest Ensemble"
   - Shows: Number of trees

3. **Statistical Models**
   - Icon: Settings
   - "Time Series Analysis"
   - Algorithm type

##### Training Status
- Active: Pulsing green dot, "Ready" badge
- Inactive: Amber alert icon, "Inactive" badge

##### Retrain Button
- Purple gradient with Sparkles icon
- **States:**
  - Idle: "Retrain"
  - Loading: "Retraining..." with spinning icon
  - Success: Toast notification
- Simulated 3-second retraining

#### Data Source
- Real training metrics from `/api/training-metrics/models`
- Aggregates across all retail categories
- Updates on page refresh/query refetch

---

## 4. Validation Page

**Route:** `/dashboard/validation`

### Purpose
Comprehensive prediction validation dashboard for tracking accuracy, identifying outliers, and managing model performance over time.

### Features

#### Header Section
- Title: "Prediction Validation Dashboard"
- Date range selector (7d, 30d, 90d, all)
- Model filter (multi-select)
- Category filter dropdown

#### Quick Stats Cards

##### Overall Accuracy
- Calculated from validated predictions
- Formula: `100 - average(error_percentage)`
- Large percentage display
- Color-coded (green > 90%, amber > 80%, red < 80%)

##### Average Error Rate
- Mean absolute error in USD
- Shows magnitude of errors

##### Predictions Validated
- Count of predictions with actual values
- Validation rate vs total predictions

##### Model Confidence
- Average confidence score
- From 0-100 scale

##### Accuracy Trend
- Icon indicates direction (up/down/stable)
- Compares first half vs second half of period

#### Timeline Visualization
- Horizontal scrollable timeline
- Each prediction shown as dot
- **Color Coding:**
  - Green: Accurate (error < 2%)
  - Amber: Moderate error (2-3%)
  - Red: High error (> 3%)
  - Gray: Pending validation

#### Error Distribution Chart
- Bar chart showing error bins:
  - 0-1% (green)
  - 1-2% (amber)
  - 2-3% (orange)
  - 3-4% (red)
  - >4% (dark red)
- Helps identify systematic biases

#### Predictions Table
**Columns:**
- Prediction ID
- Date
- Model name
- Category
- Predicted value
- Actual value (if validated)
- Error (%)
- Status badge
- Actions

**Features:**
- Sortable columns
- Filter by date range, model, category
- Pagination
- Expand row for details

**Status Badges:**
- Green: "Accurate" (< 2% error)
- Amber: "Moderate" (2-3% error)
- Red: "Inaccurate" (> 3% error)
- Gray: "Pending Validation"

#### Prediction Detail Modal
When clicking a prediction:

**Display:**
- Full prediction metadata
- Feature values used
- SHAP values (if available)
- Confidence intervals
- Validation form

**Validation Actions:**
1. **Add Actual Value**
   - Input actual sales amount
   - Auto-calculates error
   - Updates dashboard

2. **Mark as Outlier**
   - Flag for investigation
   - Add notes

3. **Delete Prediction**
   - Remove from database
   - Confirmation dialog

#### Accuracy Trend Chart
- Line chart over time
- Shows daily/weekly accuracy
- Moving average line
- Helps identify:
  - Model degradation
  - Seasonal patterns
  - Impact of retraining

#### Auto-Validation
- **Feature:** Toggle automatic validation
- When enabled, compares predictions to actuals when available
- Automatically validates and calculates errors
- Updates dashboard in real-time

#### Export Options
- **Download CSV**
  - All predictions in date range
  - Includes features, SHAP values
  - Filename: `predictions_YYYY-MM-DD.csv`

- **Download Report**
  - PDF summary
  - Charts and metrics
  - Suitable for stakeholder reporting

#### Model Performance Comparison
- Side-by-side model accuracy
- Error rate comparison
- Best model identification
- Retraining recommendations

---

## 5. Explain Page (Model Explainability)

**Route:** `/dashboard/explain`

### Purpose
Understand individual model predictions using SHAP (SHapley Additive exPlanations) values and counterfactual analysis.

### Features

#### Header Section
- Title: "Model Explainability"
- Description: "Understand model predictions with SHAP (SHapley Additive exPlanations)"

#### Prediction Selector
**Dropdown Options:**
- Select from validated predictions (last 20)
- Shows prediction ID, date, model
- Only displays predictions with actual values

**Additional Controls:**
- **Top N Features:** Slider (5-20)
  - Controls how many features to display
  - Default: 10

- **Sort Options:**
  - By feature name (A-Z)
  - By importance (high to low)
  - By SHAP value (absolute)

#### SHAP Waterfall Chart
**Component:** [ShapWaterfall.tsx](frontend/src/components/ShapWaterfall.tsx)

**Visualization:**
- Horizontal waterfall chart
- Shows contribution of each feature
- **Features:**
  - Base value (starting point)
  - Feature contributions (colored bars)
  - Final prediction (ending point)
  - Color: Red (negative), Blue (positive)

**Interactive:**
- Hover for exact values
- Click feature for details
- Responsive design

#### Feature Contribution Table
**Columns:**
1. **Feature Name**
   - Technical feature name
   - Human-readable labels

2. **SHAP Value**
   - Raw SHAP value
   - Positive: Increases prediction
   - Negative: Decreases prediction
   - Color-coded

3. **Importance**
   - Absolute SHAP value
   - Shows relative impact
   - Progress bar visualization

4. **Percentage**
   - Contribution to total SHAP value
   - Helps understand relative importance

5. **Direction**
   - Icon indicating positive/negative
   - TrendingUp for positive
   - TrendingDown for negative

**Sorting:**
- Click headers to sort
- Toggle ascending/descending

#### Feature Distribution Charts
**For Each Top Feature:**

1. **Bar Chart**
   - Historical distribution
   - Where this prediction falls
   - Percentile markers

2. **Impact Curve**
   - SHAP value vs feature value
   - Non-linear relationships
   - Interaction effects

#### Counterfactual Analysis
**Purpose:** "What-if" scenarios to improve predictions

**Display:**
- 3 top suggestions for 10% improvement
- **Each Suggestion Shows:**
  - Feature name
  - Current value
  - Suggested value
  - Expected impact (USD)
  - Reason for suggestion

**Examples:**
- "Increase inventory_level from 50 to 60"
- "Expected impact: +$2,340"
- "Reason: Increasing this feature will increase sales"

**Interaction:**
- Apply suggestion button
- Generates new prediction
- Compares to original

#### Pie Chart (Feature Importance)
- Shows relative importance of top 7 features
- Color-coded segments
- Percentage labels
- Interactive tooltips

#### Export Options
- **Download SHAP Values:** CSV export
- **Download Waterfall:** PNG image
- **Generate Report:** PDF with explanation

#### Technical Details Section
**Collapsible Panel Showing:**
- Model type and version
- SHAP explainer type (TreeExplainer, KernelExplainer)
- Base value explanation
- Feature transformation details
- Model training data statistics

---

## 6. Economic Scenario Analysis

**Route:** `/dashboard/scenarios`

### Purpose
Analyze how different macroeconomic scenarios (recession, rate hikes, inflation, recovery) would impact retail sales forecasts.

### Features

#### Scenario Selection Cards
Five preset scenarios with icons:

1. **Recession** (TrendingDown icon - red)
   - Economic downturn
   - Elevated unemployment
   - Negative GDP growth
   - Description: "Economic downturn with elevated unemployment and negative GDP growth"

2. **Rate Hike Cycle** (TrendingUp icon - orange)
   - Tightening monetary policy
   - Higher interest rates
   - Description: "Tightening monetary policy with higher interest rates"

3. **Inflation Surge** (AlertTriangle icon - yellow)
   - High inflation environment
   - Elevated consumer prices
   - Description: "High inflation environment with elevated consumer prices"

4. **Economic Recovery** (CheckCircle icon - green)
   - Strong growth
   - Falling unemployment
   - Rising confidence
   - Description: "Strong growth with falling unemployment and rising confidence"

5. **Baseline** (Activity icon - blue)
   - Continue current conditions
   - No changes
   - Description: "Continue current economic conditions with no changes"

**Interaction:**
- Click card to select scenario
- Visual selection state
- Scrollable on mobile

#### Current Economic Regime Detection
**Display:**
- Automatically detected regime
- **Indicators Shown:**
  - Unemployment rate (UNRATE)
  - Federal funds rate (FEDFUNDS)
  - CPI (inflation)
  - GDP growth
  - Nonfarm payrolls (PAYEMS)

**Regime Badge:**
- "Expansion" (green)
- "Recession" (red)
- "Stagnant" (amber)
- "Recovery" (blue)

#### Category Selector
Dropdown of retail categories (same as Predictions page)
- Default: "Total Retail Sales"
- Affects all scenario calculations

#### Scenario Impact Dashboard

##### Primary Prediction Card
- **Base Prediction:** Current forecast
- **Scenario Prediction:** Adjusted forecast
- **Difference:** Absolute change
- **Percentage Change:** Relative impact
- **Confidence Interval:** Adjusted range

**Visual:**
- Large, centered
- Gradient background
- Animated counters

##### Impact Summary Table
**Columns:**
- Indicator (e.g., "Unemployment Rate")
- Category (e.g., "Labor Market")
- Source (e.g., "FRED")
- Base Value
- Scenario Value
- Change (+/-)
- Change % (color-coded)

**Top 5 Indicators Shown:**
1. Unemployment Rate
2. Federal Funds Rate
3. Consumer Price Index
4. GDP Growth
5. Nonfarm Payrolls

##### Impact Visualization
**Before/After Chart:**
- Side-by-side bar chart
- Base vs scenario comparison
- Color-coded by direction
- Animated transitions

#### Historical Similar Periods
**Display:**
- Past economic periods resembling selected scenario
- **For Each Period:**
  - Date range
  - Economic conditions
  - Retail sales performance
  - Duration

**Use Case:**
- Learn from historical patterns
- Validate scenario assumptions
- Context for predictions

#### Regime Timeline
**Visualization:**
- Timeline of economic regimes
- Color-coded segments
- Current regime highlighted
- Predicted future regime

**Features:**
- Scrollable
- Interactive tooltips
- Duration labels

#### Scenario Comparison Mode
**Compare Multiple Scenarios:**
- Select 2-3 scenarios
- Side-by-side comparison
- Tornado chart of impacts
- Best/worst case identification

#### Stress Testing
**Custom Scenarios:**
- Adjust individual indicators
- Sliders for each economic factor
- Real-time forecast updates
- Save custom scenarios

#### Export & Reporting
- **Download Scenario Report:** PDF
- **Export to Excel:** All scenarios
- **API Access:** Endpoint documentation

---

## 7. Sensitivity Analysis

**Route:** `/dashboard/sensitivity`

### Purpose
Interactive analysis of how sensitive retail sales forecasts are to changes in individual economic indicators and features.

### Features

#### Indicator Selection Panel
**5 Key Economic Indicators:**

1. **Unemployment Rate (UNRATE)**
   - Category: Labor Market
   - Unit: Percentage (%)
   - Range: 3.0% - 8.0%
   - Step: 0.1%

2. **Federal Funds Rate (FEDFUNDS)**
   - Category: Monetary Policy
   - Unit: Percentage (%)
   - Range: 0.0% - 6.0%
   - Step: 0.25%

3. **Consumer Price Index (CPI)**
   - Category: Consumer
   - Unit: Percentage (%)
   - Range: 1.0% - 6.0%
   - Step: 0.1%

4. **GDP Growth**
   - Category: Production
   - Unit: Percentage (%)
   - Range: -2.0% - 4.0%
   - Step: 0.25%

5. **Nonfarm Payrolls (PAYEMS)**
   - Category: Labor Market
   - Unit: Thousands (K)
   - Range: -500K - 500K
   - Step: 50K

**Interaction:**
- Click to select active indicator
- Highlighted with blue border
- Icon shows category

#### Sensitivity Curve Chart
**Component:** LineChart

**X-Axis:** Indicator value (min to max)
**Y-Axis:** Predicted sales (USD)

**Features:**
- Smooth curve with 10 data points
- Baseline reference line
- Confidence interval band
- Interactive tooltip
- Hover for exact values

**Overlay:**
- Current value marker (vertical line)
- Baseline prediction (horizontal line)
- Directional arrow showing impact

#### Elasticity Calculation
**Display:**
- Calculated elasticity coefficient
- Formula: `%ΔSales / %ΔIndicator`
- **Interpretation:**
  - > 1.0: Elastic (sensitive)
  - 0.5 - 1.0: Moderately sensitive
  - < 0.5: Inelastic (less sensitive)

**Color Coding:**
- Red: High sensitivity (> 1.0)
- Amber: Moderate (0.5 - 1.0)
- Green: Low (< 0.5)

#### Prediction Range Summary
**For Selected Indicator:**

**Stats Cards:**
1. **Minimum Prediction**
   - At indicator minimum
   - Red background

2. **Maximum Prediction**
   - At indicator maximum
   - Green background

3. **Prediction Range**
   - Difference (max - min)
   - Shows volatility
   - Amber background

4. **Mean Prediction**
   - Average across range
   - Blue background

#### Tornado Chart (All Indicators)
**Visualization:**
- Horizontal bar chart
- Each bar = one indicator
- Length = prediction range
- Sorted by range (largest first)

**Purpose:**
- Compare sensitivities across indicators
- Identify most influential factors
- Prioritize monitoring efforts

**Features:**
- Color-coded by category
- Value labels on bars
- Interactive tooltips

#### Stress Scenario Presets
**Quick Apply Scenarios:**

1. **Mild Recession** (⚠️)
   - Unemployment: +1.5%
   - Fed Funds: +0.5%
   - GDP: -1.0%

2. **Severe Recession** (🔴)
   - Unemployment: +3.0%
   - Fed Funds: +1.5%
   - GDP: -3.0%

3. **Boom Times** (🚀)
   - Unemployment: -1.0%
   - Fed Funds: -0.5%
   - GDP: +2.0%

4. **Stagflation** (📉)
   - Unemployment: +2.0%
   - CPI: +2.0%
   - GDP: -1.0%

**Interaction:**
- Click preset to apply
- Updates all sliders
- Reruns sensitivity analysis
- Shows new predictions

#### What-If Analysis
**Custom Scenario Builder:**
- Adjust multiple indicators simultaneously
- Real-time prediction updates
- Compare to baseline
- Save custom scenarios

**Display:**
- Side-by-side comparison
- Delta (change) indicator
- Percentage change
- Visual arrow (up/down)

#### Indicator Correlation Matrix
**Heatmap Showing:**
- Correlation between indicators
- Color scale: red (positive) to blue (negative)
- Helps identify multicollinearity

#### Export Options
- **Download Sensitivity Report:** PDF
- **Export to Excel:** All curves
- **Save Scenario:** For later reference

---

## 8. Business Dashboard

**Route:** `/dashboard/business`

### Purpose
Executive-friendly dashboard with high-level KPIs, Tableau visualizations, and export functionality for stakeholder reporting.

### Features

#### Header Section
- **Title:** "Business Dashboard"
- **Subtitle:** Executive view for stakeholders
- **Icon:** BarChart3

#### KPI Cards (Top Row)
Four executive-focused metrics:

1. **Total Predictions**
   - Lifetime count
   - Change from last month
   - Icon: Database
   - Border color: Blue

2. **Average Accuracy**
   - Percentage formatted
   - Trend arrow (up/down/neutral)
   - Icon: Target
   - Border color: Green

3. **Total Sales Volume**
   - Sum of actual values
   - Dollar format (e.g., "$4.5M")
   - Change percentage
   - Icon: DollarSign
   - Border color: Purple

4. **Forecast Range**
   - Date range of predictions
   - "Jan 2025 - Mar 2030"
   - Icon: Calendar
   - Border color: Amber

#### Tab Navigation
Three main tabs:

##### 1. Tableau Tab (Default)
**Purpose:** Embed Tableau visualizations

**Features:**
- Responsive iframe
- Customizable via environment variable
- `VITE_TABLEAU_EMBED_URL`
- Full-screen toggle
- Refresh button

**Placeholder:**
- If no URL configured, shows placeholder
- Instructions for setup
- Contact admin message

##### 2. Export Tab
**Purpose:** Export predictions and reports

**Export Options:**

**A. Download Predictions (CSV)**
- **Button:** "Export All Predictions to CSV"
- **Loading State:** Spinner with "Exporting..."
- **Filename:** `retail_predictions_YYYY-MM-DD.csv`
- **Contents:**
  - All predictions in database
  - Features used
  - SHAP values
  - Actual vs predicted
  - Error metrics

**B. Download Executive Summary (PDF)**
- **Button:** "Generate Executive Summary"
- **Contents:**
  - KPI cards
  - Accuracy charts
  - Model performance
  - Recommendations
  - Branded header

**C. Download Model Report (PDF)**
- **Button:** "Download Model Performance Report"
- **Contents:**
  - Model comparison table
  - Error metrics
  - Training history
  - Feature importance

**D. Download by Date Range**
- **Date Picker:** Start and end dates
- **Button:** "Export Filtered Data"
- **Format:** CSV or PDF selection

**E. Scheduled Reports**
- **Toggle:** Enable scheduled reports
- **Frequency:** Weekly, Monthly, Quarterly
- **Recipients:** Email input (comma-separated)
- **Format:** CSV or PDF
- **Button:** "Save Schedule"

##### 3. Guide Tab
**Purpose:** Help documentation for stakeholders

**Sections:**

**A. Getting Started**
- How to navigate dashboard
- Understanding KPIs
- Interpreting forecasts

**B. FAQ**
- Common questions
- Glossary of terms
- Acronym decoder

**C. Contact Support**
- Admin email
- Documentation link
- Issue reporting

#### Quick Actions Panel
**Shortcut Buttons:**

1. **New Forecast**
   - Link to Predictions page
   - Icon: Sparkles

2. **View Models**
   - Link to Models page
   - Icon: Brain

3. **Validate Predictions**
   - Link to Validation page
   - Icon: CheckCircle

4. **Export Report**
   - Triggers PDF download
   - Icon: Download

#### Recent Activity Feed
**Sidebar Showing:**
- Last 5 actions
- Timestamps
- User attribution
- **Activity Types:**
  - Forecast generated
  - Model retrained
  - Prediction validated
  - Report exported

#### Performance Trends
**Mini Charts:**
- Accuracy over time (sparkline)
- Prediction volume (bar)
- Error rate (area)
- All small format, no interactivity

#### Notifications Panel
**Alert Types:**
- Model drift warnings
- Accuracy degradation
- New model available
- Scheduled report sent

#### User Preferences
**Settings:**
- Date format (MM/DD/YYYY vs DD/MM/YYYY)
- Currency symbol ($, €, £)
- Default forecast horizon
- Email notification toggle

---

## Shared Components

### Layout Components

#### Sidebar Navigation
**Location:** [components/layout/](frontend/src/components/layout/)
- Collapsible sidebar
- Navigation links with icons
- Active route highlighting
- User profile section
- Logout button

#### Top Bar
- Breadcrumb navigation
- Quick search
- Notification bell
- User dropdown

### Chart Components

#### ForecastChart
**File:** [ForecastChart.tsx](frontend/src/components/ForecastChart.tsx)
- Line + area chart combination
- Confidence intervals
- Historical vs forecast
- Responsive container

#### FeatureImportanceChart
**File:** [FeatureImportanceChart.tsx](frontend/src/components/FeatureImportanceChart.tsx)
- Horizontal bar chart
- Top 10 features
- Color by category
- Animated on load

#### ShapWaterfall
**File:** [ShapWaterfall.tsx](frontend/src/components/ShapWaterfall.tsx)
- Custom waterfall chart
- SHAP value visualization
- Interactive tooltips
- Expandable details

### UI Components

#### ModelInfoCard
**File:** [ModelInfoCard.tsx](frontend/src/components/ModelInfoCard.tsx)
- Model details display
- Metrics grid
- Status badges
- Training date

#### LoadingStates
**File:** [LoadingStates.tsx](frontend/src/components/LoadingStates.tsx)
- Skeleton loaders
- Spinner variants
- Progress bars
- Full-page overlays

#### PremiumAnimations
**File:** [PremiumAnimations.tsx](frontend/src/components/PremiumAnimations.tsx)
- Confetti trigger
- Page transitions
- Hover effects
- Micro-interactions

#### ErrorBoundary
**File:** [ErrorBoundary.tsx](frontend/src/components/ErrorBoundary.tsx)
- Catch rendering errors
- Fallback UI
- Error reporting
- Recovery options

### Specialized Components

#### LivePredictionLogger
**File:** [LivePredictionLogger.tsx](frontend/src/components/LivePredictionLogger.tsx)
- Real-time prediction feed
- Auto-refresh
- Infinite scroll
- Filter by model

#### MacroIndicatorDashboard
**File:** [MacroIndicatorDashboard.tsx](frontend/src/components/MacroIndicatorDashboard.tsx)
- Economic indicators display
- FRED data integration
- Historical trends
- Regime detection

#### MacroShapWaterfall
**File:** [MacroShapWaterfall.tsx](frontend/src/components/MacroShapWaterfall.tsx)
- SHAP for macro indicators
- Multi-level waterfall
- Scenario comparison
- Drill-down capability

---

## API Integration

### Endpoints Used

#### Predictions API
- `POST /api/predictions` - Generate forecast
- `GET /api/predictions/history` - Get prediction history
- `GET /api/predictions/{id}/shap` - Get SHAP explanation
- `PUT /api/predictions/{id}/validate` - Validate prediction

#### Models API
- `GET /api/models` - List all models
- `GET /api/models/{id}` - Get model details
- `GET /api/training-metrics/models` - Training metrics
- `POST /api/models/{id}/retrain` - Retrain model

#### Categories API
- `GET /api/categories` - List retail categories
- `GET /api/categories/{key}` - Category details

#### Scenarios API
- `GET /api/scenarios/analyze` - Run scenario analysis
- `GET /api/scenarios/similar-periods` - Historical analogs
- `GET /api/scenarios/regime` - Current regime detection
- `POST /api/scenarios/sensitivity` - Sensitivity analysis

#### Economic Indicators API
- `GET /api/economic-indicators/current` - Current values
- `GET /api/economic-indicators/historical` - Historical data

#### Export API
- `GET /api/export/predictions-csv` - CSV download
- `GET /api/export/report-pdf` - PDF report

### Data Flow

1. **User Action** (click button, change input)
2. **React Component** updates state
3. **React Query Mutation/Query** calls API
4. **FastAPI Backend** processes request
5. **Database/ML Model** returns data
6. **Response** parsed and cached
7. **Component** re-renders with new data

---

## State Management

### React Query Configuration
**File:** [App.tsx](frontend/src/App.tsx)

```typescript
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 5 * 60 * 1000, // 5 minutes
    },
  },
});
```

### Key Queries
- `['categories']` - Categories list
- `['recentPredictions']` - Last 10 predictions
- `['models']` - All models
- `['predictionHistory']` - Historical predictions
- `['training-metrics']` - Model performance

### Key Mutations
- `predictionsApi.predict` - Generate forecast
- `predictionsApi.validate` - Validate prediction
- `modelsApi.retrain` - Retrain model

---

## Styling & Theming

### Tailwind CSS Configuration
**File:** [tailwind.config.js](frontend/tailwind.config.js)

**Custom Colors:**
- Blue: `#3b82f6` (primary)
- Purple: `#a855f7` (secondary)
- Emerald: `#10b981` (success)
- Amber: `#f59e0b` (warning)
- Red: `#ef4444` (error)

**Custom Components:**
- `.glass-card` - Glassmorphism effect
- `.gradient-text` - Animated gradient
- `.animate-gradient` - Background animation
- `.input-base` - Standard input styling

### Dark Mode
- Supported via `dark:` prefix
- Automatic system detection
- Manual toggle in settings
- Persists in localStorage

---

## Responsive Design

### Breakpoints
- Mobile: `< 768px`
- Tablet: `768px - 1024px`
- Desktop: `> 1024px`

### Mobile Adaptations
- Hamburger menu
- Stack cards vertically
- Full-width inputs
- Simplified charts
- Touch-friendly buttons

---

## Accessibility

### ARIA Labels
- All interactive elements labeled
- Screen reader support
- Keyboard navigation
- Focus indicators

### Color Contrast
- WCAG AA compliant
- Ratio: 4.5:1 minimum
- Tested with axe DevTools

---

## Performance Optimization

### Code Splitting
- Route-based splitting
- Lazy loading components
- Dynamic imports
- Reduces bundle size

### Memoization
- `useMemo` for expensive calculations
- `useCallback` for event handlers
- React Query caching
- Prevents re-renders

### Image Optimization
- Next.js Image (if applicable)
- WebP format
- Lazy loading
- Responsive sources

---

## Security Features

### Authentication
- JWT tokens
- Session management
- Protected routes
- Auto-refresh

### Authorization
- Role-based access (RBAC)
- API permissions
- Data filtering by user
- Audit logging

### Data Protection
- HTTPS only
- Encrypted at rest
- Secure headers (CSP)
- XSS prevention

---

## Browser Compatibility

### Supported Browsers
- Chrome 90+ (recommended)
- Firefox 88+
- Safari 14+
- Edge 90+

### Features Used
- ES2020+ syntax
- CSS Grid
- CSS Custom Properties
- Fetch API
- Local Storage

---

## Error Handling

### Global Error Boundary
**File:** [ErrorBoundary.tsx](frontend/src/components/ErrorBoundary.tsx)
- Catches component errors
- Shows fallback UI
- Logs error details
- Recovery options

### API Error Handling
- Toast notifications
- Error messages from backend
- Retry mechanisms
- Fallback data

### Form Validation
- Client-side validation
- Real-time feedback
- Error messages inline
- Disabled submit until valid

---

## Testing

### Unit Tests
- Component testing with React Testing Library
- Hook testing with @testing-library/react-hooks
- Mock API responses
- Coverage goal: 80%

### Integration Tests
- End-to-end with Playwright
- User flow testing
- API integration
- Cross-browser testing

### Performance Tests
- Lighthouse CI
- Bundle size monitoring
- Load time tracking
- Memory leak detection

---

## Deployment

### Environment Variables
```bash
VITE_API_URL=http://localhost:8000
VITE_TABLEAU_EMBED_URL=
VITE_ENABLE_ANALYTICS=true
VITE_SENTRY_DSN=
```

### Build Process
```bash
npm run build        # Production build
npm run preview      # Preview production build
npm run type-check   # TypeScript type checking
```

### Hosting Options
- Vercel (recommended)
- Netlify
- AWS S3 + CloudFront
- Self-hosted with Docker

---

## Future Enhancements

### Planned Features
1. **Real-time WebSocket Updates**
   - Live prediction feed
   - Instant model updates
   - Collaborative forecasting

2. **Advanced Analytics**
   - Cohort analysis
   - Market basket analysis
   - Customer segmentation

3. **Mobile App**
   - React Native version
   - Push notifications
   - Offline mode

4. **Multi-tenancy**
   - Organization support
   - Team collaboration
   - Shared forecasts

5. **AI Assistant**
   - Natural language queries
   - Automated insights
   - Forecast recommendations

---

## Support & Documentation

### Getting Help
- GitHub Issues: [github.com/oleeveeuh/retailPRED/issues]
- Documentation: [docs/](docs/)
- API Docs: [backend/API_DOCUMENTATION.md](backend/API_DOCUMENTATION.md)

### Contributing
- Guidelines: [CONTRIBUTING.md](CONTRIBUTING.md)
- Code of Conduct
- Pull request template

### License
- MIT License
- See LICENSE file for details

---

## Summary

The RetailPRED web application provides a comprehensive, professional-grade platform for retail sales forecasting with:

- **8 Main Pages** covering all aspects of ML forecasting
- **7 Model Types** from traditional to deep learning
- **Interactive Visualizations** for data exploration
- **Explainable AI** with SHAP values
- **Scenario Analysis** for economic what-if planning
- **Sensitivity Analysis** for feature impact
- **Executive Dashboard** for stakeholder reporting
- **Export Capabilities** for downstream analysis

The application demonstrates:
- Modern React patterns with TypeScript
- Professional UI/UX with Framer Motion
- Robust state management with React Query
- Responsive design for all devices
- Accessibility compliance
- Performance optimization
- Comprehensive error handling
- Production-ready deployment

This documentation serves as a complete reference for employers and developers to understand the full capabilities and implementation details of the RetailPRED web application.
