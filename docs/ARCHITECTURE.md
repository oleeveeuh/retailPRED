# RetailPRED System Architecture

Complete architectural overview of the RetailPRED forecasting platform, including components, data flow, technology stack, and deployment architecture.

---

## Table of Contents

- [System Overview](#system-overview)
- [Component Architecture](#component-architecture)
- [Data Flow](#data-flow)
- [Technology Stack](#technology-stack)
- [Frontend Architecture](#frontend-architecture)
- [Backend Architecture](#backend-architecture)
- [Database Schema](#database-schema)
- [Model Architecture](#model-architecture)
- [Deployment Architecture](#deployment-architecture)
- [Scalability Considerations](#scalability-considerations)

---

## System Overview

RetailPRED is a full-stack macroeconomic retail forecasting system that combines multi-source data ingestion, machine learning models, and interactive visualizations.

```
┌─────────────────────────────────────────────────────────────┐
│                      USER INTERACTION                        │
│  (Web Browser, Dashboard, API Clients)                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND LAYER                            │
│  React 19 + TypeScript + Vite                               │
│  - Data Visualization (Recharts)                            │
│  - Interactive Dashboards                                    │
│  - Model Explainability (SHAP)                              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    API LAYER                                 │
│  FastAPI + Python 3.9+                                       │
│  - RESTful Endpoints                                         │
│  - Prediction Service                                        │
│  - Data Validation                                           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   BUSINESS LOGIC                             │
│  - Forecasting Engine (Multi-Model Ensemble)                 │
│  - Feature Engineering                                       │
│  - SHAP Explainability                                       │
│  - Economic Scenario Modeling                                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    DATA LAYER                                │
│  SQLite (Development) / PostgreSQL (Production)              │
│  - Predictions (7,873 records)                               │
│  - Model Metrics                                             │
│  - Economic Indicators                                       │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  EXTERNAL DATA SOURCES                       │
│  - FRED API (Federal Reserve Economic Data)                 │
│  - MRTS (Monthly Retail Trade Survey)                       │
│  - Yahoo Finance (Stock Market Data)                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### Frontend Components

```
frontend/
├── src/
│   ├── pages/              # Route-level components
│   │   ├── Dashboard.tsx           # Main dashboard
│   │   ├── PredictionsPage.tsx    # Prediction history
│   │   ├── ModelsPage.tsx         # Model comparison
│   │   ├── ExplainPage.tsx        # SHAP explainability
│   │   ├── ValidationPage.tsx     # Validation tracking
│   │   ├── BusinessDashboard.tsx  # Tableau integration
│   │   └── EconomicScenarioAnalysis.tsx
│   │
│   ├── components/         # Reusable components
│   │   ├── Dashboard.tsx           # Summary cards
│   │   ├── ForecastChart.tsx      # Time series visualization
│   │   ├── ModelInfoCard.tsx      # Model metrics display
│   │   ├── FeatureImportanceChart.tsx  # SHAP visualization
│   │   ├── TableauEmbed.tsx        # Tableau integration
│   │   └── layout/                 # Layout components
│   │       ├── Sidebar.tsx
│   │       ├── Header.tsx
│   │       └── Layout.tsx
│   │
│   ├── api/                # API layer
│   │   ├── client.ts               # Axios client
│   │   └── unifiedApi.ts           # Demo/Real API switcher
│   │
│   ├── services/           # Business logic
│   │   └── demoDataService.ts      # Static JSON loader
│   │
│   └── config/             # Configuration
│       └── environment.ts          # Environment variables
│
└── public/
    └── demo-data/          # Static JSON for demo mode
        ├── predictions.json
        ├── economic-indicators.json
        └── summary.json
```

### Backend Components

```
backend/
├── main.py                 # FastAPI application entry
├── api/                    # API route handlers
│   ├── predictions.py      # Prediction endpoints
│   ├── models.py           # Model information
│   ├── scenarios.py        # Economic scenarios
│   └── export.py           # Data export
│
├── services/               # Business logic
│   ├── prediction_service.py
│   ├── model_service.py
│   └── data_export_service.py
│
├── models/                 # ML models
│   ├── train_multi_resolution.py
│   ├── robust_timecopilot_trainer.py
│   └── long_term_forecaster.py
│
└── db/                     # Database layer
    ├── database.py         # SQLite connection
    └── schema.sql          # Database schema
```

---

## Data Flow

### 1. Data Ingestion Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ FRED API     │────▶│ ETL Scripts  │────▶│ SQLite DB    │
│ MRTS Data    │     │ (fetch_*.py) │     │ (raw_data)   │
└──────────────┘     └──────────────┘     └──────────────┘
                                                   │
                                                   ▼
                                        ┌──────────────┐
                                        │ Feature      │
                                        │ Engineering │
                                        └──────────────┘
                                                   │
                                                   ▼
                                        ┌──────────────┐
                                        │ Training     │
                                        │ Data (242    │
                                        │ features)    │
                                        └──────────────┘
```

### 2. Model Training Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Training     │────▶│ Model        │────▶│ Trained      │
│ Data         │     │ Training     │     │ Models       │
│ (2010-2024)  │     │ (LightGBM,   │     │ (22 models)  │
└──────────────┘     │ RandomForest)│     └──────────────┘
                     └──────────────┘              │
                                                   ▼
                                        ┌──────────────┐
                                        │ Model        │
                                        │ Metrics      │
                                        │ (MAPE, RMSE) │
                                        └──────────────┘
```

### 3. Prediction Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ User Request │────▶│ API Endpoint │────▶│ Load Model   │
│ (POST /predict)    │ (/predict)   │     │ from Disk    │
└──────────────┘     └──────────────┘     └──────────────┘
                                                  │
                                                  ▼
                                       ┌──────────────┐
                                       │ Generate     │
                                       │ Prediction   │
                                       └──────────────┘
                                                  │
                          ┌─────────────────────────┤
                          ▼                         ▼
                   ┌──────────────┐         ┌──────────────┐
                   │ Calculate    │         │ Compute SHAP │
                   │ Confidence   │         │ Values       │
                   │ Interval     │         └──────────────┘
                   └──────────────┘                  │
                          │                         │
                          └──────────┬───────────────┘
                                     ▼
                          ┌──────────────┐
                          │ Save to DB   │
                          │ + Return     │
                          └──────────────┘
```

### 4. Frontend Data Flow (Demo Mode)

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ User Opens   │────▶│ React App    │────▶│ Check Mode   │
│ Browser      │     │ Loads        │     │ (demo=true)  │
└──────────────┘     └──────────────┘     └──────────────┘
                                                   │
                                                   ▼
                                        ┌──────────────┐
                                        │ Load Static  │
                                        │ JSON Files   │
                                        │ (demo-data/) │
                                        └──────────────┘
                                                   │
                                                   ▼
                                        ┌──────────────┐
                                        │ Display Data │
                                        │ (No API)     │
                                        └──────────────┘
```

---

## Technology Stack

### Frontend Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| **React** | 19.2.0 | UI framework |
| **TypeScript** | 5.9.3 | Type safety |
| **Vite** | 5.4.11 | Build tool |
| **React Router** | 7.11.0 | Routing |
| **React Query** | 5.90.16 | Data fetching |
| **Recharts** | 3.6.0 | Data visualization |
| **Tailwind CSS** | 4.1.18 | Styling |
| **Framer Motion** | 12.23.26 | Animations |
| **Axios** | 1.13.2 | HTTP client |

### Backend Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.9+ | Backend language |
| **FastAPI** | 0.100+ | API framework |
| **Uvicorn** | Latest | ASGI server |
| **SQLite** | 3 | Database (dev) |
| **Pydantic** | 2.0+ | Data validation |
| **LightGBM** | 4.0+ | ML models |
| **Scikit-learn** | 1.3+ | ML utilities |
| **SHAP** | 0.41+ | Model explainability |

### ML Model Stack

| Model | Purpose | Count |
|-------|---------|-------|
| **LightGBM** | Primary forecasting | 11 |
| **RandomForest** | Secondary/Ensemble | 11 |
| **AutoARIMA** | Time series baseline | 11 |
| **AutoETS** | Exponential smoothing | 11 |
| **SeasonalNaive** | Simple baseline | 11 |
| **PatchTST** | Deep learning (optional) | - |
| **TimesNet** | Deep learning (optional) | - |

---

## Frontend Architecture

### Component Hierarchy

```
App
└── Layout
    ├── Sidebar
    │   ├── Navigation
    │   └── User Menu
    └── Main Content
        ├── Dashboard Route
        │   ├── Summary Cards
        │   ├── Forecast Chart
        │   └── Model Info Cards
        ├── Predictions Route
        │   ├── Filter Bar
        │   └── Predictions Table
        ├── Models Route
        │   └── Model Comparison Cards
        ├── Explainability Route
        │   ├── Category Selector
        │   └── SHAP Charts
        ├── Validation Route
        │   └── Validation Table
        └── Business Dashboard Route
            └── Tableau Embed
```

### State Management

**React Query** for server state:
```typescript
// Predictions data
const { data: predictions } = useQuery({
  queryKey: ['predictions', filters],
  queryFn: () => predictionsApi.getPredictions(filters),
  staleTime: 5 * 60 * 1000, // 5 minutes
});

// Models data
const { data: models } = useQuery({
  queryKey: ['models'],
  queryFn: () => modelsApi.getAllModels(),
});
```

**Local state** with React hooks:
```typescript
const [selectedCategory, setSelectedCategory] = useState('total_sales');
const [dateRange, setDateRange] = useState({ start, end });
```

### API Layer

**Unified API Pattern** (Demo/Real mode switch):

```typescript
// config/environment.ts
export const config = {
  isDemoMode: import.meta.env.VITE_DEMO_MODE === 'true',
  apiUrl: import.meta.env.VITE_API_URL || 'http://localhost:8000',
};

// api/unifiedApi.ts
export const api = config.isDemoMode
  ? {
      ...demoPredictionsApi,
      ...demoModelsApi,
      // ... other demo APIs
    }
  : {
      ...realPredictionsApi,
      ...realModelsApi,
      // ... other real APIs
    };
```

### Routing

```typescript
// App.tsx
<Routes>
  <Route path="/" element={<Layout />}>
    <Route index element={<Dashboard />} />
    <Route path="predictions" element={<PredictionsPage />} />
    <Route path="models" element={<ModelsPage />} />
    <Route path="explain" element={<ExplainPage />} />
    <Route path="validation" element={<ValidationPage />} />
    <Route path="business-dashboard" element={<BusinessDashboard />} />
  </Route>
</Routes>
```

---

## Backend Architecture

### API Structure

```
backend/
└── main.py
    ├── /health              # Health check
    ├── /api
    │   ├── /predictions     # Prediction CRUD
    │   │   ├── GET /               # List all predictions
    │   │   ├── GET /{id}           # Get single prediction
    │   │   ├── POST /predict       # Generate new prediction
    │   │   └── GET /export/csv     # Export predictions
    │   │
    │   ├── /models          # Model information
    │   │   ├── GET /               # List all models
    │   │   ├── GET /{id}           # Get model details
    │   │   └── GET /metrics        # Get model metrics
    │   │
    │   ├── /categories      # Retail categories
    │   │   └── GET /               # List categories
    │   │
    │   ├── /scenarios       # Economic scenarios
    │   │   ├── POST /analyze       # What-if analysis
    │   │   ├── POST /sensitivity   # Sensitivity analysis
    │   │   └── GET /similar-periods
    │   │
    │   ├── /economic-indicators
    │   │   └── GET /current        # Get current indicators
    │   │
    │   └── /training-metrics
    │       └── GET /models         # Training metrics
    │
    └── /docs               # Auto-generated API docs (Swagger)
```

### Service Layer

```python
# services/prediction_service.py
class PredictionService:
    def generate_prediction(
        self,
        category: str,
        model_name: str,
        weeks_ahead: int
    ) -> Prediction:
        # 1. Load model
        model = self.load_model(category, model_name)
        
        # 2. Fetch latest features
        features = self.fetch_latest_features(category)
        
        # 3. Generate prediction
        prediction = model.predict(features)
        
        # 4. Calculate confidence interval
        confidence = self.calculate_confidence(model, features)
        
        # 5. Compute SHAP values
        shap_values = self.compute_shap(model, features)
        
        # 6. Save to database
        self.save_prediction(prediction, confidence, shap_values)
        
        return prediction
```

### Dependency Injection

```python
# main.py
from fastapi import FastAPI, Depends
from services.prediction_service import PredictionService

app = FastAPI()

# Dependency
def get_prediction_service():
    return PredictionService(db_session=Session())

# Route with dependency
@app.post("/api/predict")
def predict(
    category: str,
    model_name: str,
    service: PredictionService = Depends(get_prediction_service)
):
    return service.generate_prediction(category, model_name)
```

---

## Database Schema

### Tables

```sql
-- Predictions table
CREATE TABLE prediction_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    store_id INTEGER,
    product_id INTEGER,
    prediction_date TEXT NOT NULL,
    predicted_value REAL NOT NULL,
    actual_value REAL,
    confidence_interval_lower REAL,
    confidence_interval_upper REAL,
    is_validated BOOLEAN DEFAULT FALSE,
    error_percentage REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'completed'
);

-- Model metrics table
CREATE TABLE model_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    category TEXT NOT NULL,
    metric_name TEXT NOT NULL,  -- RMSE, MAE, MAPE, etc.
    metric_value REAL NOT NULL,
    test_period TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- SHAP values table
CREATE TABLE shap_values (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL,
    feature_name TEXT NOT NULL,
    shap_value REAL NOT NULL,
    FOREIGN KEY (prediction_id) REFERENCES prediction_log(id)
);

-- Economic indicators table
CREATE TABLE economic_indicators (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    indicator_name TEXT NOT NULL,
    value REAL NOT NULL,
    date TEXT NOT NULL,
    source TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

### Relationships

```
prediction_log (1) ─────┐
                        │
                        │ (1:N)
                        │
                   shap_values (N)
                        │
        ┌───────────────┼───────────────┐
        │               │               │
    feature_1       feature_2       feature_3
    (SHAP value)    (SHAP value)    (SHAP value)
```

---

## Model Architecture

### Multi-Model Ensemble

```
┌────────────────────────────────────────────────────┐
│              Multi-Model Ensemble                  │
├────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────────┐  ┌──────────────┐               │
│  │  LightGBM    │  │ RandomForest │               │
│  │  (Primary)   │  │  (Ensemble)  │               │
│  └──────┬───────┘  └──────┬───────┘               │
│         │                 │                        │
│         └────────┬────────┘                        │
│                  ▼                                 │
│         ┌───────────────┐                         │
│         │ Weighted Avg  │                         │
│         │ Ensemble      │                         │
│         └───────┬───────┘                         │
│                 │                                 │
│  ┌──────────────┼──────────────┐                 │
│  ▼              ▼               ▼                 │
│ AutoARIMA    AutoETS     SeasonalNaive            │
│ (Baseline)    (Baseline)    (Baseline)            │
│                                                     │
└────────────────────────────────────────────────────┘
```

### Feature Engineering Pipeline

```
Raw Data (5,814 daily observations, 2010-2025)
    │
    ▼
┌──────────────────────────────────────┐
│   Temporal Features                   │
│   - Day of week                       │
│   - Month                             │
│   - Quarter                           │
│   - Year                              │
│   - Is holiday                        │
│   - Is month end                      │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│   Economic Features (FRED)           │
│   - GDP growth                       │
│   - Unemployment rate                │
│   - CPI                              │
│   - Interest rates                   │
│   - Consumer confidence               │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│   Stock Market Features (Yahoo)      │
│   - S&P 500 index                    │
│   - Dow Jones                        │
│   - Retail sector ETF                │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│   Lag Features                       │
│   - Sales lags (1, 7, 30 days)       │
│   - Moving averages (7, 30, 90 days) │
│   - Year-over-year growth            │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│   Interaction Features               │
│   - Holiday × month                  │
│   - Economic × trend                 │
│   - Seasonality × trend              │
└──────────────┬───────────────────────┘
               │
               ▼
    Final Feature Set (242 features)
```

### SHAP Explainability

```
Model Prediction
    │
    ▼
┌──────────────────────────────────────┐
│   SHAP Value Calculation             │
│                                     │
│  1. TreeExplainer (for tree models)  │
│  2. Compute SHAP values              │
│  3. Aggregate by feature             │
│  4. Rank by importance               │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│   Visualization                      │
│                                     │
│  - Feature importance bar chart      │
│  - Positive/negative impact          │
│  - Interaction effects               │
│  - Temporal trends                   │
└──────────────────────────────────────┘
```

---

## Deployment Architecture

### Current Deployment (Vercel - Demo Mode)

```
                    ┌─────────────┐
                    │   User      │
                    │  Browser    │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   Vercel    │
                    │   CDN       │
                    └──────┬──────┘
                           │
                    ┌──────────▼──────────┐
                    │  Static Build       │
                    │  (dist/)            │
                    │  - index.html       │
                    │  - assets/          │
                    │  - demo-data/       │
                    └─────────────────────┘
```

**Characteristics**:
- ✅ No backend required
- ✅ Static JSON data
- ✅ Global CDN
- ✅ Automatic HTTPS
- ✅ Zero infrastructure cost

### Full Stack Deployment (Docker)

```
┌────────────────────────────────────────────────┐
│                 Docker Host                    │
├────────────────────────────────────────────────┤
│                                                 │
│  ┌──────────────┐    ┌──────────────┐         │
│  │   nginx      │    │   Backend    │         │
│  │  (Port 80)   │◄───┤  (Port 8000) │         │
│  │              │    │              │         │
│  │  ┌─────────┐ │    │  ┌─────────┐ │         │
│  │  │Frontend │ │    │  │FastAPI  │ │         │
│  │  │Build    │ │    │  │App      │ │         │
│  │  └─────────┘ │    │  └─────────┘ │         │
│  └──────────────┘    └──────┬───────┘         │
│                             │                 │
│                             ▼                 │
│                      ┌──────────────┐        │
│                      │  SQLite DB   │        │
│                      │  (Volume)    │        │
│                      └──────────────┘        │
│                                                 │
└────────────────────────────────────────────────┘
```

**Characteristics**:
- ✅ Real-time predictions
- ✅ Full API access
- ✅ Database persistence
- ❌ Requires infrastructure
- ❌ Higher complexity

---

## Scalability Considerations

### Frontend Scaling

| Technique | Implementation | Impact |
|-----------|----------------|--------|
| **Code Splitting** | Dynamic imports with React.lazy | Reduces initial bundle size |
| **Lazy Loading** | React Suspense for components | Faster initial load |
| **Memoization** | React.memo, useMemo, useCallback | Reduces re-renders |
| **Virtual Scrolling** | react-virtual for large lists | Improves performance |
| **CDN Caching** | Vercel edge network | Global distribution |

### Backend Scaling

| Technique | Implementation | Impact |
|-----------|----------------|--------|
| **Connection Pooling** | SQLAlchemy pool | Reduces DB connections |
| **Caching** | Redis for model predictions | Reduces compute time |
| **Async Processing** | FastAPI async endpoints | Improves throughput |
| **Rate Limiting** | slowapi middleware | Prevents abuse |
| **Horizontal Scaling** | Kubernetes/Docker Swarm | Increases capacity |

### Database Scaling

| Technique | Implementation | Impact |
|-----------|----------------|--------|
| **Indexing** | Strategic indexes on common queries | Faster queries |
| **Partitioning** | Partition by date | Better query performance |
| **Replication** | Read replicas | Distributes load |
| **Connection Pooling** | PgBouncer (PostgreSQL) | Manages connections |

---

## Security Architecture

### Authentication & Authorization

```
┌─────────────────────────────────────────────┐
│         Security Layers                     │
├─────────────────────────────────────────────┤
│ 1. HTTPS/TLS (Transport Layer)             │
│ 2. API Key Authentication (Application)     │
│ 3. Rate Limiting (DDoS Prevention)          │
│ 4. Input Validation (SQL Injection Prevention) │
│ 5. CORS Policy (Browser Security)           │
└─────────────────────────────────────────────┘
```

### Data Protection

- **Environment Variables**: All secrets in `.env` files (never committed)
- **Database Encryption**: SQLite encryption at rest (optional)
- **API Security**: CORS, rate limiting, input validation
- **Dependency Scanning**: `npm audit` and `pip-audit`

---

## Monitoring & Observability

### Frontend Monitoring

```typescript
// Error boundary
class ErrorBoundary extends React.Component {
  componentDidCatch(error, errorInfo) {
    // Log to service (e.g., Sentry)
    console.error('Error:', error, errorInfo);
  }
}

// Performance monitoring
const perfObserver = new PerformanceObserver((list) => {
  for (const entry of list.getEntries()) {
    console.log('Performance:', entry.name, entry.duration);
  }
});
perfObserver.observe({ entryTypes: ['measure'] });
```

### Backend Monitoring

```python
# Logging
import logging
logger = logging.getLogger(__name__)

# Metrics (Prometheus)
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()
Instrumentator().instrument(app).expose(app)
```

---

## Future Enhancements

### Planned Improvements

1. **Real-time Updates**
   - WebSocket integration for live predictions
   - Server-sent events for model training status

2. **Advanced Analytics**
   - Time series decomposition visualization
   - Correlation matrix for features
   - Anomaly detection

3. **Multi-tenancy**
   - User authentication
   - Custom model deployments
   - Per-user data isolation

4. **Model Optimization**
   - Automated hyperparameter tuning
   - Model versioning (MLflow)
   - A/B testing framework

5. **Infrastructure**
   - Kubernetes deployment
   - Terraform infrastructure as code
   - CI/CD pipeline enhancement

---

**Last Updated**: January 7, 2025
**Architecture Version**: 1.0
**Maintained By**: RetailPRED Team

For deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md).
