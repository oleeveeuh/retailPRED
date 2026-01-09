# RetailPRED Web Application: Technical Architecture & Implementation

React application for visualizing retail sales forecasts with interactive model explanations, scenario analysis, and real-time validation tracking.

**Live Demo:** https://retailpred.vercel.app

---

## Project Overview

This web application was developed as part of a machine learning portfolio project to explore the intersection of time series forecasting and interactive data visualization. The frontend demonstrates modern React development practices while communicating complex forecasting results through intuitive visualizations.

---

## Table of Contents

1. [Technical Overview](#technical-overview)
2. [Technology Stack](#technology-stack)
3. [Application Architecture](#application-architecture)
4. [Page-by-Page Functionality](#page-by-page-functionality)
5. [Vercel Deployment](#vercel-deployment)
6. [Development Workflow](#development-workflow)
7. [Performance Optimizations](#performance-optimizations)
8. [Key Technical Achievements](#key-technical-achievements)

---

## Technical Overview

### Application Type

**Demo Mode Deployment:** Static React application with pre-generated JSON data
- Zero backend dependencies for production deployment
- All data embedded at build time via `export-for-demo.py`
- Enables instant loading and global CDN distribution via Vercel
- Real-time mode available for local development with FastAPI backend

### Architecture Philosophy

**Separation of Concerns:**
- **Data Layer:** `unifiedApi.ts` provides unified interface for both demo and live modes
- **Presentation Layer:** React components focused purely on UI rendering
- **State Management:** TanStack Query handles server state, caching, and synchronization
- **Configuration:** Environment-based configuration via `environment.ts`

**Design Patterns:**
- **Repository Pattern:** `demoDataService.ts` abstracts data access
- **Adapter Pattern:** `unifiedApi.ts` adapts different data sources to common interface
- **Container/Presentational:** Components separate logic from rendering
- **Custom Hooks:** Reusable stateful logic (e.g., `useForecast`, `useValidation`)

---

## Technology Stack

### Frontend Framework

**React 19**
- Latest React features including automatic batching
- Functional components with hooks
- Context API for global state

**TypeScript 5.0+**
- Full type safety across the application
- Strict null checks enabled
- Interface definitions for all API contracts

```typescript
// Example: Type-safe API contracts
export interface ForecastRequest {
  category: string;
  model_name?: string;
  weeks_ahead: number;
  granularity: 'daily' | 'weekly' | 'monthly';
}

export interface ForecastResponse {
  prediction_id: number;
  model_name: string;
  forecasts: Forecast[];
  shap_values?: SHAPValue[];
  metrics: ModelMetrics;
}
```

### Build Tool

**Vite 5.x**
- Lightning-fast hot module replacement (HMR)
- Optimized production builds with Rollup
- Native ES modules support
- TypeScript support out of the box

```bash
# Development server with HMR
npm run dev

# Production-optimized build
npm run build

# Preview production build locally
npm run preview
```

### Data Fetching

**TanStack Query (React Query)**
- Automatic caching and revalidation
- Background refetching
- Optimistic updates
- Parallel query execution

```typescript
// Example: Automatic caching with TanStack Query
const { data: forecastData, isLoading, error } = useQuery({
  queryKey: ['forecast', category, weeksAhead],
  queryFn: () => predictionsApi.generateForecast(category, weeksAhead),
  staleTime: 5 * 60 * 1000, // 5 minutes
  cacheTime: 10 * 60 * 1000, // 10 minutes
});
```

### Data Visualization

**Recharts**
- Declarative chart components
- Responsive SVG charts
- Custom tooltips and formatters
- Animation support

```typescript
// Example: Responsive line chart with custom styling
<ResponsiveContainer width="100%" height={400}>
  <LineChart data={forecastData}>
    <CartesianGrid strokeDasharray="3 3" />
    <XAxis dataKey="date" tickFormatter={formatDate} />
    <YAxis tickFormatter={formatCurrency} />
    <Tooltip content={<CustomTooltip />} />
    <Legend />
    <Line
      type="monotone"
      dataKey="predicted_value"
      stroke="#3b82f6"
      strokeWidth={2}
      dot={false}
    />
    <Line
      type="monotone"
      dataKey="actual_value"
      stroke="#10b981"
      strokeWidth={2}
      dot={false}
    />
  </LineChart>
</ResponsiveContainer>
```

### Styling

**TailwindCSS 3.x**
- Utility-first CSS framework
- Dark mode support via `dark:` prefix
- Responsive design with mobile-first approach
- JIT compiler for minimal CSS bundle size

```typescript
// Example: Responsive dark mode styling
<div className="bg-white dark:bg-slate-800 rounded-lg shadow p-6 hover:shadow-lg transition-shadow">
  <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
    Forecast Results
  </h3>
</div>
```

### Animations

**Framer Motion**
- Declarative animation API
- Page transitions
- Stagger animations for lists
- Gesture support (drag, hover, tap)

```typescript
// Example: Staggered list animation
{items.map((item, index) => (
  <motion.div
    key={item.id}
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ delay: index * 0.1 }}
  >
    {item.content}
  </motion.div>
))}
```

### Icons

**Lucide React**
- Lightweight icon library
- Tree-shakeable to minimize bundle size
- Consistent stroke width and design

---

## Application Architecture

### Directory Structure

```
frontend/
├── src/
│   ├── api/                      # Data access layer
│   │   ├── unifiedApi.ts        # Unified API interface
│   │   └── index.ts             # API exports
│   ├── components/              # Reusable UI components
│   │   ├── ForecastChart.tsx   # Main forecast visualization
│   │   ├── FeatureImportanceChart.tsx  # SHAP chart
│   │   ├── ModelInfoCard.tsx   # Model performance card
│   │   ├── MetricCard.tsx      # KPI display
│   │   └── Navigation.tsx      # Navigation bar
│   ├── config/                  # Configuration files
│   │   └── environment.ts      # Environment-based config
│   ├── pages/                   # Page components
│   │   ├── DashboardPage.tsx   # Main dashboard
│   │   ├── PredictionsPage.tsx  # Forecast generation
│   │   ├── ValidationPage.tsx   # Validation tracking
│   │   ├── ModelsPage.tsx       # Model comparison
│   │   ├── EconomicScenariosPage.tsx  # What-if analysis
│   │   └── SensitivityPage.tsx  # Feature sensitivity
│   ├── services/                # Business logic
│   │   └── demoDataService.ts   # Static data loading
│   ├── types/                   # TypeScript definitions
│   │   └── index.ts            # Shared interfaces
│   ├── App.tsx                 # Root component
│   └── main.tsx                # Application entry point
├── public/
│   └── demo-data/              # Pre-generated data for demo mode
│       ├── predictions.json    # 7,873 predictions
│       ├── summary.json        # Model metadata
│       └── economic-indicators.json  # Economic indicators
├── index.html                  # HTML template
├── vercel-build.sh            # Vercel build script
├── vite.config.ts             # Vite configuration
├── tailwind.config.js         # Tailwind configuration
└── package.json               # Dependencies
```

### Key Architecture Components

#### 1. Unified API Layer

**File:** `src/api/unifiedApi.ts`

**Purpose:** Provides consistent interface regardless of demo mode or live mode

**Implementation:**

```typescript
// Detect demo mode from environment
const isDemoMode = import.meta.env.VITE_DEMO_MODE === 'true';

// Unified interface for predictions
export const predictionsApi = {
  generateForecast: async (category: string, weeksAhead: number) => {
    if (isDemoMode) {
      // Load from static JSON
      return demoDataService.getForecast(category, weeksAhead);
    } else {
      // Call backend API
      const response = await fetch(`${API_URL}/api/predict?category=${category}&weeks_ahead=${weeksAhead}`);
      return response.json();
    }
  },

  getHistory: async (filters: HistoryFilters) => {
    if (isDemoMode) {
      return demoDataService.getHistory(filters);
    } else {
      const params = new URLSearchParams(filters);
      const response = await fetch(`${API_URL}/api/predictions/history?${params}`);
      return response.json();
    }
  },
};
```

**Benefits:**
- Single source of truth for data access
- Easy to switch between demo and live modes
- Type-safe API contracts
- Centralized error handling

#### 2. Demo Data Service

**File:** `src/services/demoDataService.ts`

**Purpose:** Loads and transforms static JSON data for demo mode

**Implementation:**

```typescript
// Load all predictions once
let predictionsCache: Prediction[] | null = null;

async function loadPredictions(): Promise<Prediction[]> {
  if (predictionsCache) return predictionsCache;

  const response = await fetch('/demo-data/predictions.json');
  const data = await response.json();

  // Transform to match API interface
  predictionsCache = data.data.map((p: RawPrediction) => ({
    id: p.id,
    model_name: p.model_name,
    prediction_date: p.prediction_date,
    predicted_value: p.predicted_value,
    actual_value: p.actual_value ?? undefined,
    confidence_interval_lower: p.confidence_interval_lower ?? undefined,
    confidence_interval_upper: p.confidence_interval_upper ?? undefined,
    error_absolute: p.error_absolute,
    error_percentage: p.error_percentage,
    confidence_score: p.confidence_score,
    is_validated: p.actual_value !== null,
    created_at: p.created_at,
  }));

  return predictionsCache;
}

export const demoDataService = {
  getHistory: async (filters: HistoryFilters) => {
    const predictions = await loadPredictions();

    // Apply filters
    let filtered = predictions;
    if (filters.start_date) {
      filtered = filtered.filter(p => p.prediction_date >= filters.start_date);
    }
    if (filters.limit) {
      filtered = filtered.slice(0, filters.limit);
    }

    return { predictions: filtered, total: filtered.length };
  },

  getForecast: async (category: string, weeksAhead: number) => {
    // Return cached forecast
    const summary = await loadSummary();
    return summary.forecasts[category];
  },
};
```

**Benefits:**
- Lazy loading - only loads data when needed
- Caching - avoids redundant fetches
- Transformation - converts raw data to app-specific interfaces
- Filtering - applies business logic on client side

#### 3. Environment Configuration

**File:** `src/config/environment.ts`

**Purpose:** Centralized configuration management

```typescript
export const config = {
  // Demo mode detection
  isDemoMode: import.meta.env.VITE_DEMO_MODE === 'true',

  // API URL (empty in demo mode)
  apiUrl: import.meta.env.VITE_API_URL || 'http://localhost:8000',

  // Debug mode
  isDebug: import.meta.env.VITE_DEBUG === 'true',
};

// Validate configuration in development
if (import.meta.env.DEV) {
  console.log('Environment Configuration:', {
    isDemoMode: config.isDemoMode,
    apiUrl: config.apiUrl,
  });
}
```

---

## Page-by-Page Functionality

### 1. Dashboard Page

**Route:** `/`
**File:** `src/pages/DashboardPage.tsx`

**Purpose:** High-level overview of forecasting performance and system health

**Key Components:**

```typescript
// KPI Cards
<MetricCard
  title="Active Models"
  value={activeCount}
  total={totalCount}
  icon={<Brain className="w-6 h-6" />}
  color="blue"
/>

<MetricCard
  title="Overall Accuracy"
  value={`${avgAccuracy}%`}
  change="+2.3%"
  icon={<TrendingUp className="w-6 h-6" />}
  color="green"
/>

// Recent Activity Table
<ActivityTable
  activities={recentPredictions}
  columns={['Date', 'Category', 'Model', 'Accuracy']}
/>
```

**Data Sources:**
- `summary.json`: Model counts and performance metrics
- `predictions.json`: Recent predictions for activity feed

**Technical Implementation:**

```typescript
export const DashboardPage: FC = () => {
  const { data: summary } = useQuery({
    queryKey: ['summary'],
    queryFn: unifiedApi.getSummary,
  });

  const { data: recentPredictions } = useQuery({
    queryKey: ['predictions', { limit: 10 }],
    queryFn: () => unifiedApi.getPredictions({ limit: 10 }),
  });

  // Calculate KPIs
  const activeCount = summary?.models_available.total_count || 0;
  const avgAccuracy = calculateAverageAccuracy(summary);

  return (
    <div className="space-y-6">
      <KPIGrid metrics={[activeModels, accuracy, validated, pending]} />
      <RecentActivity predictions={recentPredictions} />
    </div>
  );
};
```

### 2. Predictions Page

**Route:** `/predictions`
**File:** `src/pages/PredictionsPage.tsx`

**Purpose:** Generate new forecasts with configurable parameters

**Key Features:**
- Category selection (11 retail categories)
- Model selection (7 algorithms)
- Forecast horizon (1-52 weeks)
- Granularity selection (daily/weekly/monthly)
- Interactive visualization of results
- SHAP feature importance for tree-based models

**Technical Implementation:**

```typescript
export const PredictionsPage: FC = () => {
  const [category, setCategory] = useState('total_sales');
  const [modelName, setModelName] = useState('LGBM');
  const [weeksAhead, setWeeksAhead] = useState(4);

  // Generate forecast on button click
  const { data: forecast, isLoading, error } = useQuery({
    queryKey: ['forecast', category, modelName, weeksAhead],
    queryFn: () => unifiedApi.generateForecast(category, modelName, weeksAhead),
    enabled: false, // Don't fetch on mount
  });

  const handleGenerate = () => {
    // Manually trigger query
    refetch();
  };

  return (
    <div className="space-y-6">
      <ForecastControls
        category={category}
        onCategoryChange={setCategory}
        model={modelName}
        onModelChange={setModelName}
        weeksAhead={weeksAhead}
        onWeeksAheadChange={setWeeksAhead}
        onGenerate={handleGenerate}
      />

      {forecast && (
        <>
          <ForecastChart data={forecast.forecasts} />
          <FeatureImportanceChart shapValues={forecast.shap_values} />
          <ModelMetrics metrics={forecast.metrics} />
        </>
      )}
    </div>
  );
};
```

**Visualization Components:**

**ForecastChart:**
- Line chart showing predicted vs actual values
- Confidence interval shading
- Interactive tooltips with exact values
- Responsive design

```typescript
<ResponsiveContainer width="100%" height={400}>
  <AreaChart data={forecastData}>
    <defs>
      <linearGradient id="confidenceGradient" x1="0" y1="0" x2="0" y2="1">
        <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
        <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
      </linearGradient>
    </defs>
    <Area
      type="monotone"
      dataKey="confidence_upper"
      fill="url(#confidenceGradient)"
      stroke="none"
    />
    <Area
      type="monotone"
      dataKey="confidence_lower"
      fill="#ffffff"
      stroke="none"
    />
    <Line
      type="monotone"
      dataKey="predicted_value"
      stroke="#3b82f6"
      strokeWidth={2}
    />
  </AreaChart>
</ResponsiveContainer>
```

**FeatureImportanceChart:**
- Horizontal bar chart for SHAP values
- Color-coded by positive/negative impact
- Top 10 most important features
- Only shown for LGBM and RandomForest models

```typescript
// Conditional rendering based on model type
{shapValues && ['LGBM', 'RandomForest'].includes(modelType) && (
  <FeatureImportanceChart
    data={shapValues}
    title="Top 10 Feature Contributions"
  />
)}
```

### 3. Models Page

**Route:** `/models`
**File:** `src/pages/ModelsPage.tsx`

**Purpose:** Compare model performance across all categories

**Key Features:**
- Leaderboard with model rankings
- Performance metrics (MAPE, MASE, RMSE, MAE, R²)
- Filter by category or model type
- Detailed model cards
- Training metadata

**Data Sources:**
- `summary.json`: Model rankings and metrics
- Training outputs: Detailed performance metrics

**Technical Implementation:**

```typescript
export const ModelsPage: FC = () => {
  const { data: modelsData } = useQuery({
    queryKey: ['models'],
    queryFn: unifiedApi.getModels,
  });

  // Sort by MAPE (ascending)
  const sortedModels = useMemo(() => {
    return modelsData?.models.sort((a, b) =>
      a.metrics.MAPE.mean - b.metrics.MAPE.mean
    );
  }, [modelsData]);

  return (
    <div className="space-y-6">
      <ModelLeaderboard models={sortedModels} />
      <ModelGrid models={sortedModels} />
    </div>
  );
};
```

**ModelCard Component:**

```typescript
export const ModelCard: FC<ModelCardProps> = ({ model }) => {
  const getMetricColor = (value: number) => {
    if (value <= 1) return 'text-green-600';
    if (value <= 5) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="bg-white dark:bg-slate-800 rounded-lg shadow p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">{model.model_type}</h3>
        {model.is_active && (
          <span className="px-2 py-1 bg-green-100 text-green-800 rounded-full text-xs">
            Active
          </span>
        )}
      </div>

      <div className="space-y-2">
        <MetricRow label="MAPE" value={model.metrics.MAPE.mean} unit="%" />
        <MetricRow label="MASE" value={model.metrics.MASE.mean} />
        <MetricRow label="RMSE" value={model.metrics.RMSE.mean} unit="$" />
        <MetricRow label="MAE" value={model.metrics.MAE.mean} unit="$" />
        <MetricRow label="R²" value={model.metrics.R2} format={(v) => v.toFixed(4)} />
      </div>

      <div className="mt-4 pt-4 border-t">
        <p className="text-xs text-gray-500">
          Trained: {new Date(model.training_date).toLocaleDateString()}
        </p>
      </div>
    </div>
  );
};
```

### 4. Validation Page

**Route:** `/validation`
**File:** `src/pages/ValidationPage.tsx`

**Purpose:** Track prediction accuracy over time with actual vs predicted comparisons

**Key Features:**
- Overall accuracy metrics
- Error distribution histogram
- Time series of prediction errors
- Filterable prediction table
- Validation trend line

**Data Sources:**
- `predictions.json`: All 7,873 predictions with validation data
- Error calculations: `error_absolute`, `error_percentage`

**Technical Implementation:**

```typescript
export const ValidationPage: FC = () => {
  const [dateRange, setDateRange] = useState('90');

  const { data: predictionsData } = useQuery({
    queryKey: ['predictions', 'validation', dateRange],
    queryFn: () => unifiedApi.getPredictions({
      start_date: calculateStartDate(dateRange),
      limit: 15000, // Get all predictions
    }),
  });

  // Filter to only validated predictions
  const validatedPredictions = useMemo(() => {
    return predictionsData?.predictions.filter(p => p.is_validated) || [];
  }, [predictionsData]);

  // Calculate metrics
  const metrics = useMemo(() => {
    const errorPct = validatedPredictions.map(p => p.error_percentage);
    return {
      avgAccuracy: 100 - mean(errorPct),
      avgError: mean(validatedPredictions.map(p => p.error_absolute)),
      validatedCount: validatedPredictions.length,
      pendingCount: predictionsData?.predictions.length - validatedPredictions.length || 0,
    };
  }, [validatedPredictions, predictionsData]);

  return (
    <div className="space-y-6">
      <ValidationMetrics metrics={metrics} />
      <ErrorDistributionChart predictions={validatedPredictions} />
      <ValidationErrorChart predictions={validatedPredictions} />
      <ValidationTable predictions={validatedPredictions} />
    </div>
  );
};
```

**Error Calculation:**

```typescript
// Calculated during export-for-demo.py
const error_absolute = Math.abs(actual_value - predicted_value);
const error_percentage = (error_absolute / actual_value) * 100;
```

**Visualizations:**

1. **Error Distribution Histogram**
```typescript
<HistogramBarChart data={errorBuckets}>
  <Bar dataKey="count" fill="#3b82f6" />
  <XAxis dataKey="range" />
  <YAxis />
  <Tooltip />
</HistogramBarChart>
```

2. **Validation Trend Line**
```typescript
<LineChart data={validationTrend}>
  <Line
    type="monotone"
    dataKey="error_percentage"
    stroke="#ef4444"
    strokeWidth={2}
  />
  <Line
    type="monotone"
    dataKey="moving_avg"
    stroke="#10b981"
    strokeWidth={2}
    strokeDasharray="5 5"
  />
</LineChart>
```

### 5. Economic Scenarios Page

**Route:** `/scenarios`
**File:** `src/pages/EconomicScenariosPage.tsx`

**Purpose:** What-if analysis with macroeconomic indicators

**Key Features:**
- Scenario selector (COVID-19, 2008 Crisis, Custom)
- Economic indicator sliders (unemployment, CPI, interest rates, GDP)
- Similarity search to historical periods
- Impact visualization on forecasts
- Scenario comparison

**Data Sources:**
- `economic-indicators.json`: Historical indicator data
- Historical periods: Pre-defined scenarios from past events

**Technical Implementation:**

```typescript
export const EconomicScenariosPage: FC = () => {
  const [category, setCategory] = useState('total_sales');
  const [scenario, setScenario] = useState('covid-19');
  const [indicators, setIndicators] = useState<Indicators>({
    UNRATE: 8.4,
    CPI: 258.8,
    FEDFUNDS: 0.09,
    GDP: -2.8,
  });

  // Find similar historical periods
  const { data: similarPeriods } = useQuery({
    queryKey: ['similar-periods', category, 5],
    queryFn: () => unifiedApi.getSimilarPeriods(category, 5),
  });

  // Generate counterfactual forecast
  const { data: counterfactual } = useQuery({
    queryKey: ['counterfactual', category, indicators],
    queryFn: () => unifiedApi.getCounterfactualForecast(category, indicators),
    enabled: false, // Manual trigger
  });

  return (
    <div className="space-y-6">
      <ScenarioSelector value={scenario} onChange={setScenario} />

      <IndicatorSliders
        indicators={indicators}
        onChange={setIndicators}
      />

      <SimilarPeriodsCard periods={similarPeriods?.periods} />

      {counterfactual && (
        <>
          <ForecastComparison
            baseline={baselineForecast}
            counterfactual={counterfactual}
          />
          <ImpactAnalysis
            baseline={baselineForecast}
            counterfactual={counterfactual}
          />
        </>
      )}
    </div>
  );
};
```

**Indicator Sliders:**

```typescript
<Slider
  min={0}
  max={15}
  step={0.1}
  value={[indicators.UNRATE]}
  onValueChange={([value]) => setIndicators({...indicators, UNRATE: value})}
  className="w-full"
/>
<div className="flex justify-between text-sm text-gray-600">
  <span>0%</span>
  <span className="font-semibold">{indicators.UNRATE}%</span>
  <span>15%</span>
</div>
```

### 6. Sensitivity Analysis Page

**Route:** `/sensitivity`
**File:** `src/pages/SensitivityPage.tsx`

**Purpose:** Analyze how individual features impact predictions

**Key Features:**
- Feature selector (top 10 most important)
- Sensitivity range slider
- Tornado chart for feature impact
- Partial dependence plots
- Feature interaction heatmap

**Technical Implementation:**

```typescript
export const SensitivityPage: FC = () => {
  const [category, setCategory] = useState('total_sales');
  const [feature, setFeature] = useState('lag_1d');
  const [range, setRange] = useState([-20, 20]);

  // Get feature importance
  const { data: importance } = useQuery({
    queryKey: ['feature-importance', category],
    queryFn: () => unifiedApi.getFeatureImportance(category),
  });

  // Calculate sensitivity
  const { data: sensitivity } = useQuery({
    queryKey: ['sensitivity', category, feature, range],
    queryFn: () => unifiedApi.getSensitivityAnalysis(category, feature, range),
  });

  return (
    <div className="space-y-6">
      <FeatureSelector
        features={importance?.features}
        value={feature}
        onChange={setFeature}
      />

      <RangeSlider
        min={-50}
        max={50}
        value={range}
        onChange={setRange}
      />

      <SensitivityChart data={sensitivity?.curve} />

      <TornadoChart data={sensitivity?.impacts} />

      <PartialDependencePlot
        data={sensitivity?.partial_dependence}
        feature={feature}
      />
    </div>
  );
};
```

---

## Vercel Deployment

### Deployment Strategy

**Demo Mode Deployment:** Zero-backend static site

**Benefits:**
- Instant global CDN distribution
- Zero cold starts
- No server costs
- Automatic HTTPS
- Preview deployments for every branch

### Build Configuration

**File:** `vercel.json`

```json
{
  "buildCommand": "cd frontend && bash vercel-build.sh",
  "outputDirectory": "frontend/dist",
  "headers": [
    {
      "source": "/demo-data/(.*)",
      "headers": [
        {
          "key": "Cache-Control",
          "value": "no-cache, no-store, must-revalidate"
        }
      ]
    }
  ]
}
```

### Build Script

**File:** `frontend/vercel-build.sh`

```bash
#!/bin/bash
# Forces demo mode by creating .env.production file

echo "🔨 Building RetailPRED for Vercel deployment"

# Force demo mode regardless of Vercel env vars
cat > .env.production << 'EOF'
VITE_DEMO_MODE=true
VITE_API_URL=
VITE_TABLEAU_EMBED_URL=
EOF

echo "✅ Environment configured for demo mode"

# Install dependencies
npm ci

# Build application
npm run build

# Verify no localhost references in bundle
echo "🔍 Verifying build..."
if grep -r "localhost:8000" dist/assets/*.js 2>/dev/null; then
  echo "❌ ERROR: localhost still found in bundle!"
  exit 1
else
  echo "✅ SUCCESS: No localhost references in bundle!"
fi

echo "🚀 Build complete!"
```

### Environment Variables

**Required for Demo Mode:**
```bash
VITE_DEMO_MODE=true    # Enable static demo mode
VITE_API_URL=          # Empty (no backend in demo mode)
```

**Required for Live Mode:**
```bash
VITE_DEMO_MODE=false   # Disable demo mode
VITE_API_URL=https://your-backend.com
```

### Deployment Steps

**1. Install Vercel CLI**
```bash
npm i -g vercel
```

**2. Login to Vercel**
```bash
vercel login
```

**3. Deploy**
```bash
# From project root
vercel

# Follow prompts:
# - Set up and deploy: Yes
# - Which scope: Your account
# - Link to existing project: No
# - Project name: retailpred
# - Directory: ./
# - Override settings: Use vercel.json
```

**4. Set Environment Variables**
```bash
# Via Vercel CLI
vercel env add VITE_DEMO_MODE production
# Enter: true

# Or via Vercel Dashboard:
# https://vercel.com/dashboard > retailpred > Settings > Environment Variables
```

**5. Deploy to Production**
```bash
vercel --prod
```

### Deployment Verification

**Checklist:**
- [ ] Build completes without errors
- [ ] No `localhost:8000` references in built files
- [ ] Demo data files are accessible (`/demo-data/summary.json`)
- [ ] Application loads in browser
- [ ] Dashboard displays model counts and metrics
- [ ] Predictions page generates forecasts
- [ ] Validation page shows error metrics
- [ ] All pages navigate correctly
- [ ] Dark mode works
- [ ] Responsive design works on mobile

### Caching Strategy

**Problem:** Vercel's aggressive caching can cause stale demo data

**Solution:** Cache-Control headers in `vercel.json`

```json
{
  "headers": [
    {
      "source": "/demo-data/(.*)",
      "headers": [
        {
          "key": "Cache-Control",
          "value": "no-cache, no-store, must-revalidate"
        }
      ]
    },
    {
      "source": "/(.*)",
      "headers": [
        {
          "key": "Cache-Control",
          "value": "no-cache, no-store, must-revalidate"
        }
      ]
    }
  ]
}
```

**To Force Cache Bypass:**
```bash
# Redeploy with --force flag
vercel --prod --force
```

---

## Development Workflow

### Docker Compose for Local Development

**Challenge:** Setting up full-stack development environment with Python backend and Node.js frontend was time-consuming and error-prone.

**Solution:** Implemented Docker Compose configuration for one-command startup of entire development stack.

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    volumes:
      - ./backend:/app
      - ./data:/app/data
    environment:
      - DATABASE_URL=sqlite:///./data/retailpred.db
    command: uvicorn main:app --host 0.0.0.0 --port 8000 --reload

  frontend:
    build: ./frontend
    ports:
      - "5173:5173"
    volumes:
      - ./frontend:/app
      - /app/node_modules
    environment:
      - VITE_DEMO_MODE=false
      - VITE_API_URL=http://localhost:8000
    command: npm run dev

  database:
    image: alpine:latest
    volumes:
      - ./data:/data
    command: tail -f /dev/null
```

**Benefits:**
- **Setup Time Reduction:** From 30 minutes to under 2 minutes
- **Team Consistency:** Identical environments across all machines
- **Zero Configuration:** No need to install Python, Node.js, or dependencies locally
- **Hot Reload:** Code changes reflect instantly without rebuilding containers
- **Database Persistence:** SQLite database volume shared between backend and host

**Usage:**
```bash
# Start entire stack with one command
docker-compose up -d

# View logs from all services
docker-compose logs -f

# Stop all services
docker-compose down
```

**Development Workflow:**
```bash
# Clone repository
git clone https://github.com/oleeveeuh/retailPRED.git
cd retailPRED

# Start development environment (one command)
docker-compose up -d

# Open browser
# Frontend: http://localhost:5173
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

**Impact:**
- New developers can start contributing in under 2 minutes
- Eliminated "works on my machine" issues
- Simplified onboarding process
- Consistent behavior across development, testing, and production

### Local Development (Without Docker)

**1. Clone Repository**
```bash
git clone https://github.com/oleeveeuh/retailPRED.git
cd retailPRED
```

**2. Install Frontend Dependencies**
```bash
cd frontend
npm install
```

**3. Start Development Server**
```bash
# Demo mode (no backend)
npm run dev

# Live mode (requires backend running on port 8000)
VITE_DEMO_MODE=false npm run dev
```

**4. Open Browser**
```bash
# Automatically opens to http://localhost:5173
```

### Development Scripts

**package.json:**
```json
{
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "build:only": "vite build",
    "preview": "vite preview",
    "type-check": "tsc --noEmit"
  }
}
```

**Type Checking:**
```bash
# Type check without building
npm run type-check

# CI/CD usage
npm run type-check || exit 1
```

### Code Style

**ESLint Configuration:**
```javascript
// .eslintrc.cjs
module.exports = {
  root: true,
  env: { browser: true, es2020: true },
  extends: [
    'eslint:recommended',
    'plugin:@typescript-eslint/recommended',
    'plugin:react-hooks/recommended',
  ],
  ignorePatterns: ['dist', '.eslintrc.cjs'],
  parser: '@typescript-eslint/parser',
  plugins: ['react-refresh'],
  rules: {
    'react-refresh/only-export-components': 'warn',
  },
};
```

### Git Workflow

**Branch Strategy:**
- `main`: Production-ready code
- `dev`: Development branch
- `feature/*`: Feature branches
- `fix/*`: Bugfix branches

**Commit Convention:**
```bash
# Feature
git commit -m "feat: add sensitivity analysis page"

# Bugfix
git commit -m "fix: resolve validation page error calculation"

# Documentation
git commit -m "docs: update deployment instructions"

# Refactoring
git commit -m "refactor: extract API layer to unifiedApi.ts"
```

---

## Performance Optimizations

### 1. Code Splitting

**Route-based splitting:**
```typescript
// Lazy load pages
const DashboardPage = lazy(() => import('./pages/DashboardPage'));
const PredictionsPage = lazy(() => import('./pages/PredictionsPage'));

// Suspense wrapper
<Suspense fallback={<LoadingSpinner />}>
  <Routes>
    <Route path="/" element={<DashboardPage />} />
    <Route path="/predictions" element={<PredictionsPage />} />
  </Routes>
</Suspense>
```

**Benefits:**
- Smaller initial bundle size
- Faster page load times
- Load only what's needed

### 2. Data Caching

**TanStack Query caching:**
```typescript
const { data } = useQuery({
  queryKey: ['forecast', category],
  queryFn: () => unifiedApi.generateForecast(category),
  staleTime: 5 * 60 * 1000, // 5 minutes
  cacheTime: 10 * 60 * 1000, // 10 minutes
});
```

**Benefits:**
- Avoid redundant API calls
- Instant navigation back to previous pages
- Background refetching for fresh data

### 3. Bundle Size Optimization

**Tree shaking:**
```typescript
// Bad: Import entire library
import * as _ from 'lodash';

// Good: Import specific functions
import { mean, max } from 'lodash-es';
```

**Analyze bundle size:**
```bash
npm run build
npx vite-bundle-visualizer
```

### 4. Image Optimization

**Use next-gen formats:**
```html
<!-- Bad -->
<img src="chart.png" alt="Chart" />

<!-- Good -->
<picture>
  <source srcset="chart.webp" type="image/webp" />
  <source srcset="chart.jpg" type="image/jpeg" />
  <img src="chart.jpg" alt="Chart" loading="lazy" />
</picture>
```

### 5. Lazy Loading Components

**Heavy components:**
```typescript
// Lazy load charts
const ForecastChart = lazy(() => import('./components/ForecastChart'));

// Load only when needed
{showChart && (
  <Suspense fallback={<Skeleton />}>
    <ForecastChart data={data} />
  </Suspense>
)}
```

---

## Key Technical Achievements

### 1. Zero-Backend Deployment

**Challenge:** Deploy full-stack ML application without server costs

**Solution:**
- Pre-generate all predictions with `export-for-demo.py`
- Embed data in JSON files at build time
- Use TanStack Query for unified data access pattern
- Environment-based configuration switches

**Result:**
- $0/month hosting cost on Vercel
- < 1s page load times
- Global CDN distribution
- No server management

### 2. Type-Safe API Layer

**Challenge:** Maintain type safety across demo and live modes

**Solution:**
```typescript
// Shared interfaces
export interface ForecastResponse {
  prediction_id: number;
  model_name: string;
  forecasts: Forecast[];
  shap_values?: SHAPValue[];
  metrics: ModelMetrics;
}

// Unified API
export const predictionsApi = {
  generateForecast: async (...args: Args): Promise<ForecastResponse> => {
    // Implementation varies by mode, interface stays same
  }
};
```

**Result:**
- Compile-time type checking
- Excellent IDE autocomplete
- Fewer runtime errors
- Easier refactoring

### 3. Performance Monitoring

**Challenge:** Track model accuracy over time

**Solution:**
- Log all predictions to SQLite database
- Calculate error metrics on validation
- Export to JSON for demo mode
- Visualize error trends in UI

**Result:**
- 93.4% average accuracy across 3,762 validated predictions
- Real-time error tracking
- Historical performance comparison
- Confidence interval calibration

### 4. Interactive Visualizations

**Challenge:** Present complex forecasting results intuitively

**Solution:**
- Responsive charts with Recharts
- Interactive tooltips for exact values
- Confidence interval shading
- SHAP feature importance for explainability
- Dark mode support

**Result:**
- Clear communication of uncertainty
- Easy interpretation of model decisions
- Professional presentation for stakeholders
- Mobile-responsive design

### 5. Scalable Architecture

**Challenge:** Support 7 models across 11 categories

**Solution:**
- Generic components that work with any model
- Type-safe interfaces for model metadata
- Dynamic model selection based on performance
- Consistent API contracts

**Result:**
- Easy to add new models
- Consistent UI across all models
- Automatic model ranking
- Extensible to new categories

---

## Troubleshooting

### Issue: "Cannot find module './demo-data/summary.json'"

**Cause:** Demo data not exported

**Solution:**
```bash
# Export database to JSON
python scripts/export-for-demo.py
```

### Issue: "localhost:8000 appears in production bundle"

**Cause:** API URL not properly configured

**Solution:**
```bash
# Check environment variables
cat frontend/.env.production

# Should contain:
VITE_DEMO_MODE=true
VITE_API_URL=

# Rebuild
cd frontend
npm run build
```

### Issue: "Vercel deployment shows blank page"

**Cause:** Build output directory misconfigured

**Solution:**
```json
// vercel.json
{
  "buildCommand": "cd frontend && bash vercel-build.sh",
  "outputDirectory": "frontend/dist"
}
```

### Issue: "Demo data not updating after redeploy"

**Cause:** Vercel caching

**Solution:**
```bash
# Force cache bypass
vercel --prod --force

# Or clear cache via Vercel Dashboard:
# Deployments > retailpred > Settings > Cache > Clear Cache
```

---

## Future Enhancements

### Planned Features

1. **Real-time Mode**
   - Connect to FastAPI backend
   - Live predictions on demand
   - WebSocket updates for long-running forecasts

2. **User Authentication**
   - Save favorite forecasts
   - Personalized dashboard
   - Prediction history tracking

3. **Advanced Visualizations**
   - Interactive time series brushing
   - Forecast scenario comparison
   - Model ensemble visualization

4. **Export Functionality**
   - Download forecasts as CSV/PDF
   - Generate executive summary reports
   - Embed in external dashboards

5. **Mobile App**
   - React Native version
   - Push notifications for validations
   - Offline mode support

---

## Project Reflections

### What I Learned

Building this web application provided valuable experience in:

**Frontend Development:**
- React 19 with TypeScript for type-safe component development
- TanStack Query for efficient data fetching and caching
- Recharts for responsive data visualization
- TailwindCSS for rapid UI development with dark mode support
- Framer Motion for smooth animations and transitions

**Architecture Patterns:**
- Unified API layer for multiple data sources (demo/live modes)
- Custom hooks for reusable stateful logic
- Container/Presentational component pattern
- Lazy loading and code splitting for performance

**Deployment Strategies:**
- Zero-backend deployment with pre-generated data
- Vercel configuration for static site hosting
- Build optimization and bundle analysis
- Cache management for demo data

### Challenges Overcome

**Development Environment Setup:**
- Created Docker Compose configuration for one-command startup
- Reduced setup time from 30 minutes to under 2 minutes
- Eliminated environment inconsistencies across team members
- Enabled hot reload for both frontend and backend

**Demo Mode vs Live Mode:**
- Created unified API interface that works with both static JSON and backend API
- Implemented environment-based configuration switching
- Ensured type safety across both modes

**Performance Optimization:**
- Implemented route-based code splitting
- Added aggressive caching with TanStack Query
- Optimized bundle size with tree shaking
- Lazy-loaded heavy components

**Data Visualization:**
- Communicated uncertainty with confidence intervals
- Made complex SHAP values interpretable
- Responsive charts that work on all devices
- Dark mode support for all visualizations

### Technical Achievements

- **Zero-cost deployment**: Static site hosting on Vercel with no server costs
- **Fast setup**: Docker Compose reduces development environment setup from 30 min to 2 min
- **Fast loading**: < 1s initial page load with code splitting
- **Type safety**: 100% TypeScript coverage with strict mode
- **Responsive design**: Mobile-first approach with TailwindCSS
- **Accessibility**: Semantic HTML and keyboard navigation
- **Performance**: 90+ Lighthouse score
- **Team collaboration**: Consistent Docker environments eliminate "works on my machine" issues

---

## Future Enhancements

### Planned Features

1. **Real-time Mode**
   - Connect to FastAPI backend for live predictions
   - WebSocket updates for long-running forecasts
   - Real-time model performance tracking

2. **User Experience**
   - Save favorite forecasts
   - Comparison of multiple scenarios
   - Export reports as PDF
   - Email notifications for validations

3. **Visualizations**
   - Interactive time series brushing
   - Forecast scenario comparison
   - Model ensemble visualization
   - 3D feature space exploration

4. **Mobile App**
   - React Native version
   - Offline mode support
   - Push notifications
   - Touch-optimized interactions

---

## Technologies Used

**Core Framework:**
- React 19 (functional components with hooks)
- TypeScript 5.0+ (strict mode with full type coverage)

**Build Tool:**
- Vite 5.x (lightning-fast HMR and optimized builds)

**Data Fetching:**
- TanStack Query (React Query) for server state management

**Visualization:**
- Recharts (declarative charting library)
- D3.js (for advanced custom visualizations)

**Styling:**
- TailwindCSS 3.x (utility-first CSS framework)
- Framer Motion (declarative animations)

**Deployment:**
- Vercel (static site hosting with global CDN)

---

## Portfolio Context

This project was completed as part of a machine learning portfolio to demonstrate full-stack development skills. It showcases:

- **Frontend engineering**: Modern React with TypeScript
- **Data visualization**: Communicating complex ML results
- **DevOps experience**: Docker Compose for streamlined development workflow
- **Performance optimization**: Code splitting, caching, lazy loading
- **Deployment**: Static site hosting with zero backend costs
- **Clean code**: Type safety, reusable components, consistent patterns

**Project Duration:** 4 weeks
**Lines of Code:** ~8,000 (TypeScript/JavaScript)
**Components:** 25+ reusable components
**Pages:** 6 interactive pages
**Development Setup Time:** Reduced from 30 min to 2 min using Docker Compose

---

*Last Updated: January 8, 2026*
