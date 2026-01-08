# Demo Mode Infrastructure - Summary

## ✅ What Was Created

### 1. Demo Data Service
**File**: `frontend/src/services/demoDataService.ts`

Service class that loads data from static JSON files and mimics the API structure:

**Features**:
- ✅ Loads JSON files from `/demo-data/` directory
- ✅ Caches loaded data in memory (Map-based)
- ✅ Adds 300ms delay to simulate API calls
- ✅ Mimics exact API response structure
- ✅ Supports filtering (model_name, dates, limit)
- ✅ Transforms SHAP values to API format

**Methods**:
- `getPredictions(filters?)` - Load predictions.json
- `getEconomicIndicators()` - Load economic-indicators.json
- `getSummary()` - Load summary.json
- `getSHAPValues(predictionId)` - Get SHAP for specific prediction
- `clearCache()` - Clear cached data (for testing)

**TypeScript Types**:
- `DemoPrediction` - Prediction from JSON
- `DemoEconomicIndicator` - Economic indicator from JSON
- `DemoDataResponse<T>` - Wrapper with metadata
- `DemoSummary` - Summary statistics

---

### 2. Environment Configuration
**File**: `frontend/src/config/environment.ts`

Centralized configuration for demo mode:

```typescript
export const config = {
  isDemoMode: import.meta.env.VITE_DEMO_MODE === 'true',
  apiUrl: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  isDebug: import.meta.env.VITE_DEBUG === 'true',
};
```

**Features**:
- ✅ Checks `VITE_DEMO_MODE` environment variable
- ✅ Falls back to localhost API if not set
- ✅ Logs configuration in development mode
- ✅ Debug mode for troubleshooting

---

### 3. Demo Banner Component
**File**: `frontend/src/components/DemoBanner.tsx`

React component that shows when demo mode is active:

**Features**:
- ✅ Only displays when `VITE_DEMO_MODE=true`
- ✅ Shows "📊 Demo Mode - Using real predictions from production database"
- ✅ Includes "View on GitHub" link
- ✅ Styled with Tailwind CSS (blue gradient background)
- ✅ Responsive design (flex wrap)
- ✅ Clean, professional appearance

---

## 📋 Current API Client Structure

**File**: `frontend/src/api/client.ts`

### Current State
Your existing API client is well-structured with:

**API Objects**:
- `predictionsApi` - predict, getHistory, validate, autoValidate, getSHAPExplanation
- `dataApi` - refresh data
- `modelsApi` - getAll, train
- `categoriesApi` - list, predict, getModels
- `systemApi` - healthCheck

**Types Defined**:
- `PredictionRequest`, `PredictionResponse`
- `PredictionHistoryItem`, `PredictionHistoryResponse`
- `SHAPValue`, `SHAPExplanationResponse`
- `ModelInfo`, `ModelMetrics`
- And 15+ more types

**Current Base URL**:
```typescript
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
```

---

## 🔄 Integration Pattern (Next Steps)

To integrate demo mode, you'll need to:

### Option A: Wrapper Function (Recommended)
Create a unified API interface that switches between demo and real:

```typescript
// frontend/src/api/unifiedApi.ts
import { config } from '../config/environment';
import { demoDataService } from '../services/demoDataService';
import { predictionsApi, dataApi, modelsApi, categoriesApi, systemApi } from './client';

export const api = config.isDemoMode ? {
  // Demo mode implementations
  getHistory: demoDataService.getPredictions,
  getSHAPExplanation: demoDataService.getSHAPValues,
  getEconomicIndicators: demoDataService.getEconomicIndicators,
  getSummary: demoDataService.getSummary,
  // Add more mappings...
} : {
  // Real API implementations
  ...predictionsApi,
  ...dataApi,
  ...modelsApi,
  ...categoriesApi,
  ...systemApi,
};
```

### Option B: Component-Level Switch
Let components check `config.isDemoMode` and call appropriate service.

---

## 📂 Demo Data Files

### predictions.json (414 KB)
```json
{
  "data": [
    {
      "id": 40089,
      "model_name": "electronics_and_appliances_RandomForest_model",
      "prediction_date": "2026-12-31",
      "predicted_value": 62068.77,
      "actual_value": null,
      "confidence_interval_lower": 60827.4,
      "confidence_interval_upper": 63310.15,
      "shap_values": { /* 242 features */ },
      "features": null,
      "created_at": "2026-01-07 21:46:20"
    },
    // ... 99 more predictions
  ],
  "metadata": {
    "export_timestamp": "2026-01-07T17:51:01.257719",
    "row_count": 100,
    "total_predictions_in_db": 7873,
    "date_range": { "start": "2025-01-03", "end": "2026-12-31" },
    "models": [/* 77 model names */],
    "note": "Most recent 100 predictions"
  }
}
```

### economic-indicators.json (110 KB)
```json
{
  "data": [
    {
      "date": "2024-08-25",
      "cpi": 299.46,
      "interest_rates": 5.65,
      "unemployment": 3.43,
      "consumer_sentiment": 65.03,
      "money_supply": 19926.36,
      "industrial_production": 103.8
    },
    // ... 499 more indicators
  ],
  "metadata": {
    "export_timestamp": "2026-01-07T17:51:01.278785",
    "row_count": 500,
    "note": "Sample economic indicators for demo"
  }
}
```

### summary.json (883 B)
```json
{
  "export_timestamp": "2026-01-07T17:51:01.378785",
  "database_path": "/Users/olivialiau/retailPRED/data/retailpred.db",
  "predictions": {
    "total_count": 7873,
    "by_year": { "2026": 4067, "2025": 3806 },
    "by_model_type": {
      "RandomForest": 1159,
      "LGBM": 1126,
      "AutoARIMA": 1122,
      "AutoETS": 1122,
      "SeasonalNaive": 1122,
      "PatchTST": 1111,
      "TimesNet": 1111
    },
    "shap_coverage": 27.78
  },
  "models_available": {
    "total_count": 77,
    "models": [/* 77 model names */]
  },
  "demo_data": {
    "predictions_included": 100,
    "economic_indicators_included": 500,
    "note": "Subset of data for static demo deployment"
  }
}
```

---

## 🎯 Next Steps

### 1. Add DemoBanner to Layout
Update `frontend/src/components/layout/Layout.tsx`:

```typescript
import { DemoBanner } from '../components/DemoBanner';

export const Layout: FC<LayoutProps> = ({ children }) => {
  return (
    <div className="flex min-h-screen">
      {/* Add DemoBanner at the top */}
      <DemoBanner />

      {/* Sidebar */}
      <Sidebar />

      {/* Rest of layout... */}
    </div>
  );
};
```

### 2. Create Unified API Interface
Create `frontend/src/api/unifiedApi.ts` (see Option A above)

### 3. Update Component Imports
Find components that import from `../api/client` and update to import from unified API.

### 4. Create Environment Files
See Prompt 4 (next step)

---

## 📊 Data Coverage

**Models Included**: 7 types × 11 categories = 77 models
- LGBM (1,126 predictions, 1,028 with SHAP)
- RandomForest (1,159 predictions, 1,159 with SHAP)
- AutoARIMA (1,122 predictions, 0 with SHAP)
- AutoETS (1,122 predictions, 0 with SHAP)
- SeasonalNaive (1,122 predictions, 0 with SHAP)
- PatchTST (1,111 predictions, 0 with SHAP)
- TimesNet (1,111 predictions, 0 with SHAP)

**Date Range**: 2025-01-03 to 2026-12-31

**Total Predictions**: 7,873

**SHAP Coverage**: 2,187 predictions (27.78%)

---

## ✅ Files Created

1. ✅ `frontend/src/services/demoDataService.ts` (336 lines)
2. ✅ `frontend/src/config/environment.ts` (26 lines)
3. ✅ `frontend/src/components/DemoBanner.tsx` (56 lines)
4. ✅ `DEMO_MODE_SUMMARY.md` (this file)

---

## 🔍 Testing Checklist

- [ ] Demo banner appears when `VITE_DEMO_MODE=true`
- [ ] Demo banner hidden when `VITE_DEMO_MODE=false`
- [ ] Predictions load from JSON in demo mode
- [ ] Filters work (model_name, dates)
- [ ] SHAP values display correctly
- [ ] Economic indicators load
- [ ] Summary statistics display
- [ ] No console errors
- [ ] Network tab shows JSON file loads (not API calls)
- [ ] Cache works (second load is faster)

---

## 🚀 Ready for Next Step

Proceed to **Prompt 4: Environment Configuration** to create:
- `frontend/.env.development`
- `frontend/.env.production`
- `frontend/.env.example`
- Update `.gitignore`

This will enable switching between demo and live modes!
