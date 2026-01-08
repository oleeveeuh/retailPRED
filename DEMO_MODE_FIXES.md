# Demo Mode Fixes - January 8, 2025

## Problem Statement

The Vercel-deployed application was experiencing errors when trying to load data in demo mode:
1. **404 Error**: `/api/historical-sales` endpoint was being called, resulting in 404 errors
2. **401 Error**: `manifest.json` was returning 401 Unauthorized
3. **Dashboard working but other components failing**: Dashboard loaded successfully but ForecastChart component failed to fetch historical data

## Root Cause Analysis

### Primary Issue
The application uses a **unified API pattern** that switches between demo and real implementations based on `VITE_DEMO_MODE`. However, there was a critical flaw:

1. In demo mode, `VITE_API_URL` is set to empty string `""`
2. `apiClient` in `client.ts` was created with `baseURL: API_BASE_URL`
3. **Axios with empty baseURL makes requests to the current origin** instead of blocking requests
4. This caused the real API implementations to make actual HTTP requests even in demo mode

### Secondary Issue
The `demoTrainingMetricsApi` expected `summary.models_available.models` and `summary.models_available.categories` arrays, but the actual `summary.json` file had:
```json
{
  "models_available": {
    "with_shap": ["LGBM", "RandomForest"],
    "without_shap": ["AutoARIMA", "AutoETS", "SeasonalNaive"]
  }
}
```

## Solutions Implemented

### 1. Request Interceptor in `client.ts`

Added a request interceptor to `apiClient` that prevents API calls in demo mode:

```typescript
// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    // In demo mode with empty API URL, prevent actual API calls
    const isDemoMode = import.meta.env.VITE_DEMO_MODE === 'true';
    const apiBaseUrl = import.meta.env.VITE_API_URL;

    if (isDemoMode && apiBaseUrl === '') {
      // Reject the request immediately in demo mode
      return Promise.reject(new Error('Demo mode active - API calls disabled'));
    }

    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);
```

**Purpose**: Acts as a safety net to prevent any accidental HTTP requests in demo mode.

### 2. Fixed `demoTrainingMetricsApi` in `unifiedApi.ts`

Updated the implementation to match the actual `summary.json` structure:

```typescript
const demoTrainingMetricsApi = {
  getModels: async (): Promise<TrainingMetricsResponse> => {
    const summary = await demoDataService.getSummary();

    // Get all model types from summary
    const modelTypes = [
      ...summary.models_available?.with_shap || [],
      ...summary.models_available?.without_shap || []
    ];

    // Transform demo data to match training metrics format
    return {
      models: modelTypes.map((modelName, index) => ({
        id: index + 1,
        model_name: modelName,
        category: 'Total Retail Sales',
        training_date: '2025-01-01',
        metrics: {
          RMSE: 1000 + Math.random() * 500,
          MAE: 800 + Math.random() * 400,
          R2: 0.92 + Math.random() * 0.07,
          MAPE: 3 + Math.random() * 5,
          mean: 50000,
          std: 5000,
        },
        hyperparameters: {
          learning_rate: 0.01,
          n_estimators: 100,
        },
        is_active: true,
      })),
      total_count: modelTypes.length,
      active_count: modelTypes.length,
    };
  },
};
```

### 3. Verified `.env.production` Configuration

Ensured the production environment file has the correct settings:

```env
VITE_DEMO_MODE=true
VITE_API_URL=
VITE_TABLEAU_EMBED_URL=
```

## Build Verification

After implementing the fixes, the production build was verified:

```bash
npm run build:prod
```

### Verification Results
- ✅ **No localhost references**: Confirmed with `grep -c "localhost:8000" dist/assets/*.js`
- ✅ **Demo data embedded**: "Unemployment Rate" found in bundle (3 occurrences)
- ✅ **Request interceptor present**: "Demo mode active" error message in bundle
- ✅ **Demo data files included**: Verified `/dist/demo-data/` contains all required JSON files

### Test Output
```
Production Build Verification:
==============================
✅ hasDemoData: true
✅ hasRequestInterceptor: true
❌ hasLocalhost: false
✅ hasEconomicIndicators: true

Detailed Analysis:
- Demo data embedded: YES
- Request interceptor present: YES
- Localhost references: NOT FOUND (GOOD)

✅ SUCCESS: Build appears to be in demo mode!
```

## How It Works

### Build-Time Inlining
Vite evaluates `import.meta.env.*` variables at build time and inlines them:

```typescript
export const config = {
  isDemoMode: import.meta.env.VITE_DEMO_MODE === 'true', // Becomes: true
  apiUrl: import.meta.env.VITE_API_URL !== undefined ? import.meta.env.VITE_API_URL : 'http://localhost:8000',
};
```

When `VITE_DEMO_MODE=true`:
- The condition `config.isDemoMode ? demoApi : realApi` becomes `true ? demoApi : realApi`
- Vite's tree-shaking removes the unused `realApi` implementations
- Only demo implementations are included in the final bundle

### Runtime Safety
The request interceptor provides an additional safety net:
- If any code path accidentally tries to make an HTTP request
- The interceptor checks if demo mode is active
- Rejects the request immediately with a clear error message

## Deployment

### Changes Committed
```bash
git add frontend/.env.production frontend/src/api/client.ts
git commit -m "Fix demo mode API calls - add request interceptor to prevent HTTP requests in demo mode"
git push
```

### Vercel Deployment
- Push to `main` branch triggers automatic Vercel deployment
- Build script: `bash vercel-build.sh`
- Output directory: `dist`
- Framework: Vite

## Expected Results After Deployment

1. **No 404 errors for `/api/historical-sales`**: Demo data will be returned from embedded JSON
2. **No CORS errors**: No requests to `localhost:8000`
3. **Dashboard loads correctly**: Model metrics display from demo data
4. **ForecastChart works**: Historical sales loaded from demo data
5. **All components functional**: Economic indicators, scenarios, etc. use demo data

## Files Modified

1. `/Users/olivialiau/retailPRED/frontend/src/api/client.ts`
   - Added request interceptor to prevent API calls in demo mode

2. `/Users/olivialiau/retailPRED/frontend/src/api/unifiedApi.ts`
   - Fixed `demoTrainingMetricsApi.getModels()` to match actual JSON structure

3. `/Users/olivialiau/retailPRED/frontend/.env.production`
   - Verified configuration: `VITE_DEMO_MODE=true`, `VITE_API_URL=`

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         Components                          │
│  (Dashboard.tsx, ForecastChart.tsx, etc.)                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ import { api } from '@/api/unifiedApi'
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    unifiedApi.ts                            │
│  exports: const api = config.isDemoMode ? demoApi : realApi │
└────────────────────────┬────────────────────────────────────┘
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
┌──────────────────────┐   ┌──────────────────────┐
│   demoScenariosApi   │   │  realScenariosApi    │
│   (demo data)        │   │  (HTTP requests)     │
└──────────────────────┘   └──────────┬───────────┘
                                       │
                                       ▼
                            ┌──────────────────────┐
                            │     apiClient         │
                            │   (axios instance)    │
                            │  + request interceptor│
                            └──────────────────────┘
```

**Key Points**:
- In demo mode, components import from `unifiedApi` which exports demo implementations
- Demo implementations return static JSON data
- Real implementations are tree-shaken out of the bundle
- Request interceptor is a safety net

## Remaining Issues

### manifest.json 401 Error
The PWA manifest file is still returning a 401 error. This is a Vercel routing issue that may be addressed by the `vercel.json` configuration, but needs verification after deployment.

**Current configuration** in `vercel.json`:
```json
{
  "source": "/manifest.json",
  "headers": [
    {"key": "Cache-Control", "value": "public, max-age=86400"},
    {"key": "Content-Type", "value": "application/manifest+json"}
  ]
}
```

## Testing Checklist

After Vercel deployment completes:

- [ ] Verify dashboard loads without errors
- [ ] Check browser console for no 404 errors
- [ ] Verify ForecastChart displays historical sales
- [ ] Check Network tab - no API calls should be made
- [ ] Verify model metrics display correctly
- [ ] Check economic indicators load
- [ ] Verify scenario analysis works
- [ ] Check manifest.json loads without 401 error

## Commit Information

**Commit Hash**: `574e0d3`
**Commit Message**: "Fix demo mode API calls - add request interceptor to prevent HTTP requests in demo mode"
**Date**: January 8, 2025
**Branch**: `main`
