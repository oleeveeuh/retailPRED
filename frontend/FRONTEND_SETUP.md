# React + TypeScript Dashboard - Complete Setup

Comprehensive dashboard for RetailPRED with routing, API integration, and interactive visualizations.

## Table of Contents

- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Development](#development)
- [Components](#components)
- [API Integration](#api-integration)
- [Routing](#routing)
- [Styling](#styling)

---

## Project Structure

```
frontend/src/
├── api/
│   └── client.ts                 # Typed API client with axios
├── components/
│   ├── layout/
│   │   ├── Sidebar.tsx          # Navigation sidebar
│   │   ├── Header.tsx           # Header with refresh button
│   │   ├── Layout.tsx           # Main layout wrapper
│   │   └── index.ts            # Layout exports
│   └── (other components)
├── pages/
│   ├── PredictionsPage.tsx      # Main forecasting interface
│   ├── ModelsPage.tsx           # Model comparison table
│   ├── ValidationPage.tsx       # Prediction vs actual tracking
│   ├── ExplainPage.tsx          # SHAP explorer
│   └── index.ts                # Page exports
├── App.tsx                      # Main app with routing
├── index.css                    # Global styles with Tailwind
└── main.tsx                     # Entry point
```

---

## Features

### 1. **Predictions Page** (`/dashboard/predictions`)
- ✅ Generate sales forecasts
- ✅ Configure prediction parameters (store, product, weeks ahead, granularity)
- ✅ View forecast chart with confidence intervals
- ✅ SHAP value visualization
- ✅ Detailed prediction results table
- ✅ Real-time prediction generation

### 2. **Models Page** (`/dashboard/models`)
- ✅ List all trained models
- ✅ Model performance comparison charts
- ✅ Train new models button
- ✅ Filter by active status
- ✅ Detailed metrics table (RMSE, MAE, R²)
- ✅ Model status indicators

### 3. **Validation Page** (`/dashboard/validation`)
- ✅ Validate predictions with actual values
- ✅ Track prediction accuracy over time
- ✅ Predicted vs actual comparison chart
- ✅ Accuracy scatter plot
- ✅ Summary statistics (avg error, min/max error)
- ✅ Validated predictions table

### 4. **Explainability Page** (`/dashboard/explain`)
- ✅ Get SHAP explanations for any prediction
- ✅ Feature contributions bar chart
- ✅ Feature importance pie chart
- ✅ Positive/negative impact summary
- ✅ Detailed feature breakdown table
- ✅ Text-based explanation summary

### 5. **Layout Components**
- ✅ **Sidebar** - Navigation with active state highlighting
- ✅ **Header** - Data refresh button with status
- ✅ **Responsive Design** - Mobile-friendly layout

---

## Installation

### 1. Install Dependencies

```bash
cd frontend
npm install
```

Required packages:
```json
{
  "dependencies": {
    "react": "^18.x",
    "react-dom": "^18.x",
    "react-router-dom": "^6.x",
    "axios": "^1.x",
    "@tanstack/react-query": "^5.x",
    "recharts": "^2.x"
  },
  "devDependencies": {
    "@types/react": "^18.x",
    "@types/react-dom": "^18.x",
    "@vitejs/plugin-react": "^4.x",
    "autoprefixer": "^10.x",
    "postcss": "^8.x",
    "tailwindcss": "^3.x",
    "typescript": "^5.x",
    "vite": "^5.x"
  }
}
```

### 2. Environment Variables

Create `.env` file:
```bash
VITE_API_URL=http://localhost:8000
```

---

## Development

### Start Development Server

```bash
cd frontend
npm run dev
```

Dashboard will be available at: http://localhost:5173

### Build for Production

```bash
npm run build
```

Output in `dist/` directory.

### Preview Production Build

```bash
npm run preview
```

---

## Components

### Layout Components

#### **Sidebar** (`components/layout/Sidebar.tsx`)

Navigation sidebar with:
- Logo and branding
- Navigation links (Predictions, Models, Validation, Explainability)
- Active state highlighting
- Footer with version info

```tsx
import { Sidebar } from './components/layout';

// Used in Layout wrapper
```

#### **Header** (`components/layout/Header.tsx`)

Header component with:
- Page title
- Data refresh button
- System status indicator
- User info display
- Refresh status messages

```tsx
import { Header } from './components/layout';

// Features:
// - Refreshes data from backend API
// - Shows success/error states
// - Invalidates TanStack Query cache
```

#### **Layout** (`components/layout/Layout.tsx`)

Main layout wrapper combining Sidebar and Header.

```tsx
import { Layout } from './components/layout';

<Layout>
  <YourPage />
</Layout>
```

---

### Page Components

#### **PredictionsPage** (`pages/PredictionsPage.tsx`)

**Features:**
- Prediction parameter form
- Store ID, Product ID, Weeks Ahead, Granularity
- Forecast line chart with confidence intervals
- SHAP value bar chart
- Detailed results table

**TanStack Query:**
```tsx
const predictionMutation = useMutation({
  mutationFn: predictionsApi.predict,
  onSuccess: (data) => {
    console.log('Prediction successful:', data);
  },
});
```

**Usage:**
```tsx
<PredictionsPage />
```

---

#### **ModelsPage** (`pages/ModelsPage.tsx`)

**Features:**
- Model comparison bar chart
- Train new models button
- Active/inactive filter
- Detailed metrics table
- Stats cards (total, active, inactive)

**TanStack Query:**
```tsx
const { data: modelsData, isLoading, refetch } = useQuery({
  queryKey: ['models', showActiveOnly],
  queryFn: () => modelsApi.getAll({ active_only: showActiveOnly }),
});
```

**Usage:**
```tsx
<ModelsPage />
```

---

#### **ValidationPage** (`pages/ValidationPage.tsx`)

**Features:**
- Validation form (Prediction ID + Actual Value)
- Predicted vs Actual line chart
- Accuracy scatter plot
- Summary statistics cards
- Validated predictions table

**TanStack Query:**
```tsx
const { data: historyData, refetch } = useQuery({
  queryKey: ['predictionHistory'],
  queryFn: () => predictionsApi.getHistory({ limit: 100 }),
});
```

**Usage:**
```tsx
<ValidationPage />
```

---

#### **ExplainPage** (`pages/ExplainPage.tsx`)

**Features:**
- SHAP explanation query form
- Feature contributions bar chart
- Feature importance pie chart
- Positive/negative impact summary
- Detailed features table
- Text explanation summary

**TanStack Query:**
```tsx
const { data: shapData } = useQuery({
  queryKey: ['shapExplanation', predictionId, topN],
  queryFn: () => predictionsApi.getSHAPExplanation(Number(predictionId), topN),
  enabled: !!predictionId,
});
```

**Usage:**
```tsx
<ExplainPage />
```

---

## API Integration

### API Client (`api/client.ts`)

Comprehensive typed API client with:

**Types:**
- `ModelType` enum
- `Granularity` enum
- `PredictionRequest`
- `PredictionResponse`
- `SHAPValue`
- `ForecastPoint`
- `ModelInfo`
- `ModelMetrics`
- `ValidationRequest`
- `ValidationResponse`
- And more...

**API Functions:**

```typescript
import { predictionsApi, modelsApi, dataApi } from './api/client';

// Make a prediction
const result = await predictionsApi.predict({
  store_id: 1,
  product_id: 101,
  weeks_ahead: 4,
  granularity: Granularity.WEEKLY,
});

// Get all models
const models = await modelsApi.getAll({ active_only: true });

// Refresh data
const refresh = await dataApi.refresh();

// Validate prediction
const validation = await predictionsApi.validate({
  prediction_id: 123,
  actual_value: 1525.75,
});

// Get SHAP explanation
const shap = await predictionsApi.getSHAPExplanation(123, 10);
```

---

## Routing

### React Router Setup

**App.tsx:**
```tsx
<BrowserRouter>
  <Layout>
    <Routes>
      <Route path="/" element={<Navigate to="/dashboard/predictions" replace />} />
      <Route path="/dashboard/predictions" element={<PredictionsPage />} />
      <Route path="/dashboard/models" element={<ModelsPage />} />
      <Route path="/dashboard/validation" element={<ValidationPage />} />
      <Route path="/dashboard/explain" element={<ExplainPage />} />
    </Routes>
  </Layout>
</BrowserRouter>
```

### Navigation

**Sidebar links:**
```tsx
<Link to="/dashboard/predictions">Predictions</Link>
<Link to="/dashboard/models">Models</Link>
<Link to="/dashboard/validation">Validation</Link>
<Link to="/dashboard/explain">Explainability</Link>
```

### Programmatic Navigation

```tsx
import { useNavigate } from 'react-router-dom';

const navigate = useNavigate();
navigate('/dashboard/predictions');
```

---

## Styling

### Tailwind CSS

**Configuration** (`tailwind.config.js`):
```js
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
```

**Global Styles** (`index.css`):
```css
@tailwind base;
@tailwind components;
@tailwind utilities;

body {
  margin: 0;
  min-width: 320px;
  min-height: 100vh;
}
```

### Common Tailwind Classes Used

**Layout:**
- `flex`, `flex-1`, `flex-col`
- `grid`, `grid-cols-1`, `grid-cols-2`, `grid-cols-3`, `grid-cols-4`
- `gap-4`, `gap-6`
- `p-6`, `px-6`, `py-4`
- `min-h-screen`

**Colors:**
- `bg-white`, `bg-gray-50`, `bg-slate-900`
- `text-gray-900`, `text-gray-600`, `text-gray-500`
- `bg-blue-600`, `text-blue-600`
- `bg-green-600`, `text-green-600`
- `bg-red-600`, `text-red-600`

**Components:**
- `rounded-lg`, `rounded-full`
- `shadow`, `shadow-sm`
- `border`, `border-gray-200`
- `hover:bg-gray-50`
- `focus:ring-2`, `focus:ring-blue-500`

---

## TanStack Query

### Query Client Setup

```tsx
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

### Using Queries

```tsx
import { useQuery } from '@tanstack/react-query';
import { modelsApi } from './api/client';

const { data, isLoading, error } = useQuery({
  queryKey: ['models'],
  queryFn: modelsApi.getAll,
});
```

### Using Mutations

```tsx
import { useMutation } from '@tanstack/react-query';
import { predictionsApi } from './api/client';

const mutation = useMutation({
  mutationFn: predictionsApi.predict,
  onSuccess: (data) => {
    console.log('Success:', data);
  },
});

mutation.mutate({ store_id: 1, weeks_ahead: 4 });
```

### Cache Invalidation

```tsx
import { useQueryClient } from '@tanstack/react-query';

const queryClient = useQueryClient();

// Invalidate all queries
queryClient.invalidateQueries();

// Invalidate specific query
queryClient.invalidateQueries({ queryKey: ['models'] });
```

---

## Charts

### Recharts Integration

**Line Chart** (Predictions & Validation):
```tsx
<LineChart data={data}>
  <CartesianGrid strokeDasharray="3 3" />
  <XAxis dataKey="date" />
  <YAxis />
  <Tooltip />
  <Legend />
  <Line type="monotone" dataKey="Predicted" stroke="#3b82f6" />
  <Line type="monotone" dataKey="Actual" stroke="#10b981" />
</LineChart>
```

**Bar Chart** (Models & SHAP):
```tsx
<BarChart data={data}>
  <CartesianGrid strokeDasharray="3 3" />
  <XAxis dataKey="name" />
  <YAxis />
  <Tooltip />
  <Bar dataKey="value" fill="#3b82f6" />
</BarChart>
```

**Pie Chart** (Explainability):
```tsx
<PieChart>
  <Pie data={data} dataKey="value" name="Feature" cx="50%" cy="50%">
    {data.map((entry, index) => (
      <Cell key={index} fill={COLORS[index % COLORS.length]} />
    ))}
  </Pie>
  <Tooltip />
</PieChart>
```

---

## Features by Page

### Predictions Page

1. **Parameter Selection**
   - Store ID (optional)
   - Product ID (optional)
   - Weeks ahead (1-12)
   - Granularity (daily/weekly/monthly)

2. **Generate Prediction**
   - Mutation to backend API
   - Loading state
   - Error handling

3. **Visualizations**
   - Forecast line chart with confidence intervals
   - SHAP value bar chart
   - Results table

4. **Details**
   - Prediction ID
   - Model name and type
   - SHAP values table
   - Timestamp

### Models Page

1. **Statistics**
   - Total models
   - Active models
   - Inactive models

2. **Comparison Chart**
   - RMSE, MAE, R² comparison
   - Grouped bar chart

3. **Table**
   - Model details
   - Performance metrics
   - Status badges
   - Training date

4. **Training**
   - Train new models button
   - Training progress
   - Success/error feedback

### Validation Page

1. **Validate Predictions**
   - Enter Prediction ID
   - Enter Actual Value
   - Submit to API

2. **Accuracy Summary**
   - Total predictions
   - Validated count
   - Average error %
   - Min error %

3. **Charts**
   - Predicted vs Actual line chart
   - Accuracy scatter plot

4. **Table**
   - Validated predictions
   - Error percentages
   - Color-coded accuracy

### Explain Page

1. **Get Explanation**
   - Enter Prediction ID
   - Select Top N features
   - Query SHAP API

2. **Summary Cards**
   - Prediction ID
   - Predicted value
   - Base value
   - Total SHAP value

3. **Charts**
   - Feature contributions (horizontal bar)
   - Feature importance (pie chart)

4. **Impact Analysis**
   - Positive impact total
   - Negative impact total

5. **Details Table**
   - Feature name
   - Contribution value
   - Percentage
   - Impact type (positive/negative)

---

## Error Handling

### API Errors

```tsx
const { data, error, isLoading } = useQuery({
  queryKey: ['models'],
  queryFn: modelsApi.getAll,
});

if (error) {
  return (
    <div className="bg-red-50 border border-red-200 rounded-lg p-4">
      <p className="text-red-800">Failed to load models</p>
    </div>
  );
}
```

### Mutation Errors

```tsx
const mutation = useMutation({
  mutationFn: predictionsApi.predict,
  onError: (error) => {
    console.error('Prediction failed:', error);
  },
});

if (mutation.error) {
  return (
    <div className="bg-red-50 border border-red-200 rounded-lg p-4">
      <p className="text-red-800">Failed to generate forecast</p>
    </div>
  );
}
```

---

## Best Practices

1. **Type Safety** - Use TypeScript types for all API calls
2. **Error Boundaries** - Handle errors gracefully
3. **Loading States** - Show loading indicators during queries
4. **Cache Management** - Invalidate cache after mutations
5. **Responsive Design** - Use Tailwind responsive classes
6. **Accessibility** - Use semantic HTML and ARIA labels
7. **Performance** - Use React.memo for expensive components

---

## Troubleshooting

### Issue: API calls failing

**Check:**
- Backend is running (`http://localhost:8000`)
- VITE_API_URL is set correctly
- CORS is enabled in backend

### Issue: Charts not rendering

**Check:**
- Recharts is installed
- Data is in correct format
- Container has defined height

### Issue: Routing not working

**Check:**
- react-router-dom is installed
- BrowserRouter wraps routes
- Route paths match navigation links

---

## Summary

The React dashboard provides:

✅ **4 Complete Pages** - Predictions, Models, Validation, Explainability
✅ **Full API Integration** - Typed client with all endpoints
✅ **React Router** - Navigation with active states
✅ **TanStack Query** - Data fetching and caching
✅ **Recharts** - Interactive visualizations
✅ **Tailwind CSS** - Responsive styling
✅ **TypeScript** - Type safety throughout
✅ **Error Handling** - Graceful error states
✅ **Loading States** - User feedback during queries
✅ **Production Ready** - Optimized and documented

**Ready to run!** 🚀

```bash
cd frontend
npm install
npm run dev
```

Then visit http://localhost:5173
