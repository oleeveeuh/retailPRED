# Tableau Integration Guide for RetailPRED

## Overview

This guide explains how to integrate RetailPRED with Tableau for executive-level dashboards and business intelligence reporting.

## Features

- **CSV Export**: Export predictions, historical data, and model performance to CSV
- **Tableau Public Embed**: Embed interactive Tableau Public dashboards
- **Real-time Data Sync**: Export latest predictions for Tableau analysis
- **Executive Summary View**: Business-focused dashboard in RetailPRED

## Quick Start

### 1. Export Data from RetailPRED

The easiest way to get data into Tableau is via CSV export:

```bash
# Export all predictions
curl "http://localhost:8000/api/export/predictions-csv" -o predictions.csv

# Export with date filter
curl "http://localhost:8000/api/export/predictions-csv?start_date=2025-01-01&end_date=2025-12-31" -o predictions_2025.csv

# Export specific model
curl "http://localhost:8000/api/export/predictions-csv?model_name=lightgbm" -o lgbm_predictions.csv

# Export historical data
curl "http://localhost:8000/api/export/historical-csv?category=total_sales" -o historical_sales.csv

# Export model performance metrics
curl "http://localhost:8000/api/export/model-performance-csv" -o model_performance.csv
```

### 2. Import into Tableau

1. Open Tableau Desktop or Tableau Public
2. Click "Connect to Data" → "Text File"
3. Select the exported CSV file
4. Tableau will auto-detect the schema

### 3. Create Visualizations

**Recommended Visualizations**:

- **Line Chart**: Predicted vs Actual sales over time
  - Drag `date` to Columns
  - Drag `predicted_sales` and `actual_sales` to Rows
  - Use dual axis for comparison

- **Scatter Plot**: Prediction accuracy
  - Drag `predicted_sales` to Columns
  - Drag `actual_sales` to Rows
  - Add `error_pct` to Color

- **Bar Chart**: Model performance comparison
  - Drag `model_name` to Columns
  - Drag `error_pct` to Rows
  - Sort by error percentage

- **Heat Map**: Seasonal patterns
  - Drag `date` to Columns (set to Month)
  - Drag `model_name` to Rows
  - Drag `predicted_sales` to Color

## CSV Schema

### Predictions CSV (`/api/export/predictions-csv`)

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `date` | Date | Prediction date | 2025-01-15 |
| `store` | String | Store ID or "All Stores" | 12345 |
| `product` | String | Product ID or "All Products" | 67890 |
| `predicted_sales` | Float | Predicted sales value | 21877.51 |
| `actual_sales` | Float | Actual sales (NULL if not validated) | 22045.50 |
| `model_name` | String | Model used for prediction | total_sales_lightgbm_model |
| `error_pct` | Float | Percentage error (if validated) | 0.77 |
| `confidence_lower` | Float | Lower confidence bound | 21755.0 |
| `confidence_upper` | Float | Upper confidence bound | 22000.0 |
| `is_validated` | String | Whether prediction has actual value | Yes/No |

### Historical Data CSV (`/api/export/historical-csv`)

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `date` | Date | Sales date | 2025-01-01 |
| `category` | String | Category name | Total Sales |
| `sales` | Float | Actual sales value | 40528.00 |
| `created_at` | Timestamp | Database record creation time | 2025-01-02 10:30:00 |

### Model Performance CSV (`/api/export/model-performance-csv`)

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `model_name` | String | Full model name | total_sales_lgbm_model |
| `model_type` | String | Model architecture | LGBM |
| `category` | String | Category forecasted | Total Sales |
| `mape_percentage` | Float | Mean Absolute Percentage Error | 1.42 |
| `rmse` | Float | Root Mean Squared Error | 245.67 |
| `mae` | Float | Mean Absolute Error | 198.34 |
| `r_squared` | Float | R-squared score | 0.95 |
| `is_active` | String | Whether model is active | Yes/No |
| `created_at` | Timestamp | Model training date | 2025-01-01 15:20:00 |

## Tableau Public Integration

### Option 1: Embed Existing Dashboard

If you have a Tableau Public dashboard:

1. **Create Dashboard on Tableau Public**
   - Go to [tableaupublic.com](https://public.tableau.com)
   - Create account and upload your workbook
   - Publish your dashboard

2. **Get Embed URL**
   - Open your dashboard on Tableau Public
   - Click "Share" → "Embed Code"
   - Copy the URL (should look like: `https://public.tableau.com/views/...`)

3. **Configure in RetailPRED**
   ```bash
   # Add to .env file
   VITE_TABLEAU_EMBED_URL=https://public.tableau.com/views/YourDashboardName
   ```

4. **Restart Frontend**
   ```bash
   cd frontend
   npm run dev
   ```

5. **View in Business Dashboard**
   - Navigate to "Business View" in sidebar
   - Click "Executive Summary" tab
   - See embedded Tableau dashboard

### Option 2: Use CSV Export (Recommended for Flexibility)

1. Export data from RetailPRED using the export endpoints
2. Import into Tableau Desktop/Public
3. Create custom visualizations
4. Publish to Tableau Public (optional)
5. Embed in RetailPRED if desired

## Creating a Tableau Dashboard from Scratch

### Step 1: Prepare the Data

Export multiple CSVs for a complete view:

```bash
# Create directory for Tableau data
mkdir tableau_data
cd tableau_data

# Export all data sources
curl "http://localhost:8000/api/export/predictions-csv" -o predictions.csv
curl "http://localhost:8000/api/export/historical-csv?category=total_sales" -o historical.csv
curl "http://localhost:8000/api/export/model-performance-csv" -o models.csv
```

### Step 2: Import to Tableau

1. Open Tableau Desktop
2. Connect to `predictions.csv`
3. Add `historical.csv` as a second data source
4. Add `models.csv` as a third data source
5. Create relationships:
   - `predictions.date` ↔ `historical.date`
   - `predictions.model_name` ↔ `models.model_name`

### Step 3: Build Dashboard Sheets

**Sheet 1: Sales Forecast vs Actual**
```
Type: Line Chart
Columns: date (Month)
Rows: SUM(predicted_sales), SUM(actual_sales)
Colors: Measure Names
```

**Sheet 2: Prediction Error Distribution**
```
Type: Histogram
Columns: error_pct (bin)
Rows: COUNT(predictions)
Colors: model_name
```

**Sheet 3: Model Performance Leaderboard**
```
Type: Bar Chart
Columns: model_name
Rows: AVG(error_pct)
Sort: Descending by error_pct
```

**Sheet 4: Seasonal Heat Map**
```
Type: Heat Map
Columns: MONTH(date)
Rows: model_name
Color: AVG(predicted_sales)
```

**Sheet 5: Forecast Accuracy Trend**
```
Type: Line Chart
Columns: date (Week)
Rows: AVG(error_pct)
Filters: is_validated = Yes
```

### Step 4: Create Dashboard

1. Click "New Dashboard"
2. Drag all 5 sheets onto dashboard
3. Arrange in grid layout
4. Add filters for:
   - Date range
   - Model name
   - Store/product
5. Add title: "RetailPRED Sales Forecasting Dashboard"

### Step 5: Publish and Embed

**To Tableau Public** (Free):
1. File → Publish to Tableau Public
2. Enter credentials
3. Click "Publish"
4. Copy embed URL from browser

**To Tableau Online/Server** (Paid):
1. File → Publish to Tableau Server
2. Enter server details
3. Click "Publish"
4. Use embed URL for internal dashboards

## Advanced Features

### Real-time Data Refresh

Tableau Desktop doesn't auto-refresh CSVs, but you can:

**Option 1: Manual Refresh**
1. Export new CSV from RetailPRED
2. In Tableau: Data → Refresh
3. Re-publish if using Tableau Public

**Option 2: Use Tableau Server/Online**
1. Set up scheduled extract refresh
2. Point to CSV file on network share
3. Schedule refresh every hour/day/week

**Option 3: Use Web Data Connector**
For production setups, create a Web Data Connector that calls:
```
http://localhost:8000/api/export/predictions-csv
```

### Calculated Fields

Add these calculated fields in Tableau for advanced analysis:

**Prediction Accuracy Score**:
```
IF [is_validated] = "Yes" THEN
  IF [error_pct] < 1 THEN "Excellent"
  ELSEIF [error_pct] < 3 THEN "Good"
  ELSEIF [error_pct] < 5 THEN "Fair"
  ELSE "Poor"
END
END
```

**Forecast Bias**:
```
AVG([predicted_sales] - [actual_sales])
```
*Positive = over-forecasting, Negative = under-forecasting*

**Week-over-Week Change**:
```
LOOKUP([predicted_sales], -1) - [predicted_sales]
```

### Dashboard Interactivity

Add these features for better UX:

1. **Filter Actions**: Click on model name to filter all sheets
2. **Hover Tooltips**: Show confidence intervals on hover
3. **URL Actions**: Click on prediction to open in RetailPRED
4. **Parameters**: Let users select forecast horizon

## Environment Variables

Configure Tableau integration in `.env`:

```bash
# Tableau Public Embed URL (optional)
VITE_TABLEAU_EMBED_URL=https://public.tableau.com/views/YourWorkbook/Dashboard1

# Tableau Server URL (for enterprise)
VITE_TABLEAU_SERVER_URL=https://tableau.yourcompany.com

# Enable/disable Tableau features
VITE_TABLEAU_ENABLED=true
```

## Troubleshooting

### Issue: Dashboard shows blank iframe

**Cause**: Invalid Tableau embed URL or dashboard not public

**Fix**:
1. Verify Tableau dashboard is published as "Public"
2. Check embed URL format: `https://public.tableau.com/views/...`
3. Ensure no auth required (Tableau Public only)

### Issue: CSV export is empty

**Cause**: No predictions in database or incorrect filters

**Fix**:
1. Generate predictions first via `/api/predict`
2. Check database: `sqlite3 data/retailpred.db "SELECT COUNT(*) FROM prediction_log"`
3. Remove date filters to export all data

### Issue: Data not updating in Tableau

**Cause**: Tableau using cached data

**Fix**:
1. Tableau Desktop: Data → Refresh All Extracts
2. Tableau Server: Check refresh schedule
3. Clear browser cache for embedded views

### Issue: CORS errors when fetching CSV

**Cause**: Frontend can't access backend endpoint

**Fix**:
1. Ensure backend is running: `http://localhost:8000`
2. Check CORS settings in `backend/main.py`
3. Verify endpoint works via curl first

## Best Practices

### Data Export
- Export data daily/weekly for analysis
- Use date filters to reduce file size
- Archive old exports by month/quarter

### Dashboard Design
- Use consistent color scheme
- Add context with annotations
- Include executive summary text
- Test on different screen sizes

### Performance
- Limit predictions to < 10K rows for Tableau Public
- Use extracts instead of live connections
- Aggregate data by week/month for faster loading

### Security
- Don't expose sensitive data to Tableau Public
- Use Tableau Server for internal data
- Add authentication to export endpoints if needed

## API Reference

### GET /api/export/predictions-csv

Export predictions to CSV format.

**Query Parameters**:
- `start_date` (optional): Filter predictions from this date (YYYY-MM-DD)
- `end_date` (optional): Filter predictions until this date (YYYY-MM-DD)
- `category` (optional): Filter by category
- `model_name` (optional): Filter by model name (supports partial match)

**Example**:
```bash
curl "http://localhost:8000/api/export/predictions-csv?start_date=2025-01-01&model_name=lightgbm"
```

**Response**: CSV file download

### GET /api/export/historical-csv

Export historical sales data to CSV.

**Query Parameters**:
- `category` (optional): Category to export (default: total_sales)
- `start_date` (optional): Start date filter
- `end_date` (optional): End date filter

**Example**:
```bash
curl "http://localhost:8000/api/export/historical-csv?category=total_sales&start_date=2024-01-01"
```

**Response**: CSV file download

### GET /api/export/model-performance-csv

Export model performance metrics to CSV.

**Example**:
```bash
curl "http://localhost:8000/api/export/model-performance-csv"
```

**Response**: CSV file download with all models and their metrics

## Example Use Cases

### Use Case 1: Monthly Executive Report

1. Export predictions for last month
2. Import into Tableau
3. Create dashboard showing:
   - Total forecast vs actual
   - Top 5 best/worst predictions
   - Model performance comparison
4. Export as PDF
5. Share with stakeholders

### Use Case 2: Model Selection Analysis

1. Export model performance CSV
2. Import into Tableau
3. Create scatter plot:
   - X-axis: MAPE
   - Y-axis: R²
   - Size: Training samples
4. Identify best models by category
5. Update model selection in RetailPRED

### Use Case 3: Seasonal Pattern Discovery

1. Export 2+ years of predictions
2. Import into Tableau
3. Create heat map:
   - Rows: Month
   - Columns: Year
   - Color: Average forecast
4. Identify seasonal trends
5. Plan inventory accordingly

## Resources

- [Tableau Public](https://public.tableau.com) - Free hosting for public dashboards
- [Tableau Desktop](https://www.tableau.com/products/desktop) - Free for personal use
- [Tableau Learning](https://www.tableau.com/learn) - Tutorials and training
- [Tableau Community](https://community.tableau.com) - Forums and support

## Support

For issues with:
- **RetailPRED export endpoints**: Check `/docs/API_DOCUMENTATION.md`
- **Tableau configuration**: See Tableau documentation
- **Dashboard design**: Use Tableau Community forums

## Summary

The Tableau integration provides a powerful way to create executive dashboards from RetailPRED predictions. Whether you use CSV export, Tableau Public embed, or Tableau Server, you can create stunning visualizations that help stakeholders understand forecast performance at a glance.

**Next Steps**:
1. Export data using `/api/export/predictions-csv`
2. Import into Tableau Desktop/Public
3. Create your first dashboard
4. (Optional) Embed back into RetailPRED
5. Share with stakeholders!
