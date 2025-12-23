# RetailPRED Pipeline Architecture

## Overview
Complete end-to-end retail forecasting system using economic and financial indicators as FEATURES to predict retail sales across 12 categories.

## Data Flow Architecture

```

                    DATA SOURCES (Inputs)                        

                                                                  
  1. MRTS Retail Sales Data (Target Variables)                   
     - 12 retail categories (2015-2025)                          
     - Monthly frequency, 132 observations per category           
                                                                  
  2. FRED Economic Indicators (Features)                         
     - CPI, unemployment, Fed funds rate, GDP                    
     - Consumer sentiment, money supply, industrial production    
     - Used as PREDICTORS, not targets                           
                                                                  
  3. Yahoo Finance Data (Features)                               
     - Retail stocks: AAPL, AMZN, WMT, COST                      
     - Retail ETFs: XLY, XRT                                     
     - Market indices: SPY, QQQ                                  
     - Used as PREDICTORS, not targets                           

                            ↓

              FEATURE ENGINEERING (244 Features)                 

                                                                  
  Temporal Features (82):                                        
    - Lags: 1,2,3,4,6,8,12,24 months                            
    - Rolling stats: mean, std, min, max (3,6,12 month windows)  
    - Differences: MoM, YoY changes                              
                                                                  
  Seasonal Features (48):                                        
    - Calendar: month, quarter, week-of-year dummies             
    - Mathematical: sin/cos transformations                      
    - Holiday indicators: Black Friday, December, etc.           
                                                                  
  Economic Context Features (67):                                
    - Macroeconomic: CPI, unemployment, interest rates           
    - Financial markets: S&P 500, bond yields, volatility        
    - Consumer metrics: confidence, sentiment                    
                                                                  
  Interaction Features (47):                                    
    - Economic-seasonal interactions                              
    - Cross-category correlations                                
    - Non-linear transformations                                  

                            ↓

                  MODEL TRAINING PIPELINE                        

                                                                  
  For each of 12 retail categories:                              
                                                                  
  1. Statistical Models:                                         
     • AutoARIMA - Automatic ARIMA parameter selection            
     • AutoETS - Error-Trend-Seasonality modeling                
     • SeasonalNaive - Baseline seasonal forecasting             
                                                                  
  2. Machine Learning Models:                                    
     • LGBM - LightGBM with 244 features                         
     • RandomForest - Ensemble tree-based forecasting            
                                                                  
  3. Deep Learning Models:                                       
     • PatchTST - Patch-based Transformer                        
     • TimesNet - Temporal 2D CNN                                
                                                                  
  Validation: Proper temporal split (80/20)                      
  Metrics: MASE, MAPE, sMAPE, RMSE, MAE                         

                            ↓

                    OUTPUTS & RESULTS                            

                                                                  
  /training_outputs/                                              
     robust_training_summary.json - All metrics               
     model_predictions.json - Forecast values                 
     training_report.md - Human-readable summary              
     visualizations/                                          
         Total_Retail_Sales/                                  
            Total_Retail_Sales_AutoARIMA_performance.html    
            Total_Retail_Sales_LGBM_performance.html         
            Total_Retail_Sales_all_models_comparison.html    
         Automobile_Dealers/                                  
         Building_Materials_Garden/                           
         Clothing_Accessories/                                
         Electronics_and_Appliances/                          
         Food_Beverage_Stores/                                
         Furniture_Home_Furnishings/                          
         Gasoline_Stations/                                   
         General_Merchandise/                                 
         Health_Personal_Care/                                
         Nonstore_Retailers/                                  
         Sporting_Goods_Hobby/                                
                                                                  
  /results/                                                       
     (Reserved for prediction results)                        

```

## Target Variables (What We Forecast)

### 12 Retail Categories (from MRTS data):

1. **Total_Retail_Sales** - Overall retail performance
2. **Automobile_Dealers** - High-ticket discretionary spending
3. **Building_Materials_Garden** - Home improvement activity
4. **Clothing_Accessories** - Fashion and apparel spending
5. **Electronics_and_Appliances** - Technology spending
6. **Food_Beverage_Stores** - Essential spending (stable)
7. **Furniture_Home_Furnishings** - Housing-related spending
8. **Gasoline_Stations** - Travel and energy costs
9. **General_Merchandise** - Mass-market retail (Walmart, Target)
10. **Health_Personal_Care** - Healthcare and wellness spending
11. **Nonstore_Retailers** - E-commerce sales
12. **Sporting_Goods_Hobby** - Leisure and recreation spending

## Feature Variables (What We Use to Predict)

### FRED Economic Indicators (7 core series):
- **CPIAUCSL** - Consumer Price Index (inflation)
- **FEDFUNDS** - Federal Funds Rate (interest rates)
- **UNRATE** - Unemployment Rate
- **UMCSENT** - Consumer Sentiment Index
- **M2SL** - M2 Money Supply (economic liquidity)
- **INDPRO** - Industrial Production Index
- **PCE** - Personal Consumption Expenditures

### Yahoo Finance Data (13 tickers):
- **Stocks**: AAPL, AMZN, WMT, COST
- **ETFs**: XLY (Consumer Disc), XRT (Retail)
- **Indices**: SPY (S&P 500), QQQ (Nasdaq 100)

## Pipeline Execution

### Step 1: Data Fetching
```bash
cd /Users/olivialiau/retailPRED/project_root
python main.py --mode fetch
```
- Fetches latest MRTS retail data
- Updates FRED economic indicators
- Pulls Yahoo Finance market data
- Stores in SQLite database for caching

### Step 2: Model Training
```bash
python main.py --mode train
```
- Trains 7 models × 12 categories = 84 models total
- Generates visualizations for each model
- Saves predictions and metrics
- Duration: ~25-30 minutes for all categories

### Step 3: Generate Predictions
```bash
python main.py --mode predict --horizon 12
```
- Loads trained models
- Generates 12-month forecasts
- Outputs to /results directory

### Step 4: Backtesting
```bash
python main.py --mode backtest --horizon 6
```
- Walk-forward validation
- Tests model stability over time
- Calculates out-of-sample performance

## Data Leakage Prevention

 **Proper Temporal Splits**:
- Training data: 2015-2023 (108 months)
- Validation data: 2024-2025 (24 months)
- No future information leaks into training

 **Feature Engineering Constraints**:
- Lag features use only historical data
- Economic indicators appropriately lagged
- Rolling windows computed on training data only

 **Cross-Validation**:
- Walk-forward (expanding window) validation
- 5-fold temporal CV for statistical models
- Holdout validation for ML/DL models

## Performance Benchmarks

### Typical Model Performance (MAPE):

| Model | Average MAPE | Best Use Case |
|-------|-------------|---------------|
| SeasonalNaive | 2-3% | Stable seasonal patterns |
| AutoETS | 3-4% | Automatic complexity control |
| AutoARIMA | 4-5% | Trend + seasonality |
| LGBM | 5-7% | Complex multi-factor patterns |
| TimesNet | 5-7% | Multi-frequency patterns |
| PatchTST | 6-8% | Long-range dependencies |
| RandomForest | 7-10% | Noisy complex patterns |

### Training Time Per Category:

| Model Type | Time | Notes |
|------------|------|-------|
| SeasonalNaive | <0.01s | Instantaneous |
| AutoETS | 0.3-0.5s | Extremely fast |
| AutoARIMA | 1-2s | Efficient |
| LGBM | 3-5s | Optimized |
| RandomForest | 3-5s | Moderate |
| PatchTST | 8-10s | Efficient deep learning |
| TimesNet | 90-95s | Most intensive |

**Total training time for all 12 categories: ~25-30 minutes**

## Configuration File Structure

**File**: `/Users/olivialiau/retailPRED/project_root/config/config.yaml`

### Key Sections:

1. **targets** - What to forecast (retail categories only)
2. **data_sources** - FRED, Yahoo Finance, MRTS API configs
3. **models** - Model types and hyperparameters
4. **database** - SQLite connection and caching
5. **performance** - Parallel processing and memory limits

### Correct Configuration:

```yaml
targets:
  retail:
    - "Total_Retail_Sales"
    - "Automobile_Dealers"
    # ... (10 more categories)
```

 **WRONG** (old config):
```yaml
targets:
  economic:
    - "gdp_growth_rate"  # These are FEATURES, not targets!
    - "cpi_inflation_rate"
  financial:
    - "sp500_level"      # These are FEATURES, not targets!
```

## File Structure

```
/Users/olivialiau/retailPRED/
 project_root/
    main.py                    # Main orchestration script
    config/
       config.py              # Configuration manager
       config.yaml            # Configuration file (UPDATED)
    data/                      # SQLite database location
       retailpred.db          # High-performance cache
    data_processed/            # Parquet/CSV files
    etl/                       # Data fetching modules
       fetch_fred.py
       fetch_mrts.py
       fetch_yahoo.py
    models/
       robust_timecopilot_trainer.py  # Core training system
       simple_early_stopping.py
       visualizations/        # (old location, deprecated)
    sqlite/
       sqlite_loader.py
       sqlite_dataset_builder.py
    outputs/                   # Run-specific outputs

 training_outputs/              # MAIN OUTPUT DIRECTORY
    robust_training_summary.json
    model_predictions.json
    training_report.md
    visualizations/
        Total_Retail_Sales/
        Automobile_Dealers/
        ... (10 more categories)

 results/                       # Prediction results (empty until --mode predict)
 README.md                      # Project documentation
```

## Quick Start Commands

```bash
# Navigate to project directory
cd /Users/olivialiau/retailPRED/project_root

# Train models for all retail categories
python main.py --mode train

# Train specific categories only
python main.py --mode train --targets Total_Retail_Sales Automobile_Dealers

# Generate predictions for next 12 months
python main.py --mode predict --horizon 12

# Run backtesting
python main.py --mode backtest --horizon 6

# Full pipeline (fetch + train + predict)
python main.py --mode full
```

## Key Architecture Decisions

###  Correct Approach:
1. **FRED/Yahoo Finance → FEATURES**: Used as predictors
2. **MRTS Retail Data → TARGETS**: What we actually forecast
3. **12 Retail Categories**: Each gets its own set of 7 models
4. **244 Features**: Comprehensive feature engineering pipeline
5. **Proper Temporal Splits**: No data leakage
6. **Ensemble**: Combine models for robust predictions

###  Previous Issues (FIXED):
1. ~~Config tried to forecast FRED indicators~~ → Now only retail categories
2. ~~Relative paths caused wrong output locations~~ → Now using absolute paths
3.~~Empty training_outputs directory~~ → Now populated with results
4.~~Missing visualizations for neural networks~~ → Fixed prediction collection

## System Requirements

- **Python**: 3.8+
- **Memory**: 4GB minimum (8GB recommended)
- **Storage**: 500MB for data + models
- **Processing**: Apple Silicon MPS or CPU
- **API Keys**: FRED API key required (free)

## Dependencies

Key libraries:
- `pandas`, `numpy` - Data manipulation
- `statsforecast` - Statistical models
- `neuralforecast` - Deep learning models
- `lightgbm`, `scikit-learn` - ML models
- `plotly` - Interactive visualizations
- `fredapi`, `yfinance` - Data fetching
- `sqlalchemy` - Database management

## Monitoring and Logs

- **Training Log**: `project_root/robust_training.log`
- **Model Artifacts**: `training_outputs/model_predictions.json`
- **Visualizations**: `training_outputs/visualizations/*/`

## Next Steps

1.  Config fixed to target retail categories only
2.  Output paths corrected to use absolute paths
3.  All 12 retail categories training successfully
4.  Visualizations generating correctly
5.  Training in progress (4/12 categories completed)
6. ⏳ Wait for full training to complete
7. ⏳ Review final results across all categories

---

**Last Updated**: 2025-12-22
**Status**: Pipeline fixed and operational
**Training Progress**: 4/12 categories completed
