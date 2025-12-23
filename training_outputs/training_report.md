
# Robust TimeCopilot Training Report
Generated: 2025-12-22T23:36:11.391702

## Executive Summary
- **Categories Processed**: 11/11
- **Training Duration**: 0:24:38.543054
- **Overall Performance**: MASE 1.886  0.858
- **Secondary Metrics**: MAPE 7.58%  4.62%, sMAPE 7.42%  4.23%

## Model Performance

### AutoARIMA
- **Average MASE**: 2.469  0.740
- **Best MASE**: 1.116
- **Worst MASE**: 3.663
- **Secondary Metrics**: MAPE 10.66%  5.79%, sMAPE 10.04%  5.20%
- **Success Rate**: 100.0%

### AutoETS
- **Average MASE**: 1.885  0.588
- **Best MASE**: 1.161
- **Worst MASE**: 2.783
- **Secondary Metrics**: MAPE 6.84%  3.39%, sMAPE 6.91%  3.26%
- **Success Rate**: 100.0%

### SeasonalNaive
- **Average MASE**: 1.964  0.326
- **Best MASE**: 1.344
- **Worst MASE**: 2.509
- **Secondary Metrics**: MAPE 6.94%  3.01%, sMAPE 7.30%  3.27%
- **Success Rate**: 100.0%

### PatchTST
- **Average MASE**: 1.964  1.188
- **Best MASE**: 0.163
- **Worst MASE**: 4.056
- **Secondary Metrics**: MAPE 8.14%  5.01%, sMAPE 7.69%  4.52%
- **Success Rate**: 100.0%

### TimesNet
- **Average MASE**: 1.793  1.020
- **Best MASE**: 0.416
- **Worst MASE**: 3.702
- **Secondary Metrics**: MAPE 7.89%  5.19%, sMAPE 7.50%  4.62%
- **Success Rate**: 100.0%

### LGBM
- **Average MASE**: 1.240  0.435
- **Best MASE**: 0.281
- **Worst MASE**: 1.833
- **Secondary Metrics**: MAPE 4.98%  2.01%, sMAPE 5.09%  2.13%
- **Success Rate**: 100.0%

## Category Results

### Total Retail Sales
- **Models Trained**: 6/7
- **Best Model**: AutoARIMA (MASE: 1.116, MAPE: 4.58%, sMAPE: 4.50%)
- **Data Points**: 132
- **Training Time**: 110.82s

### Automobile Dealers
- **Models Trained**: 6/7
- **Best Model**: LGBM (MASE: 1.488, MAPE: 5.64%, sMAPE: 5.74%)
- **Data Points**: 132
- **Training Time**: 112.94s

### Building Materials Garden
- **Models Trained**: 6/7
- **Best Model**: LGBM (MASE: 0.281, MAPE: 1.25%, sMAPE: 1.25%)
- **Data Points**: 132
- **Training Time**: 113.41s

### Clothing Accessories
- **Models Trained**: 6/7
- **Best Model**: TimesNet (MASE: 1.498, MAPE: 4.30%, sMAPE: 4.21%)
- **Data Points**: 132
- **Training Time**: 112.94s

### Electronics And Appliances
- **Models Trained**: 6/7
- **Best Model**: AutoETS (MASE: 1.199, MAPE: 4.55%, sMAPE: 4.61%)
- **Data Points**: 132
- **Training Time**: 111.31s

### Food Beverage Stores
- **Models Trained**: 6/7
- **Best Model**: LGBM (MASE: 0.876, MAPE: 4.08%, sMAPE: 4.12%)
- **Data Points**: 132
- **Training Time**: 155.01s

### Furniture Home Furnishings
- **Models Trained**: 6/7
- **Best Model**: TimesNet (MASE: 0.938, MAPE: 3.62%, sMAPE: 3.55%)
- **Data Points**: 132
- **Training Time**: 111.02s

### Gasoline Stations
- **Models Trained**: 6/7
- **Best Model**: LGBM (MASE: 1.833, MAPE: 3.53%, sMAPE: 3.62%)
- **Data Points**: 132
- **Training Time**: 114.56s

### General Merchandise
- **Models Trained**: 6/7
- **Best Model**: AutoETS (MASE: 1.161, MAPE: 8.01%, sMAPE: 8.63%)
- **Data Points**: 132
- **Training Time**: 110.82s

### Health Personal Care
- **Models Trained**: 6/7
- **Best Model**: PatchTST (MASE: 0.163, MAPE: 1.69%, sMAPE: 1.71%)
- **Data Points**: 132
- **Training Time**: 114.17s

### Sporting Goods Hobby
- **Models Trained**: 6/7
- **Best Model**: AutoETS (MASE: 1.411, MAPE: 4.68%, sMAPE: 4.78%)
- **Data Points**: 132
- **Training Time**: 119.09s


##  Model Performance Visualizations

Individual model performance plots have been generated for each successful model:

### Visualizations Location: `/Users/olivialiau/retailPRED/training_outputs/visualizations/`

For each category, you'll find:
- **Individual model plots**: Actual vs Predicted line graphs for each model
- **Comparison plots**: All models compared side-by-side
- **HTML files**: Interactive plots (open in browser)
- **PNG files**: Static images for reports

### Example File Structure:
```
/Users/olivialiau/retailPRED/training_outputs/visualizations/
 Health_Personal_Care/
    Health_Personal_Care_TimesNet_performance.html
    Health_Personal_Care_TimesNet_performance.png
    Health_Personal_Care_all_models_comparison.html
 [Other categories...]
```

To view interactive plots:
1. Open HTML files in your browser
2. Hover over lines to see detailed values
3. Use legend to toggle models on/off
