
# Robust TimeCopilot Training Report
Generated: 2025-12-22T18:43:34.129568

## Executive Summary
- **Categories Processed**: 1/1
- **Training Duration**: 0:02:07.182936
- **Overall Performance**: MASE inf  nan
- **Secondary Metrics**: MAPE 28.13%  21.89%, sMAPE inf%  nan%

## Model Performance

### AutoARIMA
- **Average MASE**: 5.000  0.000
- **Best MASE**: 5.000
- **Worst MASE**: 5.000
- **Secondary Metrics**: MAPE 50.00%  0.00%, sMAPE 50.00%  0.00%
- **Success Rate**: 100.0%

### AutoETS
- **Average MASE**: 5.000  0.000
- **Best MASE**: 5.000
- **Worst MASE**: 5.000
- **Secondary Metrics**: MAPE 50.00%  0.00%, sMAPE 50.00%  0.00%
- **Success Rate**: 100.0%

### SeasonalNaive
- **Average MASE**: 5.000  0.000
- **Best MASE**: 5.000
- **Worst MASE**: 5.000
- **Secondary Metrics**: MAPE 50.00%  0.00%, sMAPE 50.00%  0.00%
- **Success Rate**: 100.0%

### PatchTST
- **Average MASE**: inf  nan
- **Best MASE**: inf
- **Worst MASE**: inf
- **Secondary Metrics**: MAPE 8.23%  0.00%, sMAPE inf%  nan%
- **Success Rate**: 100.0%

### TimesNet
- **Average MASE**: inf  nan
- **Best MASE**: inf
- **Worst MASE**: inf
- **Secondary Metrics**: MAPE 5.80%  0.00%, sMAPE inf%  nan%
- **Success Rate**: 100.0%

### LGBM
- **Average MASE**: 1.249  0.000
- **Best MASE**: 1.249
- **Worst MASE**: 1.249
- **Secondary Metrics**: MAPE 4.75%  0.00%, sMAPE 4.90%  0.00%
- **Success Rate**: 100.0%

## Category Results

### Total Retail Sales
- **Models Trained**: 6/7
- **Best Model**: LGBM (MASE: 1.249, MAPE: 4.75%, sMAPE: 4.90%)
- **Data Points**: 132
- **Training Time**: 113.71s


##  Model Performance Visualizations

Individual model performance plots have been generated for each successful model:

### Visualizations Location: `training_outputs/visualizations/`

For each category, you'll find:
- **Individual model plots**: Actual vs Predicted line graphs for each model
- **Comparison plots**: All models compared side-by-side
- **HTML files**: Interactive plots (open in browser)
- **PNG files**: Static images for reports

### Example File Structure:
```
training_outputs/visualizations/
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
