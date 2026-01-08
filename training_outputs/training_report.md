
# Robust TimeCopilot Training Report
Generated: 2026-01-04T22:07:44.596747

## Executive Summary
- **Categories Processed**: 4/4
- **Training Duration**: 0:31:56.310330
- **Overall Performance**: MASE 1.280  0.931
- **Secondary Metrics**: MAPE 11.88%  7.89%, sMAPE 11.70%  7.21%

## Model Performance

### AutoARIMA
- **Average MASE**: 1.303  0.375
- **Best MASE**: 0.877
- **Worst MASE**: 1.880
- **Secondary Metrics**: MAPE 12.69%  0.18%, sMAPE 13.91%  0.24%
- **Success Rate**: 100.0%

### AutoETS
- **Average MASE**: 0.991  0.268
- **Best MASE**: 0.681
- **Worst MASE**: 1.402
- **Secondary Metrics**: MAPE 9.60%  0.02%, sMAPE 9.90%  0.02%
- **Success Rate**: 100.0%

### SeasonalNaive
- **Average MASE**: 1.372  0.374
- **Best MASE**: 0.940
- **Worst MASE**: 1.947
- **Secondary Metrics**: MAPE 12.72%  0.00%, sMAPE 13.77%  0.00%
- **Success Rate**: 100.0%

### RandomForest
- **Average MASE**: 0.285  0.251
- **Best MASE**: 0.018
- **Worst MASE**: 0.575
- **Secondary Metrics**: MAPE 2.08%  1.81%, sMAPE 2.13%  1.87%
- **Success Rate**: 100.0%

### PatchTST
- **Average MASE**: 2.383  0.643
- **Best MASE**: 1.645
- **Worst MASE**: 3.373
- **Secondary Metrics**: MAPE 22.21%  0.07%, sMAPE 20.23%  0.05%
- **Success Rate**: 100.0%

### TimesNet
- **Average MASE**: 2.416  0.676
- **Best MASE**: 1.649
- **Worst MASE**: 3.463
- **Secondary Metrics**: MAPE 22.47%  0.16%, sMAPE 20.48%  0.17%
- **Success Rate**: 100.0%

### LGBM
- **Average MASE**: 0.207  0.186
- **Best MASE**: 0.013
- **Worst MASE**: 0.406
- **Secondary Metrics**: MAPE 1.42%  1.25%, sMAPE 1.45%  1.27%
- **Success Rate**: 100.0%

## Category Results

### General Merchandise
- **Models Trained**: 7/7
- **Best Model**: LGBM (MASE: 0.379, MAPE: 2.09%, sMAPE: 2.14%)
- **Data Points**: 5814
- **Training Time**: 391.77s

### Sporting Goods Hobby
- **Models Trained**: 7/7
- **Best Model**: LGBM (MASE: 0.406, MAPE: 3.13%, sMAPE: 3.19%)
- **Data Points**: 5814
- **Training Time**: 422.86s

### Furniture Home Furnishings
- **Models Trained**: 7/7
- **Best Model**: LGBM (MASE: 0.029, MAPE: 0.30%, sMAPE: 0.30%)
- **Data Points**: 5814
- **Training Time**: 433.58s

### Building Materials Garden
- **Models Trained**: 7/7
- **Best Model**: LGBM (MASE: 0.013, MAPE: 0.16%, sMAPE: 0.16%)
- **Data Points**: 5814
- **Training Time**: 421.17s


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
