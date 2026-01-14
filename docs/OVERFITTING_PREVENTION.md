# Overfitting Prevention Guide for Tree-Based Models

## Problem Analysis

**Current Issue**: LGBM and RandomForest are severely overfitting:
- **Training MASE**: LGBM (0.207), RandomForest (0.285) -看起来很棒
- **Validation MASE**: LGBM (1.5684), RandomForest (3.7187) -实际表现糟糕
- **Gap**: 7.6x to 13x degradation from training to validation!

## Root Causes

1. **Insufficient Regularization**: Default hyperparameters allow trees to grow too deep
2. **No Early Stopping**: Training continues until completion without validation checks
3. **Data Leakage**: Features may contain future information
4. **Small Dataset**: Only 5814 data points per category
5. **High Dimensionality**: Too many features relative to samples

## Solutions

### 1. RandomForest Hyperparameter Tuning

**Current Settings (BAD)**:
```python
params = {
    "n_estimators": 100,
    "max_depth": 10,           # Too deep!
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42,
}
```

**Recommended Settings**:
```python
params = {
    "n_estimators": 200,              # More trees (but limited depth)
    "max_depth": 5,                   # SHALLOWER trees to prevent overfitting
    "min_samples_split": 20,          # More samples required to split
    "min_samples_leaf": 10,           # More samples required at leaf nodes
    "max_features": 0.7,              # Use only 70% of features per tree
    "bootstrap": True,                # Bootstrap sampling
    "oob_score": True,                # Out-of-bag scoring for validation
    "random_state": 42,
    "n_jobs": -1,
}
```

**Key Changes**:
- `max_depth: 10 → 5`: Reduce tree depth by 50%
- `min_samples_split: 5 → 20`: Require 4x more samples to split
- `min_samples_leaf: 2 → 10`: Require 5x more samples at leaves
- `max_features: 0.7`: Feature bagging reduces correlation between trees
- `oob_score: True`: Use out-of-bag samples as built-in validation

### 2. LGBM Hyperparameter Tuning

**Add to config.yaml or training script**:
```python
params = {
    "n_estimators": 500,
    "max_depth": 5,                    # Shallow trees
    "num_leaves": 31,                  # Limit leaves (should be < 2^max_depth)
    "learning_rate": 0.01,             # Lower learning rate
    "min_child_samples": 20,           # Minimum samples per leaf
    "subsample": 0.8,                  # Row sampling (80%)
    "colsample_bytree": 0.7,           # Column sampling (70%)
    "reg_alpha": 0.1,                  # L1 regularization
    "reg_lambda": 0.1,                 # L2 regularization
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}
```

**Critical Parameters**:
- `max_depth: 5`: Limit tree complexity
- `learning_rate: 0.01`: Slower learning = better generalization
- `min_child_samples: 20`: Prevent leaves with few samples
- `reg_alpha` & `reg_lambda`: Add regularization penalties

### 3. Early Stopping (CRITICAL!)

**Implementation**:
```python
def train_lgbm_with_early_stopping(X_train, y_train, X_val, y_val):
    """Train LGBM with early stopping to prevent overfitting"""

    import lightgbm as lgb

    # Create validation dataset for early stopping
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        "objective": "regression",
        "metric": "mae",
        "max_depth": 5,
        "learning_rate": 0.01,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
    }

    # Train with early stopping
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,           # Maximum iterations
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),  # Stop if no improvement for 50 rounds
            lgb.log_evaluation(period=100)           # Log every 100 rounds
        ]
    )

    return model
```

**Early Stopping Benefits**:
- Stops training when validation performance stops improving
- Prevents memorization of training data
- Automatically finds optimal number of iterations

### 4. Cross-Validation

**Use Time Series Cross-Validation**:
```python
from sklearn.model_selection import TimeSeriesSplit

def train_with_cv(X, y, model_type='lgbm'):
    """Train with time series cross-validation"""

    tscv = TimeSeriesSplit(n_splits=5)

    cv_scores = []
    models = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Train model
        if model_type == 'lgbm':
            model = train_lgbm_with_early_stopping(X_train, y_train, X_val, y_val)
        else:
            model = train_rf_with_params(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_val)
        mase = calculate_mase(y_val, y_pred, baseline_naive)
        cv_scores.append(mase)
        models.append(model)

    # Select best model
    best_idx = np.argmin(cv_scores)
    return models[best_idx], np.mean(cv_scores)
```

### 5. Feature Engineering Improvements

**Problem**: Too many features can cause overfitting

**Solution**:
```python
def select_important_features(X, y, importance_threshold=0.01):
    """Select only important features to reduce dimensionality"""

    from sklearn.ensemble import RandomForestRegressor

    # Train a quick RF to get feature importances
    rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    rf.fit(X, y)

    # Get feature importances
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    })

    # Select features above threshold
    important_features = importances[
        importances['importance'] >= importance_threshold
    ]['feature'].tolist()

    print(f"Selected {len(important_features)}/{len(X.columns)} features")

    return X[important_features], important_features
```

### 6. Data Augmentation

**Increase training data size**:
```python
def augment_time_series_data(df, window_size=4):
    """Create additional training samples using sliding windows"""

    augmented_data = []

    for i in range(len(df) - window_size):
        # Create window-based samples
        window = df.iloc[i:i+window_size]
        target = df.iloc[i+window_size]

        augmented_data.append({
            **window.to_dict(),
            'target': target
        })

    return pd.DataFrame(augmented_data)
```

### 7. Hyperparameter Optimization

**Use Optuna for automated tuning**:
```python
import optuna

def objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective for LGBM"""

    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'num_leaves': trial.suggest_int('num_leaves', 20, 50),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 0.5),
    }

    model = train_lgbm_with_params(X_train, y_train, X_val, y_val, params)
    y_pred = model.predict(X_val)
    mase = calculate_mase(y_val, y_pred)

    return mase

# Run optimization
study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=50)

best_params = study.best_params
print(f"Best MASE: {study.best_value:.4f}")
print(f"Best params: {best_params}")
```

## Implementation Plan

### Phase 1: Quick Fixes (Immediate)

1. **Update RandomForest params** in `backend/ml/train_model.py:131`:
   - `max_depth: 10 → 5`
   - `min_samples_split: 5 → 20`
   - `min_samples_leaf: 2 → 10`

2. **Add LGBM params** to training script (if not exists):
   - `max_depth: 5`
   - `learning_rate: 0.01`
   - `min_child_samples: 20`

### Phase 2: Early Stopping (High Priority)

3. **Implement early stopping** for LGBM:
   - Add validation split (80/20)
   - Use `lgb.early_stopping(stopping_rounds=50)`
   - Monitor validation MAE during training

4. **Add out-of-bag scoring** for RandomForest:
   - Set `oob_score=True`
   - Use OOB score as validation metric

### Phase 3: Advanced Techniques (Medium Priority)

5. **Implement time series cross-validation**:
   - Use `TimeSeriesSplit(n_splits=5)`
   - Train multiple models and select best

6. **Feature selection**:
   - Remove low-importance features
   - Reduce dimensionality from 25 → 15 features

### Phase 4: Hyperparameter Tuning (Long-term)

7. **Implement Optuna optimization**:
   - Automated hyperparameter search
   - Find optimal regularization settings

## Expected Results

**Before (Current)**:
- RandomForest: MASE 0.285 (train) → 3.7187 (val) = 13x worse
- LGBM: MASE 0.207 (train) → 1.5684 (val) = 7.6x worse

**After (Target)**:
- RandomForest: MASE 0.8-1.2 (train) → 0.9-1.3 (val) = < 1.5x gap
- LGBM: MASE 0.7-1.0 (train) → 0.9-1.2 (val) = < 1.3x gap

**Success Criteria**:
- Validation MASE < 1.3 for both models
- Train-validation gap < 30%
- Consistent performance across all categories

## Monitoring

**Track these metrics during training**:
```python
training_metrics = {
    'train_mae': [],
    'val_mae': [],
    'train_mase': [],
    'val_mase': [],
    'overfitting_ratio': [],  # val_mase / train_mase
}

# Warning threshold
if overfitting_ratio > 1.5:
    logger.warning(f"Severe overfitting detected! Ratio: {overfitting_ratio:.2f}")
```

## Next Steps

1. ✅ Update RandomForest hyperparameters in `train_model.py`
2. ✅ Add LGBM early stopping implementation
3. ✅ Create training script with cross-validation
4. ⏳ Retrain all models with new settings
5. ⏳ Compare validation metrics
6. ⏳ If still overfitting, implement Phase 3 (feature selection, data augmentation)
