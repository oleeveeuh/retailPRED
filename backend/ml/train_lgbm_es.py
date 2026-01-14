#!/usr/bin/env python3
"""
Train LGBM with Early Stopping to Prevent Overfitting

This script demonstrates how to train LGBM with proper validation
and early stopping to prevent overfitting.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("Warning: LightGBM not installed")

logger = logging.getLogger(__name__)


def train_lgbm_with_early_stopping(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_name: str = "lgbm_model"
):
    """
    Train LGBM with early stopping to prevent overfitting

    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        model_name: Name for saving the model

    Returns:
        Trained model and training metrics
    """

    if not LGBM_AVAILABLE:
        raise ImportError("LightGBM is not installed")

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Anti-overfitting parameters
    params = {
        "objective": "regression",
        "metric": "mae",
        "max_depth": 5,                    # Shallow trees
        "num_leaves": 31,                  # Limit leaves
        "learning_rate": 0.01,             # Lower learning rate
        "min_child_samples": 20,           # Minimum samples per leaf
        "subsample": 0.8,                  # Row sampling
        "colsample_bytree": 0.7,           # Column sampling
        "reg_alpha": 0.1,                  # L1 regularization
        "reg_lambda": 0.1,                 # L2 regularization
        "verbose": -1,
    }

    logger.info(f"Training LGBM with early stopping...")
    logger.info(f"Params: {params}")

    # Train with early stopping
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,              # Maximum iterations
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),  # Stop if no improvement
            lgb.log_evaluation(period=100)           # Log every 100 rounds
        ]
    )

    # Evaluate
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_mae = np.mean(np.abs(y_train - train_pred))
    val_mae = np.mean(np.abs(y_val - val_pred))

    # Calculate overfitting ratio
    overfitting_ratio = val_mae / train_mae

    metrics = {
        "train_mae": train_mae,
        "val_mae": val_mae,
        "overfitting_ratio": overfitting_ratio,
        "best_iteration": model.num_trees(),
    }

    logger.info(f"Training complete!")
    logger.info(f"Train MAE: {train_mae:.2f}")
    logger.info(f"Val MAE: {val_mae:.2f}")
    logger.info(f"Overfitting ratio: {overfitting_ratio:.2f} (lower is better)")

    if overfitting_ratio > 1.5:
        logger.warning("⚠️  Severe overfitting detected!")

    return model, metrics


def main():
    """Example usage"""
    import joblib
    from pathlib import Path

    # Load your data
    # X_train, y_train, X_val, y_val = load_data()

    # Train model
    # model, metrics = train_lgbm_with_early_stopping(X_train, y_train, X_val, y_val)

    # Save model
    # models_dir = Path(__file__).parent / "models"
    # models_dir.mkdir(exist_ok=True)
    # joblib.dump(model, models_dir / "lgbm_model.pkl")

    print("Example script created. Modify the main() function to use your data.")


if __name__ == "__main__":
    main()
