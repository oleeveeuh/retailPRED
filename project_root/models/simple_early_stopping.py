"""
Simple Early Stopping Implementation

A lightweight early stopping utility for machine learning models.
Provides early stopping during hyperparameter tuning to prevent overfitting
and optimize training time.
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def train_with_simple_early_stopping(
    model: Any,
    param_grid: List[Dict[str, Any]],
    X_train: Any,
    y_train: Any,
    X_val: Any,
    y_val: Any,
    metric_func: Callable[[Any, Any, Any], float],
    patience: int = 3,
    minimize_metric: bool = True,
    verbose: bool = True
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train a model with early stopping during hyperparameter search.

    Args:
        model: Model class or instance that supports fit/predict
        param_grid: List of parameter dictionaries to try
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        metric_func: Function that calculates performance metric
        patience: Number of iterations without improvement before stopping
        minimize_metric: Whether to minimize (True) or maximize (False) the metric
        verbose: Whether to print progress information

    Returns:
        Tuple of (best_model, best_params_dict)
    """
    logger.info(f" {model.__class__.__name__}: Starting training with early stopping")
    logger.info(f" {model.__class__.__name__}: Train size: {len(X_train)}, Val size: {len(X_val)}")

    best_model = None
    best_score = float('inf') if minimize_metric else float('-inf')
    best_params = None
    no_improvement_count = 0
    start_time = time.time()

    for i, params in enumerate(param_grid, 1):
        try:
            # Create new model instance with current parameters
            current_model = model.__class__(**params)

            # Train the model
            current_model.fit(X_train, y_train)

            # Make predictions on validation set
            y_pred = current_model.predict(X_val)

            # Calculate metric
            score = metric_func(y_val, y_pred)

            if verbose:
                logger.info(f"   Params {i}: {params} -> MAPE: {score:.4f}")

            # Check if this is the best score so far
            is_better = (minimize_metric and score < best_score) or (not minimize_metric and score > best_score)

            if is_better:
                best_score = score
                best_model = current_model
                best_params = params.copy()
                no_improvement_count = 0
                logger.info(f"    New best MAPE: {score:.4f}")
            else:
                no_improvement_count += 1
                logger.info(f"    No improvement ({no_improvement_count}/{patience})")

            # Early stopping check
            if no_improvement_count >= patience:
                logger.info(f"    Early stopping triggered after {i} parameter sets")
                break

        except Exception as e:
            logger.warning(f"    Error with params {params}: {e}")
            continue

    training_time = time.time() - start_time

    if best_model is not None:
        logger.info(f"    Final best MAPE: {best_score:.4f}")
        logger.info(f" {model.__class__.__name__}: Training completed successfully")
        logger.info(f" {model.__class__.__name__}: Best MAPE: {best_score:.3%}")
        logger.info(f" {model.__class__.__name__}: Best params: {best_params}")
        logger.info(f" {model.__class__.__name__}: Training time: {training_time:.2f}s")
    else:
        logger.warning(f" {model.__class__.__name__}: No successful training completed")
        # Return last attempted model as fallback
        if param_grid:
            best_model = model.__class__(**param_grid[0])
            best_model.fit(X_train, y_train)
            best_params = param_grid[0]

    return best_model, {
        'best_params': best_params,
        'best_score': best_score,
        'training_time': training_time,
        'trials_completed': min(i, len(param_grid))
    }


def create_param_grid(base_params: Dict[str, Any], param_ranges: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Create a parameter grid by combining base parameters with ranges to explore.

    Args:
        base_params: Base parameters that are fixed
        param_ranges: Dictionary of parameter names to list of values to try

    Returns:
        List of parameter dictionaries
    """
    import itertools

    # Create all combinations of parameter ranges
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())

    combinations = list(itertools.product(*param_values))

    # Create parameter dictionaries
    param_grid = []
    for combo in combinations:
        params = base_params.copy()
        for name, value in zip(param_names, combo):
            params[name] = value
        param_grid.append(params)

    return param_grid


# Utility function for MAPE (commonly used in this project)
def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error."""
    import numpy as np
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Avoid division by zero
    mask = y_true != 0
    if not np.any(mask):
        return float('inf')

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


# Common parameter grids for different model types
COMMON_PARAM_GRIDS = {
    'random_forest': [
        {'n_estimators': 50, 'max_depth': 5, 'random_state': 42, 'n_jobs': -1},
        {'n_estimators': 100, 'max_depth': 8, 'random_state': 42, 'n_jobs': -1},
        {'n_estimators': 150, 'max_depth': 10, 'random_state': 42, 'n_jobs': -1},
        {'n_estimators': 200, 'max_depth': 12, 'random_state': 42, 'n_jobs': -1},
        {'n_estimators': 250, 'max_depth': 15, 'random_state': 42, 'n_jobs': -1},
    ],

    'lgbm': [
        {'n_estimators': 50, 'max_depth': 5, 'learning_rate': 0.1, 'random_state': 42, 'n_jobs': -1},
        {'n_estimators': 100, 'max_depth': 8, 'learning_rate': 0.1, 'random_state': 42, 'n_jobs': -1},
        {'n_estimators': 150, 'max_depth': 10, 'learning_rate': 0.1, 'random_state': 42, 'n_jobs': -1},
        {'n_estimators': 200, 'max_depth': 12, 'learning_rate': 0.1, 'random_state': 42, 'n_jobs': -1},
        {'n_estimators': 100, 'max_depth': 8, 'learning_rate': 0.05, 'random_state': 42, 'n_jobs': -1},
    ]
}