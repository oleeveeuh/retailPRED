-- RetailPRED Database Schema Extension
-- This file extends the existing database with prediction tracking tables

-- Prediction Log Table
-- Tracks all predictions made by the system
CREATE TABLE IF NOT EXISTS prediction_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    store_id INTEGER,
    product_id INTEGER,
    prediction_date TEXT NOT NULL,
    predicted_value REAL NOT NULL,
    actual_value REAL,
    confidence_interval_lower REAL,
    confidence_interval_upper REAL,
    features TEXT, -- JSON string containing feature values used for prediction
    shap_values TEXT, -- JSON string containing SHAP values for explainability
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Model Metadata Table
-- Tracks trained models and their metadata
CREATE TABLE IF NOT EXISTS model_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL UNIQUE,
    model_type TEXT NOT NULL, -- e.g., 'RandomForest', 'XGBoost', 'LightGBM'
    training_date TEXT NOT NULL,
    metrics TEXT NOT NULL, -- JSON string containing accuracy metrics (RMSE, MAE, R2, etc.)
    hyperparameters TEXT, -- JSON string containing model hyperparameters
    file_path TEXT NOT NULL, -- Path to the serialized model file
    is_active INTEGER DEFAULT 1, -- Boolean flag for active model
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_prediction_log_model_name ON prediction_log(model_name);
CREATE INDEX IF NOT EXISTS idx_prediction_log_prediction_date ON prediction_log(prediction_date);
CREATE INDEX IF NOT EXISTS idx_prediction_log_store_id ON prediction_log(store_id);
CREATE INDEX IF NOT EXISTS idx_prediction_log_product_id ON prediction_log(product_id);
CREATE INDEX IF NOT EXISTS idx_prediction_log_created_at ON prediction_log(created_at);
CREATE INDEX IF NOT EXISTS idx_model_metadata_model_name ON model_metadata(model_name);
CREATE INDEX IF NOT EXISTS idx_model_metadata_is_active ON model_metadata(is_active);

-- Trigger to update updated_at timestamp on model_metadata
CREATE TRIGGER IF NOT EXISTS update_model_metadata_timestamp
AFTER UPDATE ON model_metadata
FOR EACH ROW
BEGIN
    UPDATE model_metadata SET updated_at = datetime('now') WHERE id = NEW.id;
END;

-- View for active models
CREATE VIEW IF NOT EXISTS active_models AS
SELECT
    id,
    model_name,
    model_type,
    training_date,
    metrics,
    hyperparameters,
    file_path,
    created_at,
    updated_at
FROM model_metadata
WHERE is_active = 1;

-- View for prediction accuracy analysis
CREATE VIEW IF NOT EXISTS prediction_accuracy AS
SELECT
    model_name,
    store_id,
    product_id,
    COUNT(*) as total_predictions,
    COUNT(actual_value) as predictions_with_actuals,
    AVG(predicted_value) as avg_predicted_value,
    AVG(actual_value) as avg_actual_value,
    SUM(CASE WHEN actual_value IS NOT NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as actual_value_coverage_pct
FROM prediction_log
GROUP BY model_name, store_id, product_id;
