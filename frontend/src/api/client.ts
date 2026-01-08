/**
 * RetailPRED API Client
 * Typed Axios client for all backend endpoints
 */

import axios, { AxiosError } from 'axios';

// Use environment variable if set, otherwise default to localhost
// Empty string is valid (no backend), only use localhost if undefined
const API_BASE_URL = import.meta.env.VITE_API_URL !== undefined
  ? import.meta.env.VITE_API_URL
  : 'http://localhost:8000';

// Create axios instance
// Note: Don't add /api prefix here - backend routers already include it in their prefix
export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 120000, // Increased to 120 seconds for long-running predictions
});

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    // In demo mode with empty API URL, prevent actual API calls
    const isDemoMode = import.meta.env.VITE_DEMO_MODE === 'true';
    const apiBaseUrl = import.meta.env.VITE_API_URL;

    if (isDemoMode && apiBaseUrl === '') {
      // Reject the request immediately in demo mode
      return Promise.reject(new Error('Demo mode active - API calls disabled'));
    }

    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
apiClient.interceptors.response.use(
  (response) => {
    return response;
  },
  (error: AxiosError) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// ============================================================================
// TYPES
// ============================================================================

// Model Types
export enum ModelType {
  RANDOM_FOREST = 'RandomForest',
  XGBOOST = 'XGBoost',
  LIGHTGBM = 'LightGBM',
  PATCHTST = 'PatchTST',
  TIMEGPT = 'TimeGPT',
}

export enum Granularity {
  DAILY = 'daily',
  WEEKLY = 'weekly',
  MONTHLY = 'monthly',
}

// Prediction Types
export interface SHAPValue {
  feature: string;
  value: number;
  importance: number;
}

export interface ForecastPoint {
  date: string;
  predicted_value: number;
  confidence_lower?: number;
  confidence_upper?: number;
}

export interface PredictionRequest {
  category?: string;
  store_id?: number;
  product_id?: number;
  weeks_ahead: number;
  model_name?: string;
  granularity?: Granularity;
}

export interface PredictionResponse {
  prediction_id: number;
  model_name: string;
  model_type: string;
  store_id?: number;
  product_id?: number;
  forecasts: ForecastPoint[];
  shap_values: SHAPValue[];
  features_used: Record<string, any>;
  created_at: string;
  metadata?: Record<string, any>;
}

// Data Refresh Types
export interface DataRefreshResponse {
  status: string;
  message: string;
  records_updated: number;
  new_categories?: number;
  last_fetch_time: string;
  sources_updated: string[];
}

// Model Types
export interface ModelMetrics {
  rmse: number;
  mae: number;
  r2: number;
  mape?: number;
  training_samples: number;
}

export interface ModelInfo {
  id: number;
  model_name: string;
  model_type: ModelType;
  training_date: string;
  metrics: ModelMetrics;
  hyperparameters?: Record<string, any>;
  file_path: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface ModelsListResponse {
  models: ModelInfo[];
  total_count: number;
  active_count: number;
}

// Prediction History Types
export interface PredictionHistoryItem {
  id: number;
  model_name: string;
  store_id?: number;
  product_id?: number;
  prediction_date: string;
  predicted_value: number;
  actual_value?: number;
  confidence_interval_lower?: number;
  confidence_interval_upper?: number;
  is_validated: boolean;
  error_percentage?: number;
  created_at: string;
}

export interface PredictionHistoryResponse {
  predictions: PredictionHistoryItem[];
  total_count: number;
  filters_applied: Record<string, any>;
  accuracy_summary?: {
    avg_error_percentage: number;
    min_error_percentage: number;
    max_error_percentage: number;
    total_validated: number;
  };
}

// Validation Types
export interface ValidationRequest {
  prediction_id: number;
  actual_value: number;
  notes?: string;
}

export interface ValidationResponse {
  prediction_id: number;
  previous_predicted_value: number;
  new_actual_value: number;
  error_absolute: number;
  error_percentage: number;
  is_validated: boolean;
  message: string;
}

// SHAP Explanation Types
export interface SHAPExplanationResponse {
  prediction_id: number;
  model_name: string;
  prediction_date: string;
  predicted_value: number;
  base_value: number;
  feature_contributions: SHAPValue[];
  total_shap_value: number;
  summary: string;
}

// Training Types
export interface TrainingRequest {
  model_types?: ModelType[];
  force_retrain?: boolean;
  test_size?: number;
  hyperparameters?: Record<string, Record<string, any>>;
}

export interface TrainingResponse {
  status: string;
  models_trained: string[];
  training_time_seconds: number;
  metrics: Record<string, ModelMetrics>;
  message: string;
}

// Health Check
export interface HealthResponse {
  status: string;
  timestamp: string;
  service: string;
}

// Category Types
export interface RetailCategory {
  key: string;
  display_name: string;
}

export interface CategoriesListResponse {
  categories: RetailCategory[];
  total_count: number;
}

export interface CategoryPredictionRequest {
  category: string;
  model_type: string;
  features: Record<string, any>;
}

export interface CategoryPredictionResponse {
  category: string;
  category_display_name: string;
  model_name: string;
  model_type: string;
  predicted_value: number;
  shap_values: SHAPValue[];
  features_used: Record<string, any>;
  metadata: Record<string, any>;
}

// ============================================================================
// API FUNCTIONS
// ============================================================================

/**
 * PREDICTIONS
 */
export const predictionsApi = {
  /**
   * Make a sales forecast prediction
   */
  predict: async (params: PredictionRequest): Promise<PredictionResponse> => {
    const response = await apiClient.get<PredictionResponse>('/predict', {
      params,
    });
    return response.data;
  },

  /**
   * Get prediction history
   */
  getHistory: async (filters: {
    model_name?: string;
    store_id?: number;
    product_id?: number;
    start_date?: string;
    end_date?: string;
    include_validated_only?: boolean;
    limit?: number;
  }): Promise<PredictionHistoryResponse> => {
    const response = await apiClient.get<PredictionHistoryResponse>(
      '/predictions/history',
      { params: filters }
    );
    return response.data;
  },

  /**
   * Validate a prediction with actual value
   */
  validate: async (data: ValidationRequest): Promise<ValidationResponse> => {
    const response = await apiClient.post<ValidationResponse>(
      '/predictions/validate',
      data
    );
    return response.data;
  },

  /**
   * Auto-validate predictions by fetching actual values from database
   */
  autoValidate: async (params: { category_id?: string; days_back?: number }): Promise<ValidationResponse[]> => {
    const response = await apiClient.post<ValidationResponse[]>(
      '/predictions/auto-validate',
      null,
      { params }
    );
    return response.data;
  },

  /**
   * Get SHAP explanation for a prediction
   */
  getSHAPExplanation: async (
    predictionId: number,
    topN?: number
  ): Promise<SHAPExplanationResponse> => {
    const response = await apiClient.get<SHAPExplanationResponse>(
      '/shap-explain',
      {
        params: { prediction_id: predictionId, top_n: topN },
      }
    );
    return response.data;
  },
};

/**
 * DATA MANAGEMENT
 */
export const dataApi = {
  /**
   * Refresh data from external sources
   */
  refresh: async (): Promise<DataRefreshResponse> => {
    const response = await apiClient.post<DataRefreshResponse>('/refresh-data');
    return response.data;
  },
};

/**
 * MODELS
 */
export const modelsApi = {
  /**
   * Get all models
   */
  getAll: async (params?: {
    active_only?: boolean;
    model_type?: ModelType;
  }): Promise<ModelsListResponse> => {
    const response = await apiClient.get<ModelsListResponse>('/models', {
      params,
    });
    return response.data;
  },

  /**
   * Train new models
   */
  train: async (data: TrainingRequest): Promise<TrainingResponse> => {
    const response = await apiClient.post<TrainingResponse>('/train', data);
    return response.data;
  },
};

/**
 * CATEGORIES
 */
export const categoriesApi = {
  /**
   * Get all available retail categories
   */
  list: async (): Promise<CategoriesListResponse> => {
    const response = await apiClient.get<CategoriesListResponse>('/categories/list');
    return response.data;
  },

  /**
   * Make a prediction for a specific category
   */
  predict: async (data: CategoryPredictionRequest): Promise<CategoryPredictionResponse> => {
    const response = await apiClient.post<CategoryPredictionResponse>('/categories/predict', data);
    return response.data;
  },

  /**
   * Get available models for a category
   */
  getModels: async (category: string): Promise<{category: string; category_display_name: string; available_models: string[]; total_count: number}> => {
    const response = await apiClient.get(`/categories/${category}/models`);
    return response.data;
  },
};

/**
 * TRAINING METRICS
 */
export interface TrainingMetricsModel {
  id: number;
  model_name: string;
  category: string;
  training_date: string;
  metrics: {
    RMSE?: number;
    MAE?: number;
    R2?: number;
    MAPE?: number;
    mean?: number;
    std?: number;
  };
  hyperparameters?: Record<string, any>;
  is_active: boolean;
}

export interface TrainingMetricsResponse {
  models: TrainingMetricsModel[];
  total_count: number;
  active_count: number;
}

export const trainingMetricsApi = {
  /**
   * Get all training metrics for models
   */
  getModels: async (): Promise<TrainingMetricsResponse> => {
    const response = await apiClient.get<TrainingMetricsResponse>('/training-metrics/models');
    return response.data;
  },
};

/**
 * ECONOMIC INDICATORS
 */
export interface EconomicIndicator {
  name: string;
  display: string;
  value: number;
  previousValue: number;
  unit: string;
  category: string;
  source: string;
  lead_lag: 'leading' | 'coincident' | 'lagging';
  status: 'healthy' | 'warning' | 'alert';
  date: string;
}

export interface EconomicIndicatorsResponse {
  indicators: EconomicIndicator[];
  last_updated: string;
  total_count: number;
}

export const economicIndicatorsApi = {
  /**
   * Get current economic indicators
   */
  getCurrent: async (): Promise<EconomicIndicatorsResponse> => {
    const response = await apiClient.get<EconomicIndicatorsResponse>('/economic-indicators/current');
    return response.data;
  },
};

/**
 * HISTORICAL SALES & SCENARIOS
 */
export interface HistoricalSalesPoint {
  date: string;
  value: number;
}

export interface HistoricalSalesResponse {
  data: HistoricalSalesPoint[];
  category: string;
  days_back: number;
}

export interface ScenarioAnalysisRequest {
  scenario_type: 'baseline' | 'optimistic' | 'pessimistic';
  category: string;
  custom_params?: Record<string, number>;
}

export interface ScenarioAnalysisResponse {
  scenario_type: string;
  category: string;
  prediction: number;
  confidence_interval: [number, number];
  change_from_baseline?: number;
  assumptions: Record<string, number>;
}

export interface SensitivityAnalysisRequest {
  category: string;
  feature_name: string;
  min_value: number;
  max_value: number;
  num_steps?: number;
}

export interface SensitivityAnalysisResponse {
  feature_name: string;
  predictions: number[];
  values: number[];
  prediction_range: number;
  min_prediction: number;
  max_prediction: number;
  elasticity?: number;
}

export const scenariosApi = {
  /**
   * Get historical sales data
   */
  getHistoricalSales: async (category: string, days_back: number = 365): Promise<HistoricalSalesResponse> => {
    const response = await apiClient.get<HistoricalSalesResponse>('/historical-sales', {
      params: { category, days_back }
    });
    return response.data;
  },

  /**
   * Analyze a scenario
   */
  analyzeScenario: async (request: ScenarioAnalysisRequest): Promise<ScenarioAnalysisResponse> => {
    const response = await apiClient.post<ScenarioAnalysisResponse>('/scenarios/analyze', request);
    return response.data;
  },

  /**
   * Perform sensitivity analysis
   */
  analyzeSensitivity: async (request: SensitivityAnalysisRequest): Promise<SensitivityAnalysisResponse> => {
    const response = await apiClient.post<SensitivityAnalysisResponse>('/scenarios/sensitivity', request);
    return response.data;
  },

  /**
   * Get similar historical periods
   */
  getSimilarPeriods: async (category: string, n: number = 5): Promise<any> => {
    const response = await apiClient.get<any>('/scenarios/similar-periods', {
      params: { category, n }
    });
    return response.data;
  },

  /**
   * Get regime analysis
   */
  getRegime: async (category: string): Promise<any> => {
    const response = await apiClient.get<any>('/scenarios/regime', {
      params: { category }
    });
    return response.data;
  },
};

/**
 * EXPORT
 */
export interface ExportCSVResponse {
  status: string;
  file_path: string;
  records: number;
}

export const exportApi = {
  /**
   * Export predictions to CSV
   */
  exportPredictionsCSV: async (): Promise<ExportCSVResponse> => {
    const response = await apiClient.get<ExportCSVResponse>('/export/predictions-csv');
    return response.data;
  },
};

/**
 * SYSTEM
 */
export const systemApi = {
  /**
   * Health check
   */
  healthCheck: async (): Promise<HealthResponse> => {
    const response = await apiClient.get<HealthResponse>('/health');
    return response.data;
  },
};

// ============================================================================
// CONVENIENCE EXPORTS
// ============================================================================

export const api = {
  ...predictionsApi,
  ...dataApi,
  ...modelsApi,
  ...categoriesApi,
  ...trainingMetricsApi,
  ...economicIndicatorsApi,
  ...scenariosApi,
  ...exportApi,
  ...systemApi,
};

export default apiClient;
