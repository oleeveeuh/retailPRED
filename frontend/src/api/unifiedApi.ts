/**
 * Unified API Layer
 *
 * Automatically switches between demo mode (static JSON) and real API
 * based on VITE_DEMO_MODE environment variable.
 *
 * Usage:
 *   import { api } from '@/api/unifiedApi'
 *   const result = await api.getHistory(filters)
 */

import { config } from '../config/environment';
import { demoDataService } from '../services/demoDataService';
import {
  predictionsApi as realPredictionsApi,
  dataApi as realDataApi,
  modelsApi as realModelsApi,
  categoriesApi as realCategoriesApi,
  systemApi as realSystemApi,
  Granularity,
  ModelType,
} from './client';
import type {
  PredictionHistoryResponse,
  PredictionHistoryItem,
  SHAPExplanationResponse,
  ModelsListResponse,
  CategoriesListResponse,
  HealthResponse,
} from './client';

// ============================================================================
// TYPE UTILITIES
// ============================================================================

type AsyncFunction<TArgs, TReturn> = (...args: TArgs[]) => Promise<TReturn>;

// ============================================================================
// PREDICTIONS API
// ============================================================================

const demoPredictionsApi = {
  /**
   * Get prediction history (demo mode)
   */
  getHistory: async (filters: {
    model_name?: string;
    start_date?: string;
    end_date?: string;
    limit?: number;
  }): Promise<PredictionHistoryResponse> => {
    const result = await demoDataService.getPredictions(filters);

    return {
      predictions: result.predictions,
      total_count: result.total_count,
      filters_applied: result.filters_applied,
    };
  },

  /**
   * Get SHAP explanation (demo mode)
   */
  getSHAPExplanation: async (
    predictionId: number,
    topN?: number
  ): Promise<SHAPExplanationResponse> => {
    const result = await demoDataService.getSHAPValues(predictionId);

    return {
      prediction_id: result.prediction_id,
      model_name: result.model_name,
      prediction_date: result.prediction_date,
      predicted_value: result.predicted_value,
      base_value: 0, // Not available in demo data
      feature_contributions: result.shap_values.map(v => ({
        feature: v.feature,
        value: v.value,
        importance: v.importance,
      })),
      total_shap_value: result.shap_values.reduce((sum, v) => sum + v.importance, 0),
      summary: `Top ${result.shap_values.length} features contributing to prediction`,
    };
  },

  // Other methods not supported in demo mode
  predict: async (...args: any[]) => {
    throw new Error('Predictions are read-only in demo mode');
  },
  validate: async (...args: any[]) => {
    throw new Error('Validation is not available in demo mode');
  },
  autoValidate: async (...args: any[]) => {
    throw new Error('Auto-validation is not available in demo mode');
  },
};

// ============================================================================
// MODELS API
// ============================================================================

const demoModelsApi = {
  /**
   * Get all models (demo mode)
   */
  getAll: async (): Promise<ModelsListResponse> => {
    const summary = await demoDataService.getSummary();

    // Transform summary to model list format
    return {
      models: summary.models_available.models.map((name, index) => ({
        id: index + 1,
        model_name: name,
        model_type: name.split('_').pop() as any,
        training_date: '2025-01-01',
        metrics: {
          rmse: 1000,
          mae: 800,
          r2: 0.94,
          mape: 4.5,
          training_samples: 1000,
        },
        file_path: '/models/' + name,
        is_active: true,
        created_at: '2025-01-01',
        updated_at: '2025-01-01',
      })),
      total_count: summary.models_available.total_count,
      active_count: summary.models_available.total_count,
    };
  },

  // Training not supported in demo mode
  train: async (...args: any[]) => {
    throw new Error('Model training is not available in demo mode');
  },
};

// ============================================================================
// CATEGORIES API
// ============================================================================

const demoCategoriesApi = {
  /**
   * Get all categories (demo mode)
   */
  list: async (): Promise<CategoriesListResponse> => {
    return {
      categories: [
        { key: 'total_sales', display_name: 'Total Retail Sales' },
        { key: 'building_materials', display_name: 'Building Materials & Garden' },
        { key: 'automobile_dealers', display_name: 'Automobile Dealers' },
        { key: 'gasoline_stations', display_name: 'Gasoline Stations' },
        { key: 'food_beverage', display_name: 'Food & Beverage Stores' },
        { key: 'health_personal_care', display_name: 'Health & Personal Care' },
        { key: 'general_merchandise', display_name: 'General Merchandise' },
        { key: 'furniture_home_furnishings', display_name: 'Furniture & Home Furnishings' },
        { key: 'clothing_accessories', display_name: 'Clothing & Accessories' },
        { key: 'sporting_goods_hobby', display_name: 'Sporting Goods & Hobby' },
        { key: 'electronics_and_appliances', display_name: 'Electronics & Appliances' },
      ],
      total_count: 11,
    };
  },

  // Other methods not supported in demo mode
  predict: async (...args: any[]) => {
    throw new Error('Predictions are read-only in demo mode');
  },
  getModels: async (...args: any[]) => {
    throw new Error('This endpoint is not available in demo mode');
  },
};

// ============================================================================
// SYSTEM API
// ============================================================================

const demoSystemApi = {
  /**
   * Health check (demo mode)
   */
  healthCheck: async (): Promise<HealthResponse> => {
    return {
      status: 'ok',
      timestamp: new Date().toISOString(),
      service: 'retailpred-demo',
    };
  },
};

// ============================================================================
// DATA API (not supported in demo mode)
// ============================================================================

const demoDataApi = {
  refresh: async (...args: any[]) => {
    throw new Error('Data refresh is not available in demo mode');
  },
};

// ============================================================================
// UNIFIED API EXPORT
// ============================================================================

/**
 * Complete API object that switches between demo and real implementations
 */
export const api = config.isDemoMode
  ? {
      // Demo mode APIs
      ...demoPredictionsApi,
      ...demoDataApi,
      ...demoModelsApi,
      ...demoCategoriesApi,
      ...demoSystemApi,
    }
  : {
      // Real API implementations
      ...realPredictionsApi,
      ...realDataApi,
      ...realModelsApi,
      ...realCategoriesApi,
      ...realSystemApi,
    };

// Export individual sections for backward compatibility
export const predictionsApi = config.isDemoMode ? demoPredictionsApi : realPredictionsApi;
export const dataApi = config.isDemoMode ? demoDataApi : realDataApi;
export const modelsApi = config.isDemoMode ? demoModelsApi : realModelsApi;
export const categoriesApi = config.isDemoMode ? demoCategoriesApi : realCategoriesApi;
export const systemApi = config.isDemoMode ? demoSystemApi : realSystemApi;

// ============================================================================
// HELPER EXPORTS
// ============================================================================

// Re-export enums for convenience
export { Granularity, ModelType } from './client';

// Re-export types for convenience
export type {
  PredictionRequest,
  PredictionResponse,
  PredictionHistoryItem,
  PredictionHistoryResponse,
  ValidationRequest,
  ValidationResponse,
  SHAPValue,
  SHAPExplanationResponse,
  DataRefreshResponse,
  ModelInfo,
  ModelMetrics,
  ModelsListResponse,
  TrainingRequest,
  TrainingResponse,
  RetailCategory,
  CategoriesListResponse,
  CategoryPredictionRequest,
  CategoryPredictionResponse,
  HealthResponse,
} from './client';

// ============================================================================
// MODE
// ============================================================================

export const isDemoMode = config.isDemoMode;

export default api;
