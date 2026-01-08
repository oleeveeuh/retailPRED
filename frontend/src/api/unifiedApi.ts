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
// Real API imports disabled for Vercel demo deployment
// Uncomment these to enable real backend API calls
// import {
//   predictionsApi as realPredictionsApi,
//   dataApi as realDataApi,
//   modelsApi as realModelsApi,
//   categoriesApi as realCategoriesApi,
//   trainingMetricsApi as realTrainingMetricsApi,
//   economicIndicatorsApi as realEconomicIndicatorsApi,
//   scenariosApi as realScenariosApi,
//   exportApi as realExportApi,
//   systemApi as realSystemApi,
//   Granularity,
//   ModelType,
// } from './client';
import { Granularity, ModelType } from './client';
import type {
  PredictionHistoryResponse,
  PredictionHistoryItem,
  SHAPExplanationResponse,
  ModelsListResponse,
  CategoriesListResponse,
  HealthResponse,
  TrainingMetricsResponse,
  EconomicIndicatorsResponse,
  HistoricalSalesResponse,
  ScenarioAnalysisRequest,
  ScenarioAnalysisResponse,
  SensitivityAnalysisRequest,
  SensitivityAnalysisResponse,
  ExportCSVResponse,
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
        model_type: name as any, // Use the name directly since models don't have underscores
        training_date: '2025-01-01',
        metrics: {
          rmse: 1000 + Math.random() * 500,
          mae: 800 + Math.random() * 400,
          r2: 0.92 + Math.random() * 0.07,
          mape: 3 + Math.random() * 5,
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
// TRAINING METRICS API
// ============================================================================

const demoTrainingMetricsApi = {
  /**
   * Get training metrics for models (demo mode)
   */
  getModels: async (): Promise<TrainingMetricsResponse> => {
    const summary = await demoDataService.getSummary();

    // Get all model types from summary
    const modelTypes = summary.models_available?.models || [];

    // Transform demo data to match training metrics format
    return {
      models: modelTypes.map((modelName, index) => ({
        id: index + 1,
        model_name: modelName,
        model_type: modelName, // Add model_type field for ModelsPage
        category: 'Total Retail Sales',
        training_date: '2025-01-01',
        metrics: {
          RMSE: { mean: 1000 + Math.random() * 500 },
          MAE: { mean: 800 + Math.random() * 400 },
          R2: 0.92 + Math.random() * 0.07,
          MAPE: { mean: 3 + Math.random() * 5 },
          SMAPE: { mean: 2 + Math.random() * 4 },
          mean: 50000,
          std: 5000,
          training_time: 10 + Math.random() * 20,
        },
        hyperparameters: {
          learning_rate: 0.01,
          n_estimators: 100,
        },
        is_active: true,
      })),
      total_count: modelTypes.length,
      active_count: modelTypes.length,
    };
  },
};

// ============================================================================
// ECONOMIC INDICATORS API
// ============================================================================

const demoEconomicIndicatorsApi = {
  /**
   * Get current economic indicators (demo mode)
   */
  getCurrent: async (): Promise<EconomicIndicatorsResponse> => {
    // Return demo data for economic indicators
    return {
      indicators: [
        {
          name: 'UNRATE',
          display: 'Unemployment Rate',
          value: 4.2,
          previousValue: 4.1,
          unit: '%',
          category: 'Labor Market',
          source: 'BLS',
          lead_lag: 'lagging',
          status: 'healthy',
          date: '2025-01-01',
        },
        {
          name: 'CPI',
          display: 'Consumer Price Index',
          value: 2.8,
          previousValue: 3.1,
          unit: '% YoY',
          category: 'Consumer',
          source: 'BLS',
          lead_lag: 'lagging',
          status: 'healthy',
          date: '2025-01-01',
        },
        {
          name: 'FEDFUNDS',
          display: 'Federal Funds Rate',
          value: 4.25,
          previousValue: 4.50,
          unit: '%',
          category: 'Monetary Policy',
          source: 'FRED',
          lead_lag: 'leading',
          status: 'healthy',
          date: '2025-01-01',
        },
        {
          name: 'PAYEMS',
          display: 'Nonfarm Payrolls',
          value: 200000,
          previousValue: 175000,
          unit: 'thousands',
          category: 'Labor Market',
          source: 'BLS',
          lead_lag: 'coincident',
          status: 'healthy',
          date: '2025-01-01',
        },
      ],
      last_updated: new Date().toISOString(),
      total_count: 4,
    };
  },
};

// ============================================================================
// SCENARIOS API
// ============================================================================

const demoScenariosApi = {
  /**
   * Get historical sales (demo mode)
   */
  getHistoricalSales: async (category: string, days_back: number = 365): Promise<HistoricalSalesResponse> => {
    // Generate demo historical sales data
    const data = [];
    const today = new Date();

    for (let i = days_back; i >= 0; i--) {
      const date = new Date(today);
      date.setDate(date.getDate() - i);

      // Generate realistic retail sales data with trend and seasonality
      const baseValue = 500000;
      const trend = (days_back - i) * 100; // Slight upward trend
      const seasonality = Math.sin(i / 30) * 50000; // Monthly seasonality
      const noise = Math.random() * 20000 - 10000;

      data.push({
        date: date.toISOString().split('T')[0],
        value: Math.round(baseValue + trend + seasonality + noise),
      });
    }

    return {
      data,
      category,
      days_back,
    };
  },

  /**
   * Analyze scenario (demo mode)
   */
  analyzeScenario: async (request: ScenarioAnalysisRequest): Promise<any> => {
    // Generate demo scenario predictions
    const basePrediction = 600000;
    const scenarioMultiplier =
      request.scenario_type === 'optimistic' ? 1.1 :
      request.scenario_type === 'pessimistic' ? 0.9 : 1.0;

    const prediction = basePrediction * scenarioMultiplier;
    const confidence = basePrediction * 0.05;

    // Generate impact summary for different economic indicators
    const impact_summary = [
      {
        indicator: 'UNRATE',
        category: 'Labor Market',
        source: 'BLS',
        base_value: 4.2,
        scenario_value: request.scenario_type === 'optimistic' ? 3.8 : request.scenario_type === 'pessimistic' ? 5.5 : 4.2,
        change: request.scenario_type === 'optimistic' ? -0.4 : request.scenario_type === 'pessimistic' ? 1.3 : 0,
        change_pct: request.scenario_type === 'optimistic' ? -9.5 : request.scenario_type === 'pessimistic' ? 31.0 : 0,
      },
      {
        indicator: 'FEDFUNDS',
        category: 'Monetary Policy',
        source: 'FRED',
        base_value: 4.25,
        scenario_value: request.scenario_type === 'optimistic' ? 3.5 : request.scenario_type === 'pessimistic' ? 5.5 : 4.25,
        change: request.scenario_type === 'optimistic' ? -0.75 : request.scenario_type === 'pessimistic' ? 1.25 : 0,
        change_pct: request.scenario_type === 'optimistic' ? -17.6 : request.scenario_type === 'pessimistic' ? 29.4 : 0,
      },
      {
        indicator: 'CPI',
        category: 'Consumer',
        source: 'BLS',
        base_value: 2.8,
        scenario_value: request.scenario_type === 'optimistic' ? 2.2 : request.scenario_type === 'pessimistic' ? 4.0 : 2.8,
        change: request.scenario_type === 'optimistic' ? -0.6 : request.scenario_type === 'pessimistic' ? 1.2 : 0,
        change_pct: request.scenario_type === 'optimistic' ? -21.4 : request.scenario_type === 'pessimistic' ? 42.9 : 0,
      },
      {
        indicator: 'PAYEMS',
        category: 'Labor Market',
        source: 'BLS',
        base_value: 200000,
        scenario_value: request.scenario_type === 'optimistic' ? 250000 : request.scenario_type === 'pessimistic' ? 150000 : 200000,
        change: request.scenario_type === 'optimistic' ? 50000 : request.scenario_type === 'pessimistic' ? -50000 : 0,
        change_pct: request.scenario_type === 'optimistic' ? 25.0 : request.scenario_type === 'pessimistic' ? -25.0 : 0,
      },
      {
        indicator: 'GDP',
        category: 'Production',
        source: 'BEA',
        base_value: 2.5,
        scenario_value: request.scenario_type === 'optimistic' ? 3.5 : request.scenario_type === 'pessimistic' ? -0.5 : 2.5,
        change: request.scenario_type === 'optimistic' ? 1.0 : request.scenario_type === 'pessimistic' ? -3.0 : 0,
        change_pct: request.scenario_type === 'optimistic' ? 40.0 : request.scenario_type === 'pessimistic' ? -120.0 : 0,
      },
      {
        indicator: 'Consumer Confidence',
        category: 'Consumer',
        source: 'Conference Board',
        base_value: 100,
        scenario_value: request.scenario_type === 'optimistic' ? 115 : request.scenario_type === 'pessimistic' ? 75 : 100,
        change: request.scenario_type === 'optimistic' ? 15 : request.scenario_type === 'pessimistic' ? -25 : 0,
        change_pct: request.scenario_type === 'optimistic' ? 15.0 : request.scenario_type === 'pessimistic' ? -25.0 : 0,
      },
    ];

    return {
      scenario_type: request.scenario_type,
      scenario_name: request.scenario_type === 'optimistic' ? 'Optimistic' :
                     request.scenario_type === 'pessimistic' ? 'Pessimistic' : 'Baseline',
      description: request.scenario_type === 'optimistic' ? 'Strong economic growth scenario' :
                   request.scenario_type === 'pessimistic' ? 'Economic downturn scenario' : 'Baseline economic conditions',
      category: request.category,
      prediction: Math.round(prediction),
      confidence_interval: [
        Math.round(prediction - confidence),
        Math.round(prediction + confidence),
      ],
      change_from_baseline: scenarioMultiplier !== 1.0 ? Math.round((scenarioMultiplier - 1.0) * 100) : undefined,
      impact_summary,
      assumptions: {
        gdp_growth: request.scenario_type === 'optimistic' ? 2.5 : request.scenario_type === 'pessimistic' ? 1.0 : 2.0,
        unemployment_rate: request.scenario_type === 'optimistic' ? 4.0 : request.scenario_type === 'pessimistic' ? 5.5 : 4.5,
        inflation_rate: 2.5,
      },
    };
  },

  /**
   * Perform sensitivity analysis (demo mode)
   */
  analyzeSensitivity: async (request: SensitivityAnalysisRequest): Promise<any> => {
    // Generate demo sensitivity data
    const numPoints = request.num_steps || 10;
    const values_tested = [];
    const predictions = [];

    for (let i = 0; i <= numPoints; i++) {
      const value = request.min_value + (request.max_value - request.min_value) * (i / numPoints);
      values_tested.push(value);

      // Simulate sensitivity: unemployment/gdp affect sales negatively/positively
      let impact = 1.0;
      if (request.feature_name === 'UNRATE') {
        impact = 1.0 - (value - 4.5) * 0.05; // Higher unemployment = lower sales
      } else if (request.feature_name === 'GDP') {
        impact = 1.0 + (value - 2.0) * 0.1; // Higher GDP = higher sales
      } else if (request.feature_name === 'FEDFUNDS') {
        impact = 1.0 - (value - 2.0) * 0.02; // Higher rates = lower sales
      } else if (request.feature_name === 'CPI') {
        impact = 1.0 - (value - 2.5) * 0.03; // Higher inflation = lower sales
      } else if (request.feature_name === 'PAYEMS') {
        impact = 1.0 + (value - 200) * 0.0001; // More jobs = higher sales
      }

      predictions.push(Math.round(600000 * impact));
    }

    const predictionRange = Math.max(...predictions) - Math.min(...predictions);
    const predictionMean = predictions.reduce((sum, val) => sum + val, 0) / predictions.length;
    const baselineValue = (request.min_value + request.max_value) / 2;

    return {
      feature_name: request.feature_name,
      predictions,
      values: values_tested,
      prediction_range: predictionRange,
      min_prediction: Math.min(...predictions),
      max_prediction: Math.max(...predictions),
      prediction_mean: predictionMean,
      baseline_value: baselineValue,
      elasticity: -0.5, // Demo value
    };
  },

  /**
   * Get similar periods (demo mode)
   */
  getSimilarPeriods: async (category: string, n: number = 5): Promise<any> => {
    // Return demo similar periods with indicators
    const demoPeriods = [
      {
        date: '2020-03-01',
        start_date: '2020-03-01',
        end_date: '2020-12-31',
        similarity_score: 0.92,
        description: 'COVID-19 pandemic',
        indicators: {
          UNRATE: 8.4,
          GDP: -2.8,
          CPI: 258.8,
          FEDFUNDS: 0.09
        },
        retail_sales: 485000
      },
      {
        date: '2008-09-01',
        start_date: '2008-09-01',
        end_date: '2009-06-30',
        similarity_score: 0.87,
        description: 'Financial crisis',
        indicators: {
          UNRATE: 7.3,
          GDP: -2.6,
          CPI: 218.8,
          FEDFUNDS: 1.92
        },
        retail_sales: 422000
      },
      {
        date: '2001-03-01',
        start_date: '2001-03-01',
        end_date: '2001-11-30',
        similarity_score: 0.78,
        description: 'Dot-com bubble',
        indicators: {
          UNRATE: 4.7,
          GDP: 1.0,
          CPI: 176.5,
          FEDFUNDS: 3.88
        },
        retail_sales: 389000
      },
      {
        date: '2019-01-01',
        start_date: '2019-01-01',
        end_date: '2019-12-31',
        similarity_score: 0.75,
        description: 'Pre-pandemic stability',
        indicators: {
          UNRATE: 3.7,
          GDP: 2.3,
          CPI: 255.2,
          FEDFUNDS: 2.16
        },
        retail_sales: 521000
      },
      {
        date: '2022-01-01',
        start_date: '2022-01-01',
        end_date: '2022-12-31',
        similarity_score: 0.71,
        description: 'Post-pandemic recovery',
        indicators: {
          UNRATE: 3.6,
          GDP: 2.1,
          CPI: 292.0,
          FEDFUNDS: 4.10
        },
        retail_sales: 612000
      },
    ];
    return {
      periods: demoPeriods.slice(0, n),
      total_count: n,
    };
  },

  /**
   * Get regime analysis (demo mode)
   */
  getRegime: async (category: string): Promise<any> => {
    // Return demo regime analysis
    return {
      regime: 'expansion',
      confidence: 0.85,
      characteristics: {
        growth_rate: 2.5,
        volatility: 0.15,
        trend: 'positive',
      },
      description: 'Current economic expansion phase with moderate growth',
    };
  },
};

// ============================================================================
// EXPORT API
// ============================================================================

const demoExportApi = {
  /**
   * Export predictions to CSV (demo mode - not supported)
   */
  exportPredictionsCSV: async (): Promise<ExportCSVResponse> => {
    throw new Error('CSV export is not available in demo mode');
  },
};

// ============================================================================
// UNIFIED API EXPORT
// ============================================================================

/**
 * Complete API object - demo mode for Vercel deployment
 */

// For production builds on Vercel, always use demo mode
// The real API implementations are not imported to prevent them from being included in the bundle

export const api = {
  ...demoPredictionsApi,
  ...demoDataApi,
  ...demoModelsApi,
  ...demoCategoriesApi,
  ...demoTrainingMetricsApi,
  ...demoEconomicIndicatorsApi,
  ...demoScenariosApi,
  ...demoExportApi,
  ...demoSystemApi,
};

// Export individual sections for backward compatibility
export const predictionsApi = demoPredictionsApi;
export const dataApi = demoDataApi;
export const modelsApi = demoModelsApi;
export const categoriesApi = demoCategoriesApi;
export const trainingMetricsApi = demoTrainingMetricsApi;
export const economicIndicatorsApi = demoEconomicIndicatorsApi;
export const scenariosApi = demoScenariosApi;
export const exportApi = demoExportApi;
export const systemApi = demoSystemApi;

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
  TrainingMetricsResponse,
  TrainingMetricsModel,
  EconomicIndicatorsResponse,
  EconomicIndicator,
  HistoricalSalesResponse,
  ScenarioAnalysisRequest,
  ScenarioAnalysisResponse,
  SensitivityAnalysisRequest,
  SensitivityAnalysisResponse,
  ExportCSVResponse,
} from './client';

// ============================================================================
// MODE
// ============================================================================

export const isDemoMode = config.isDemoMode;

export default api;
