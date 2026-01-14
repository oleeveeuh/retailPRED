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

    const totalShap = result.shap_values.reduce((sum, v) => sum + v.value, 0);

    return {
      prediction_id: result.prediction_id,
      model_name: result.model_name,
      prediction_date: result.prediction_date,
      predicted_value: result.predicted_value,
      base_value: result.predicted_value - totalShap, // Calculate from predicted value and SHAP values
      feature_contributions: result.shap_values.map(v => ({
        feature: v.feature,
        value: v.value,
        importance: v.importance,
      })),
      total_shap_value: Math.abs(totalShap),
      summary: `Top ${result.shap_values.length} features contributing to prediction`,
    };
  },

  // Generate demo predictions for model comparisons
  predict: async (request: { category: string; model_name: string; weeks_ahead: number }) => {
    // Generate demo predictions for different models
    const baseValue = 600000;

    // Category-specific multipliers
    const categoryMultipliers: Record<string, number> = {
      'total_sales': 1.0,
      'automobile_dealers': 0.15,
      'building_materials': 0.08,
      'clothing_accessories': 0.05,
      'electronics_appliances': 0.07,
      'food_beverage': 0.12,
      'furniture_home': 0.06,
      'gasoline_stations': 0.04,
      'general_merchandise': 0.20,
      'health_personal_care': 0.04,
      'sporting_goods': 0.03,
    };

    // Model-specific performance characteristics
    const modelMultipliers: Record<string, number> = {
      'LGBM': 1.0,
      'RandomForest': 0.98,
      'PatchTST': 1.02,
      'TimesNet': 1.01,
    };

    const categoryMultiplier = categoryMultipliers[request.category] || 1.0;
    const modelMultiplier = modelMultipliers[request.model_name] || 1.0;
    const prediction = baseValue * categoryMultiplier * modelMultiplier;

    return {
      forecasts: [
        {
          category: request.category,
          model_name: request.model_name,
          prediction_date: new Date().toISOString(),
          predicted_value: Math.round(prediction),
          confidence_interval: [
            Math.round(prediction * 0.95),
            Math.round(prediction * 1.05),
          ],
          prediction_horizon: request.weeks_ahead,
        },
      ],
    };
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
   * Using actual training results from robust_training_summary.json
   */
  getModels: async (): Promise<TrainingMetricsResponse> => {
    const summary = await demoDataService.getSummary();

    // Get all model types from summary
    const modelTypes = summary.models_available?.models || [];

    // Real training metrics from validation_metrics.json (MASE ascending, best first)
    const modelMetrics: Record<string, any> = {
      SeasonalNaive: {
        RMSE: { mean: 377 },
        MAE: { mean: 301 },
        R2: 0.90,
        MAPE: { mean: 3.91 },
        SMAPE: { mean: 4.30 },
        MASE: { mean: 1.0000 },
        training_time: 0.06,
        rank: 1
      },

      TimesNet: {
        RMSE: { mean: 372 },
        MAE: { mean: 298 },
        R2: 0.90,
        MAPE: { mean: 3.90 },
        SMAPE: { mean: 4.29 },
        MASE: { mean: 1.0115 },
        training_time: 144.0,
        rank: 4
      },

      AutoARIMA: {
        RMSE: { mean: 412 },
        MAE: { mean: 330 },
        R2: 0.90,
        MAPE: { mean: 3.92 },
        SMAPE: { mean: 4.32 },
        MASE: { mean: 1.0175 },
        training_time: 228.3,
        rank: 5
      },

      PatchTST: {
        RMSE: { mean: 392 },
        MAE: { mean: 313 },
        R2: 0.90,
        MAPE: { mean: 4.01 },
        SMAPE: { mean: 4.42 },
        MASE: { mean: 1.0381 },
        training_time: 14.1,
        rank: 7
      },

      LGBM: {
        RMSE: { mean: 140 },
        MAE: { mean: 112 },
        R2: 0.98,
        MAPE: { mean: 1.8 },
        SMAPE: { mean: 1.9 },
        MASE: { mean: 0.5244 },
        training_time: 3.1,
        rank: 3,
        version: 'v2',
        deployed: '2026-01-11'
      },

      RandomForest: {
        RMSE: { mean: 280 },
        MAE: { mean: 224 },
        R2: 0.98,
        MAPE: { mean: 3.5 },
        SMAPE: { mean: 3.8 },
        MASE: { mean: 0.4919 },
        training_time: 0.9,
        rank: 2,
        version: 'v2',
        deployed: '2026-01-11',
        improvement: '2.69x better MASE'
      },
    };

    // Transform demo data to match training metrics format
    return {
      models: modelTypes.map((modelName, index) => {
        const metrics = modelMetrics[modelName] || {
          RMSE: { mean: 1000 },
          MAE: { mean: 800 },
          R2: 0.90,
          MAPE: { mean: 5.0 },
          SMAPE: { mean: 5.0 },
          MASE: { mean: 1.0 },
          training_time: 10,
          rank: index + 1
        };

        return {
          id: index + 1,
          model_name: modelName,
          model_type: modelName,
          category: 'Total Retail Sales',
          training_date: '2026-01-04',
          metrics: {
            ...metrics,
            mean: 50000,
            std: 5000,
          },
          hyperparameters: {
            learning_rate: 0.01,
            n_estimators: 100,
            cv_samples: 12,
            successful_categories: 4,
          },
          is_active: true,
        };
      }).sort((a, b) => (a.metrics as any).rank - (b.metrics as any).rank),
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
    // Generate demo scenario predictions based on scenario type
    const basePrediction = 600000;

    // Define scenario configurations
    const scenarioConfigs: Record<string, {
      multiplier: number;
      name: string;
      description: string;
      unrate: number;
      fedfunds: number;
      cpi: number;
      payems: number;
      gdp: number;
      consumer_confidence: number;
    }> = {
      baseline: {
        multiplier: 1.0,
        name: 'Baseline',
        description: 'Continue current economic conditions with no changes',
        unrate: 4.2,
        fedfunds: 4.25,
        cpi: 2.8,
        payems: 200000,
        gdp: 2.5,
        consumer_confidence: 100,
      },
      recession: {
        multiplier: 0.85,
        name: 'Recession',
        description: 'Economic downturn with elevated unemployment and negative GDP growth',
        unrate: 6.5,
        fedfunds: 5.5,
        cpi: 3.5,
        payems: 180000,
        gdp: -1.5,
        consumer_confidence: 75,
      },
      optimistic: {
        multiplier: 1.15,
        name: 'Economic Recovery',
        description: 'Strong growth with falling unemployment and rising confidence',
        unrate: 3.5,
        fedfunds: 3.0,
        cpi: 2.2,
        payems: 210000,
        gdp: 3.5,
        consumer_confidence: 115,
      },
      pessimistic: {
        multiplier: 0.82,
        name: 'Deep Recession',
        description: 'Severe economic downturn with very high unemployment and negative GDP growth',
        unrate: 8.0,
        fedfunds: 6.0,
        cpi: 4.0,
        payems: 170000,
        gdp: -2.5,
        consumer_confidence: 65,
      },
      rate_hike: {
        multiplier: 0.92,
        name: 'Rate Hike Cycle',
        description: 'Tightening monetary policy with higher interest rates',
        unrate: 4.8,
        fedfunds: 6.0,
        cpi: 3.2,
        payems: 190000,
        gdp: 1.5,
        consumer_confidence: 85,
      },
      inflation_surge: {
        multiplier: 0.88,
        name: 'Inflation Surge',
        description: 'High inflation environment with elevated consumer prices',
        unrate: 5.2,
        fedfunds: 5.0,
        cpi: 5.5,
        payems: 185000,
        gdp: 0.5,
        consumer_confidence: 80,
      },
      recovery: {
        multiplier: 1.12,
        name: 'Economic Recovery',
        description: 'Strong growth with falling unemployment and rising confidence',
        unrate: 3.8,
        fedfunds: 3.5,
        cpi: 2.5,
        payems: 205000,
        gdp: 3.2,
        consumer_confidence: 110,
      },
    };

    // Get scenario config, default to baseline if not found
    const config = scenarioConfigs[request.scenario_type] || scenarioConfigs.baseline;

    const prediction = basePrediction * config.multiplier;
    const confidence = basePrediction * 0.05;

    // Generate impact summary for different economic indicators
    const impact_summary = [
      {
        indicator: 'UNRATE',
        category: 'Labor Market',
        source: 'BLS',
        base_value: 4.2,
        scenario_value: config.unrate,
        change: config.unrate - 4.2,
        change_pct: ((config.unrate - 4.2) / 4.2) * 100,
      },
      {
        indicator: 'FEDFUNDS',
        category: 'Monetary Policy',
        source: 'FRED',
        base_value: 4.25,
        scenario_value: config.fedfunds,
        change: config.fedfunds - 4.25,
        change_pct: ((config.fedfunds - 4.25) / 4.25) * 100,
      },
      {
        indicator: 'CPI',
        category: 'Consumer',
        source: 'BLS',
        base_value: 2.8,
        scenario_value: config.cpi,
        change: config.cpi - 2.8,
        change_pct: ((config.cpi - 2.8) / 2.8) * 100,
      },
      {
        indicator: 'PAYEMS',
        category: 'Labor Market',
        source: 'BLS',
        base_value: 200000,
        scenario_value: config.payems,
        change: config.payems - 200000,
        change_pct: ((config.payems - 200000) / 200000) * 100,
      },
      {
        indicator: 'GDP',
        category: 'Production',
        source: 'BEA',
        base_value: 2.5,
        scenario_value: config.gdp,
        change: config.gdp - 2.5,
        change_pct: ((config.gdp - 2.5) / 2.5) * 100,
      },
      {
        indicator: 'Consumer Confidence',
        category: 'Consumer',
        source: 'Conference Board',
        base_value: 100,
        scenario_value: config.consumer_confidence,
        change: config.consumer_confidence - 100,
        change_pct: config.consumer_confidence - 100,
      },
    ];

    return {
      scenario_type: request.scenario_type,
      scenario_name: config.name,
      description: config.description,
      category: request.category,
      prediction: Math.round(prediction),
      confidence_interval: [
        Math.round(prediction - confidence),
        Math.round(prediction + confidence),
      ],
      change_from_baseline: config.multiplier !== 1.0 ? Math.round((config.multiplier - 1.0) * 100) : 0,
      impact_summary,
      assumptions: {
        gdp_growth: config.gdp,
        unemployment_rate: config.unrate,
        inflation_rate: config.cpi,
      },
    };
  },

  /**
   * Perform sensitivity analysis (demo mode)
   */
  analyzeSensitivity: async (request: SensitivityAnalysisRequest): Promise<any> => {
    // Generate demo sensitivity data with realistic elasticity for each indicator
    const numPoints = request.num_steps || 10;
    const values_tested = [];
    const predictions = [];

    // Base prediction varies by category to make it more realistic
    const categoryMultipliers: Record<string, number> = {
      'total_sales': 1.0,
      'automobile_dealers': 0.15,
      'building_materials': 0.08,
      'clothing_accessories': 0.05,
      'electronics_appliances': 0.07,
      'food_beverage': 0.12,
      'furniture_home': 0.06,
      'gasoline_stations': 0.04,
      'general_merchandise': 0.20,
      'health_personal_care': 0.04,
      'sporting_goods': 0.03,
    };

    const categoryMultiplier = categoryMultipliers[request.category] || 1.0;
    const basePrediction = 600000 * categoryMultiplier;

    // Define elasticity for each indicator (how much sales change per unit change)
    const elasticities: Record<string, { sensitivity: number; baseline: number; direction: 'negative' | 'positive' }> = {
      'UNRATE': { sensitivity: 0.08, baseline: 4.5, direction: 'negative' },  // Higher unemployment = lower sales
      'FEDFUNDS': { sensitivity: 0.03, baseline: 4.25, direction: 'negative' },  // Higher rates = lower sales
      'CPI': { sensitivity: 0.04, baseline: 2.8, direction: 'negative' },  // Higher inflation = lower sales
      'GDP': { sensitivity: 0.12, baseline: 2.5, direction: 'positive' },  // Higher GDP = higher sales
      'PAYEMS': { sensitivity: 0.0005, baseline: 200, direction: 'positive' },  // More jobs = higher sales (value in thousands)
    };

    const indicatorConfig = elasticities[request.feature_name] || {
      sensitivity: 0.05,
      baseline: (request.min_value + request.max_value) / 2,
      direction: 'negative'
    };

    for (let i = 0; i <= numPoints; i++) {
      const value = request.min_value + (request.max_value - request.min_value) * (i / numPoints);
      values_tested.push(value);

      // Calculate impact based on deviation from baseline
      const deviation = value - indicatorConfig.baseline;

      // Apply elasticity with direction
      let impact;
      if (indicatorConfig.direction === 'negative') {
        impact = 1.0 - (deviation * indicatorConfig.sensitivity);
      } else {
        impact = 1.0 + (deviation * indicatorConfig.sensitivity);
      }

      // Clamp impact to reasonable bounds (0.5x to 1.5x)
      impact = Math.max(0.5, Math.min(1.5, impact));

      predictions.push(Math.round(basePrediction * impact));
    }

    const predictionRange = Math.max(...predictions) - Math.min(...predictions);
    const predictionMean = predictions.reduce((sum, val) => sum + val, 0) / predictions.length;
    const baselineValue = indicatorConfig.baseline;

    // Calculate actual elasticity (percent change in sales / percent change in indicator)
    const baselinePrediction = predictions[Math.floor(numPoints / 2)];
    const percentChangeInSales = ((predictions[numPoints] - predictions[0]) / baselinePrediction) * 100;
    const percentChangeInIndicator = ((values_tested[numPoints] - values_tested[0]) / baselineValue) * 100 || 1;
    const elasticity = percentChangeInSales / percentChangeInIndicator;

    return {
      feature_name: request.feature_name,
      predictions,
      values: values_tested,
      prediction_range: predictionRange,
      min_prediction: Math.min(...predictions),
      max_prediction: Math.max(...predictions),
      prediction_mean: predictionMean,
      baseline_value: baselineValue,
      elasticity: Number(elasticity.toFixed(3)),
      category: request.category,
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

  /**
   * Analyze scenario for a specific model (demo mode)
   */
  analyzeModelScenario: async (request: {
    scenario_type: string;
    category: string;
    model_name: string;
  }): Promise<any> => {
    // Generate different base predictions for each model to make them distinct
    const modelBasePredictions: Record<string, number> = {
      LGBM: 600000,
      RandomForest: 595000,
      PatchTST: 605000,
      TimesNet: 598000,
    };

    const basePrediction = modelBasePredictions[request.model_name] || 600000;

    // Define scenario configurations
    const scenarioConfigs: Record<string, {
      multiplier: number;
      name: string;
      description: string;
      unrate: number;
      fedfunds: number;
      cpi: number;
      payems: number;
      gdp: number;
      consumer_confidence: number;
    }> = {
      baseline: {
        multiplier: 1.01,
        name: 'Baseline',
        description: 'Continue current economic conditions with modest growth',
        unrate: 4.2,
        fedfunds: 4.25,
        cpi: 2.8,
        payems: 200000,
        gdp: 2.5,
        consumer_confidence: 100,
      },
      recession: {
        multiplier: 0.92,
        name: 'Recession',
        description: 'Economic downturn with elevated unemployment and negative GDP growth',
        unrate: 6.5,
        fedfunds: 3.5,
        cpi: 2.0,
        payems: 180000,
        gdp: -1.5,
        consumer_confidence: 75,
      },
      rate_hike: {
        multiplier: 0.97,
        name: 'Rate Hike Cycle',
        description: 'Tightening monetary policy with higher interest rates',
        unrate: 4.7,
        fedfunds: 6.25,
        cpi: 2.3,
        payems: 195000,
        gdp: 2.0,
        consumer_confidence: 90,
      },
      inflation_surge: {
        multiplier: 0.98,
        name: 'Inflation Surge',
        description: 'High inflation environment with elevated consumer prices',
        unrate: 4.5,
        fedfunds: 5.75,
        cpi: 4.8,
        payems: 190000,
        gdp: 2.2,
        consumer_confidence: 85,
      },
      recovery: {
        multiplier: 1.06,
        name: 'Economic Recovery',
        description: 'Strong growth with falling unemployment and rising confidence',
        unrate: 3.8,
        fedfunds: 3.5,
        cpi: 2.5,
        payems: 205000,
        gdp: 3.2,
        consumer_confidence: 110,
      },
    };

    // Get scenario config, default to baseline if not found
    const config = scenarioConfigs[request.scenario_type] || scenarioConfigs.baseline;

    const prediction = basePrediction * config.multiplier;
    const confidence = basePrediction * 0.05;

    // Generate impact summary for different economic indicators
    const impact_summary = [
      {
        indicator: 'UNRATE',
        category: 'Labor Market',
        source: 'BLS',
        base_value: 4.2,
        scenario_value: config.unrate,
        change: config.unrate - 4.2,
        change_pct: ((config.unrate - 4.2) / 4.2) * 100,
      },
      {
        indicator: 'FEDFUNDS',
        category: 'Monetary Policy',
        source: 'FRED',
        base_value: 4.25,
        scenario_value: config.fedfunds,
        change: config.fedfunds - 4.25,
        change_pct: ((config.fedfunds - 4.25) / 4.25) * 100,
      },
      {
        indicator: 'CPI',
        category: 'Consumer',
        source: 'BLS',
        base_value: 2.8,
        scenario_value: config.cpi,
        change: config.cpi - 2.8,
        change_pct: ((config.cpi - 2.8) / 2.8) * 100,
      },
      {
        indicator: 'PAYEMS',
        category: 'Labor Market',
        source: 'BLS',
        base_value: 200000,
        scenario_value: config.payems,
        change: config.payems - 200000,
        change_pct: ((config.payems - 200000) / 200000) * 100,
      },
      {
        indicator: 'GDP',
        category: 'Production',
        source: 'BEA',
        base_value: 2.5,
        scenario_value: config.gdp,
        change: config.gdp - 2.5,
        change_pct: ((config.gdp - 2.5) / 2.5) * 100,
      },
    ];

    return {
      scenario_type: request.scenario_type,
      scenario_name: config.name,
      description: config.description,
      category: request.category,
      model_name: request.model_name,
      prediction: Math.round(prediction),
      base_prediction: Math.round(basePrediction),
      confidence_interval: [
        Math.round(prediction - confidence),
        Math.round(prediction + confidence),
      ],
      change_from_baseline: config.multiplier !== 1.0 ? Math.round((config.multiplier - 1.0) * 100) : 0,
      impact_summary,
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
