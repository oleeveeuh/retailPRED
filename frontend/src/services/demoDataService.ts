/**
 * Demo Data Service
 *
 * Loads data from static JSON files in /demo-data/ for demo mode.
 * Mimics API response structure and adds realistic delays.
 */

import type { SHAPValue, PredictionHistoryItem } from '../api/client';

// ============================================================================
// TYPES
// ============================================================================

export interface DemoPrediction {
  id: number;
  model_name: string;
  prediction_date: string;
  predicted_value: number;
  actual_value: number | null;
  confidence_interval_lower: number | null;
  confidence_interval_upper: number | null;
  shap_values: Record<string, number> | null;
  features: string | null;
  created_at: string;
}

export interface DemoEconomicIndicator {
  date: string;
  cpi: number;
  interest_rates: number;
  unemployment: number;
  consumer_sentiment: number;
  money_supply: number;
  industrial_production: number;
}

export interface DemoDataResponse<T> {
  data: T[];
  metadata: {
    export_timestamp: string;
    row_count: number;
    [key: string]: any;
  };
}

export interface DemoSummary {
  export_timestamp: string;
  database_path: string;
  predictions: {
    total_count: number;
    by_year: Record<string, number>;
    by_model_type: Record<string, number>;
    shap_coverage: number;
  };
  models_available: {
    total_count: number;
    models: string[];
  };
  demo_data: {
    predictions_included: number;
    economic_indicators_included: number;
    note: string;
  };
}

// ============================================================================
// SERVICE CLASS
// ============================================================================

class DemoDataService {
  private cache: Map<string, any> = new Map();
  private readonly SIMULATED_DELAY_MS = 300; // 300ms delay to simulate API

  /**
   * Simulate network delay
   */
  private async delay(ms: number = this.SIMULATED_DELAY_MS): Promise<void> {
    await new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Load JSON file from public directory
   */
  private async loadJSON<T>(filename: string): Promise<T> {
    // Check cache first
    if (this.cache.has(filename)) {
      return this.cache.get(filename);
    }

    try {
      const response = await fetch(`/demo-data/${filename}`);
      if (!response.ok) {
        throw new Error(`Failed to load ${filename}: ${response.statusText}`);
      }
      const data = await response.json();

      // Cache the result
      this.cache.set(filename, data);
      return data;
    } catch (error) {
      console.error(`Error loading demo data from ${filename}:`, error);
      throw error;
    }
  }

  /**
   * Get predictions with optional filters
   */
  async getPredictions(filters?: {
    model_name?: string;
    start_date?: string;
    end_date?: string;
    limit?: number;
  }): Promise<{
    predictions: PredictionHistoryItem[];
    total_count: number;
    filters_applied: Record<string, any>;
  }> {
    await this.delay();

    const response = await this.loadJSON<DemoDataResponse<DemoPrediction>>('predictions.json');

    let predictions = response.data;

    // Apply filters
    if (filters?.model_name) {
      predictions = predictions.filter(p =>
        p.model_name.toLowerCase().includes(filters.model_name!.toLowerCase())
      );
    }

    if (filters?.start_date) {
      predictions = predictions.filter(p => p.prediction_date >= filters.start_date!);
    }

    if (filters?.end_date) {
      predictions = predictions.filter(p => p.prediction_date <= filters.end_date!);
    }

    if (filters?.limit) {
      predictions = predictions.slice(0, filters.limit);
    }

    // Transform to API format
    const transformedPredictions: PredictionHistoryItem[] = predictions.map(p => ({
      id: p.id,
      model_name: p.model_name,
      prediction_date: p.prediction_date,
      predicted_value: p.predicted_value,
      actual_value: p.actual_value ?? undefined,
      confidence_interval_lower: p.confidence_interval_lower ?? undefined,
      confidence_interval_upper: p.confidence_interval_upper ?? undefined,
      error_absolute: p.error_absolute,
      error_percentage: p.error_percentage,
      confidence_score: p.confidence_score,
      is_validated: p.actual_value !== null,
      created_at: p.created_at,
    }));

    return {
      predictions: transformedPredictions,
      total_count: response.metadata.total_predictions_in_db,
      filters_applied: filters || {},
    };
  }

  /**
   * Get economic indicators
   */
  async getEconomicIndicators(): Promise<{
    data: DemoEconomicIndicator[];
    metadata: DemoDataResponse<DemoEconomicIndicator>['metadata'];
  }> {
    await this.delay();

    const response = await this.loadJSON<DemoDataResponse<DemoEconomicIndicator>>('economic-indicators.json');

    return {
      data: response.data,
      metadata: response.metadata,
    };
  }

  /**
   * Get summary statistics
   */
  async getSummary(): Promise<DemoSummary> {
    await this.delay();

    return await this.loadJSON<DemoSummary>('summary.json');
  }

  /**
   * Get SHAP values for a specific prediction
   */
  async getSHAPValues(predictionId: number): Promise<{
    prediction_id: number;
    model_name: string;
    prediction_date: string;
    predicted_value: number;
    shap_values: SHAPValue[];
  }> {
    await this.delay();

    const response = await this.loadJSON<DemoDataResponse<DemoPrediction>>('predictions.json');
    const prediction = response.data.find(p => p.id === predictionId);

    if (!prediction || !prediction.shap_values) {
      throw new Error(`No SHAP values found for prediction ${predictionId}`);
    }

    // Transform SHAP values to API format
    const shapArray: SHAPValue[] = Object.entries(prediction.shap_values)
      .map(([feature, value]) => ({
        feature,
        value: Math.abs(value),
        importance: Math.abs(value),
      }))
      .sort((a, b) => b.importance - a.importance)
      .slice(0, 20); // Top 20 features

    return {
      prediction_id: prediction.id,
      model_name: prediction.model_name,
      prediction_date: prediction.prediction_date,
      predicted_value: prediction.predicted_value,
      shap_values: shapArray,
    };
  }

  /**
   * Clear cache (useful for testing)
   */
  clearCache(): void {
    this.cache.clear();
  }
}

// ============================================================================
// EXPORTS
// ============================================================================

export const demoDataService = new DemoDataService();
export default demoDataService;
