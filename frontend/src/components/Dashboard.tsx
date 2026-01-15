/**
 * Dashboard Component
 * Overview page with key metrics and visualizations
 * Includes economic context for anomaly interpretation
 */

import { FC, useState } from 'react';
import { Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { predictionsApi, trainingMetricsApi } from '../api/unifiedApi';
import ForecastChart from './ForecastChart';
import FeatureImportanceChart from './FeatureImportanceChart';
import ModelInfoCard from './ModelInfoCard';
import { EconomicRegimeIndicator } from './EconomicRegimeIndicator';
import { EconomicContextInfo } from './EconomicContextInfo';
import { AnomalyExplanation } from './AnomalyExplanation';
import { useCurrentRegime } from '@/hooks/useEconomicRegime';
import { useAnomalyDetection } from '@/hooks/useAnomalyDetection';
import { AlertTriangle } from 'lucide-react';

export const Dashboard: FC = () => {
  const [selectedModel, setSelectedModel] = useState<string>('all');

  // Get current economic regime
  const { regime: economicRegime, loading: regimeLoading } = useCurrentRegime();

  // Fetch predictions for metrics (fetch with high limit to get accurate total)
  const { data: historyData, isLoading: predictionsLoading, error: predictionsError } = useQuery({
    queryKey: ['recentPredictions'],
    queryFn: async () => {
      console.log('Fetching predictions...');
      const result = await predictionsApi.getHistory({ limit: 15000 });
      console.log('Predictions fetched:', result);
      return result;
    },
    retry: 2,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });

  // Fetch models with actual training metrics
  const { data: modelsData, isLoading: modelsLoading, error: modelsError } = useQuery({
    queryKey: ['training-models'],
    queryFn: () => trainingMetricsApi.getModels(),
    retry: 2,
  });

  if (predictionsLoading || modelsLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
      </div>
    );
  }

  // Debug logging (only after loading is complete)
  console.log('Dashboard historyData:', historyData);
  console.log('Dashboard predictions:', historyData?.predictions);
  console.log('Dashboard predictions length:', historyData?.predictions?.length || 0);
  console.log('Dashboard modelsData:', modelsData);

  // Calculate summary metrics from actual predictions (only after data is loaded)
  const predictionsArray = historyData?.predictions || [];

  // Debug: Check first few predictions
  console.log('First 3 predictions sample:', predictionsArray.slice(0, 3).map(p => ({
    id: p.id,
    actual_value: p.actual_value,
    error_percentage: p.error_percentage,
    hasActual: p.actual_value != null,
    hasError: p.error_percentage != null,
  })));

  const summaryMetrics = {
    totalPredictions: predictionsArray.length,
    activeModels: modelsData?.active_count || 0,
    avgAccuracy: (() => {
      if (predictionsArray.length === 0) {
        console.log('No predictions array, returning 0 accuracy');
        return 0;
      }

      // Calculate accuracy from actual prediction validation
      const validatedPredictions = predictionsArray.filter((p: any) => {
        const hasActual = p.actual_value != null; // Use != to catch both null and undefined
        const hasError = p.error_percentage != null;
        return hasActual && hasError;
      });

      console.log('Total predictions:', predictionsArray.length);
      console.log('Validated predictions:', validatedPredictions.length);

      if (validatedPredictions.length === 0) {
        console.log('No validated predictions, returning 0 accuracy');
        return 0;
      }

      const avgError = validatedPredictions.reduce((sum: number, p: any) => sum + (p.error_percentage || 0), 0) / validatedPredictions.length;
      const accuracy = 100 - avgError;
      console.log('Average error:', avgError, 'Accuracy:', accuracy);
      return accuracy;
    })(),
  };

  console.log('Dashboard summaryMetrics:', summaryMetrics);

  if (predictionsError) {
    console.error('Predictions error:', predictionsError);
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
          <p className="text-gray-600 mt-1">Overview of your retail forecasting system</p>
        </div>
        <div className="bg-red-50 border border-red-200 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-red-900 mb-2">Error Loading Prediction Data</h3>
          <p className="text-red-700">Unable to fetch prediction history. Please refresh the page and try again.</p>
          {predictionsError instanceof Error && <p className="text-red-600 text-sm mt-2">Error: {predictionsError.message}</p>}
        </div>
      </div>
    );
  }

  if (modelsError) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
          <p className="text-gray-600 mt-1">Overview of your retail forecasting system</p>
        </div>
        <div className="bg-red-50 border border-red-200 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-red-900 mb-2">Error Loading Model Data</h3>
          <p className="text-red-700">Unable to fetch training metrics. Please ensure the backend server is running and try again.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-gray-600 mt-1">
          Overview of your retail forecasting system
        </p>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-gradient-to-br from-primary to-primary-600 rounded-lg shadow p-6 text-white">
          <p className="text-sm opacity-90">Total Predictions</p>
          <p className="text-4xl font-bold mt-2">{summaryMetrics.totalPredictions}</p>
          <p className="text-sm opacity-75 mt-2">All time</p>
        </div>

        <div className="bg-gradient-to-br from-green-500 to-green-600 rounded-lg shadow p-6 text-white">
          <p className="text-sm opacity-90">Active Models</p>
          <p className="text-4xl font-bold mt-2">{summaryMetrics.activeModels}</p>
          <p className="text-sm opacity-75 mt-2">Currently deployed</p>
        </div>

        <div className="bg-gradient-to-br from-accent to-accent rounded-lg shadow p-6 text-white">
          <p className="text-sm opacity-90">Avg. Accuracy</p>
          <p className="text-4xl font-bold mt-2">{summaryMetrics.avgAccuracy.toFixed(1)}%</p>
          <p className="text-sm opacity-75 mt-2">Last 30 days</p>
        </div>
      </div>

      {/* Economic Regime Indicator */}
      {!regimeLoading && economicRegime && (
        <EconomicRegimeIndicator
          regime={economicRegime}
          showExplanation={true}
        />
      )}

      {/* Economic Context Info */}
      <EconomicContextInfo />

      {/* Recent Anomalies Alert */}
      {(import.meta.env.VITE_DEMO_MODE === 'true' || (historyData && historyData.predictions.length > 1)) && (() => {
        // Demo mode: always show alert
        if (import.meta.env.VITE_DEMO_MODE === 'true') {
          return (
            <div className="bg-gradient-to-r from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 border border-orange-200 dark:border-orange-800 rounded-lg p-6">
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-orange-500 rounded-lg">
                    <AlertTriangle className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-orange-900 dark:text-orange-100">
                      Recent Anomalies Detected
                    </h3>
                    <p className="text-sm text-orange-700 dark:text-orange-300 mt-1">
                      3 unusual predictions found in recent forecasts
                    </p>
                  </div>
                </div>
                <Link
                  to="/dashboard/anomalies"
                  className="px-4 py-2 bg-orange-600 hover:bg-orange-700 text-white rounded-lg text-sm font-medium transition-colors"
                >
                  View All
                </Link>
              </div>
            </div>
          );
        }

        // Production mode: check real data
        const recentPredictions = historyData.predictions.slice(0, 20);
        const hasPredictedValue = recentPredictions[0]?.predicted_value !== undefined;

        const anomalies = recentPredictions
          .map((pred, i) => {
            if (i === 0) return null;
            const prev = recentPredictions[i - 1];
            const currentValue = hasPredictedValue ? pred.predicted_value : pred.value;
            const previousValue = hasPredictedValue ? prev.predicted_value : prev.value;

            if (!currentValue || !previousValue) return null;

            const change = Math.abs(((currentValue - previousValue) / previousValue) * 100);
            return { prediction: pred, change: Math.abs(change) };
          })
          .filter(item => item && item.change > 5)
          .slice(0, 3);

        if (anomalies.length === 0) return null;

        return (
          <div className="bg-gradient-to-r from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 border border-orange-200 dark:border-orange-800 rounded-lg p-6">
            <div className="flex items-start justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-orange-500 rounded-lg">
                  <AlertTriangle className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-orange-900 dark:text-orange-100">
                    Recent Anomalies Detected
                  </h3>
                  <p className="text-sm text-orange-700 dark:text-orange-300 mt-1">
                    {anomalies.length} unusual prediction{anomalies.length > 1 ? 's' : ''} found in recent forecasts
                  </p>
                </div>
              </div>
              <Link
                to="/dashboard/anomalies"
                className="px-4 py-2 bg-orange-600 hover:bg-orange-700 text-white rounded-lg text-sm font-medium transition-colors"
              >
                View All
              </Link>
            </div>
          </div>
        );
      })()}

      {/* Forecast Chart */}
      <ForecastChart />

      {/* Model Comparison */}
      <div>
        <h2 className="text-2xl font-semibold text-gray-800 mb-4">All Models</h2>
        {modelsData?.models && modelsData.models.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {modelsData.models
              .map((model: any) => (
                <ModelInfoCard key={model.id} model={model} />
              ))}
          </div>
        ) : (
          <div className="bg-gray-50 border border-gray-200 rounded-lg p-6">
            <p className="text-gray-600">No model data available. Please train models first.</p>
          </div>
        )}
      </div>

      {/* Feature Importance */}
      <FeatureImportanceChart />

      {/* Recent Activity */}
      {historyData && historyData.predictions.length > 0 && (
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-2xl font-semibold text-gray-800 mb-4">Recent Predictions</h2>
          <div className="space-y-3">
            {historyData.predictions.slice(0, 5).map((prediction) => (
              <div key={prediction.id} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                <div>
                  <p className="font-medium text-gray-900">
                    {prediction.model_name}
                  </p>
                  <p className="text-sm text-gray-600">
                    {prediction.prediction_date}
                  </p>
                </div>
                <div className="text-right">
                  <p className="text-lg font-semibold text-gray-900">
                    {'$' + prediction.predicted_value.toFixed(2)}
                  </p>
                  {prediction.actual_value && (
                    <p className="text-sm text-gray-600">
                      Actual: {'$' + prediction.actual_value.toFixed(2)}
                    </p>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
