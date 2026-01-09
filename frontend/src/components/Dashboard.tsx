/**
 * Dashboard Component
 * Overview page with key metrics and visualizations
 * Includes economic context for anomaly interpretation
 */

import { FC, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { predictionsApi, trainingMetricsApi } from '../api/unifiedApi';
import ForecastChart from './ForecastChart';
import FeatureImportanceChart from './FeatureImportanceChart';
import ModelInfoCard from './ModelInfoCard';
import { EconomicRegimeIndicator } from './EconomicRegimeIndicator';
import { EconomicContextInfo } from './EconomicContextInfo';
import { useCurrentRegime } from '@/hooks/useEconomicRegime';

export const Dashboard: FC = () => {
  const [selectedModel, setSelectedModel] = useState<string>('all');

  // Get current economic regime
  const { regime: economicRegime, loading: regimeLoading } = useCurrentRegime();

  // Fetch predictions for metrics (fetch with high limit to get accurate total)
  const { data: historyData, isLoading: predictionsLoading } = useQuery({
    queryKey: ['recentPredictions'],
    queryFn: () => predictionsApi.getHistory({ limit: 15000 }),
  });

  // Fetch models with actual training metrics
  const { data: modelsData, isLoading: modelsLoading, error: modelsError } = useQuery({
    queryKey: ['training-models'],
    queryFn: () => trainingMetricsApi.getModels(),
    retry: 2,
  });

  // Calculate summary metrics
  const summaryMetrics = {
    totalPredictions: historyData?.total_count || 0,
    activeModels: modelsData?.active_count || 0,
    avgAccuracy: modelsData?.models
      ? (() => {
          const avgMape = modelsData.models.reduce((sum: number, m: any) => {
            const mape = m?.metrics?.MAPE?.mean || 0;
            return sum + mape;
          }, 0) / modelsData.models.length;
          return 100 - avgMape;
        })()
      : 0,
  };

  if (predictionsLoading || modelsLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
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
        <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg shadow p-6 text-white">
          <p className="text-sm opacity-90">Total Predictions</p>
          <p className="text-4xl font-bold mt-2">{summaryMetrics.totalPredictions}</p>
          <p className="text-sm opacity-75 mt-2">All time</p>
        </div>

        <div className="bg-gradient-to-br from-green-500 to-green-600 rounded-lg shadow p-6 text-white">
          <p className="text-sm opacity-90">Active Models</p>
          <p className="text-4xl font-bold mt-2">{summaryMetrics.activeModels}</p>
          <p className="text-sm opacity-75 mt-2">Currently deployed</p>
        </div>

        <div className="bg-gradient-to-br from-purple-500 to-purple-600 rounded-lg shadow p-6 text-white">
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
