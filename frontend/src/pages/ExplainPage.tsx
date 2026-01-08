/**
 * Explainability Page
 * SHAP visualization with counterfactual analysis
 */

import { FC, useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { predictionsApi } from '../api/unifiedApi';
import { ShapWaterfall } from '../components/ShapWaterfall';
import type { SHAPWaterfallData } from '../components/ShapWaterfall';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts';

const COLORS = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#06b6d4', '#ef4444'];

interface FeatureRow {
  feature: string;
  value: number;
  importance: number;
  percentage: number;
  direction: 'positive' | 'negative';
}

export const ExplainPage: FC = () => {
  const [selectedPredictionId, setSelectedPredictionId] = useState<number | ''>('');
  const [selectedTimestamp, setSelectedTimestamp] = useState<string>('');
  const [topN, setTopN] = useState<number>(10);
  const [sortField, setSortField] = useState<'feature' | 'importance' | 'value'>('importance');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  // Fetch prediction history for dropdown
  const { data: historyData, refetch: refetchHistory } = useQuery({
    queryKey: ['predictionHistory', 'explain'],
    queryFn: () =>
      predictionsApi.getHistory({
        limit: 15000,  // Get all predictions to have full date range
      }),
    staleTime: 0,  // Always refetch to ensure latest data
    gcTime: 1000 * 60 * 5,  // Keep in cache for 5 minutes but don't use stale data
  });

  // Fetch SHAP explanation for selected prediction
  const { data: shapData, isLoading, error, refetch } = useQuery({
    queryKey: ['shapExplanation', selectedPredictionId, topN],
    queryFn: () => predictionsApi.getSHAPExplanation(Number(selectedPredictionId), topN),
    enabled: !!selectedPredictionId,
  });

  // Prepare waterfall chart data
  const waterfallData: SHAPWaterfallData[] = useMemo(() => {
    if (!shapData) return [];

    let cumulative = shapData.base_value;

    return (shapData.feature_contributions || []).map((feature) => {
      cumulative += feature.value;
      return {
        feature: feature.feature,
        value: feature.value,
        contribution: cumulative,
        isPositive: feature.value > 0,
      };
    });
  }, [shapData]);

  // Prepare and sort table data
  const tableData: FeatureRow[] = useMemo(() => {
    if (!shapData) return [];

    const totalImportance = shapData.total_shap_value || 1;

    return (shapData.feature_contributions || []).map((feature) => ({
      feature: feature.feature,
      value: feature.value,
      importance: Math.abs(feature.importance),
      percentage: (Math.abs(feature.value) / totalImportance) * 100,
      direction: feature.value > 0 ? 'positive' : 'negative',
    })).sort((a, b) => {
      const aVal = a[sortField];
      const bVal = b[sortField];

      if (sortField === 'feature') {
        return sortOrder === 'asc'
          ? a.feature.localeCompare(b.feature)
          : b.feature.localeCompare(a.feature);
      }

      return sortOrder === 'asc' ? aVal - bVal : bVal - aVal;
    });
  }, [shapData, sortField, sortOrder]);

  // Prepare pie chart data
  const pieData = useMemo(() => {
    if (!shapData) return [];

    return (shapData.feature_contributions || [])
      .slice(0, 7)
      .map((f, i) => ({
        name: f.feature,
        value: Math.abs(f.value),
        color: COLORS[i % COLORS.length],
      }));
  }, [shapData]);

  const handleSort = (field: 'feature' | 'importance' | 'value') => {
    if (sortField === field) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortOrder('desc');
    }
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Model Explainability</h1>
        <p className="text-gray-600 mt-1">
          Understand model predictions with SHAP (SHapley Additive exPlanations)
        </p>
      </div>

      {/* Info Banner */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <div className="flex">
          <div className="flex-shrink-0">
            <svg className="h-5 w-5 text-blue-400" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
            </svg>
          </div>
          <div className="ml-3">
            <h3 className="text-sm font-medium text-blue-800">SHAP Values Available</h3>
            <div className="mt-2 text-sm text-blue-700">
              <p>SHAP feature importance explanations are available for <strong>LGBM</strong> and <strong>RandomForest</strong> models only.</p>
              <p className="mt-1">Time series models (AutoARIMA, AutoETS, SeasonalNaive) use pattern-based forecasting and don't have feature-level explanations.</p>
            </div>
          </div>
        </div>
      </div>

      {/* Prediction Selector */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold text-gray-800 mb-4">Select Prediction to Explain</h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Timestamp Dropdown */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Select Timestamp
            </label>
            <select
              value={selectedTimestamp}
              onChange={(e) => {
                const timestamp = e.target.value;
                setSelectedTimestamp(timestamp);
                // Find the prediction ID for this timestamp
                const prediction = historyData?.predictions.find((p) => p.prediction_date === timestamp);
                if (prediction) {
                  setSelectedPredictionId(prediction.id);
                }
              }}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="">Select a timestamp...</option>
              {Array.from(new Set(historyData?.predictions
                .filter((p) => (p.prediction_date.startsWith('2025') || p.prediction_date.startsWith('2026'))) // Show 2025 and 2026
                .map((p) => p.prediction_date) || []))
                .sort((a, b) => b.localeCompare(a)) // Sort by date descending
                .map((date) => (
                  <option key={date} value={date}>
                    {date} {date.startsWith('2025') ? '(2025)' : '(2026)'}
                  </option>
                ))}
            </select>
            <p className="text-xs text-gray-500 mt-1">
              Showing dates with LGBM and RandomForest predictions (models with SHAP values)
            </p>
          </div>

          {/* Model Selector (for selected timestamp) */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Model
            </label>
            <select
              value={selectedPredictionId}
              onChange={(e) => setSelectedPredictionId(e.target.value ? Number(e.target.value) : '')}
              disabled={!selectedTimestamp}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100"
            >
              <option value="">Select a model...</option>
              {selectedTimestamp && historyData?.predictions
                .filter((p) => p.prediction_date === selectedTimestamp && (p.model_name.includes('LGBM') || p.model_name.includes('RandomForest')))
                .map((prediction) => {
                  // Format model name for display
                  const modelName = prediction.model_name
                    .replace(/_/g, ' ')
                    .replace('model', '')
                    .trim();
                  return (
                    <option key={prediction.id} value={prediction.id}>
                      {modelName} - ${prediction.predicted_value.toFixed(2)}
                    </option>
                  );
                })}
            </select>
            <p className="text-xs text-gray-500 mt-1">
              {selectedTimestamp && historyData?.predictions.filter((p) => p.prediction_date === selectedTimestamp && (p.model_name.includes('LGBM') || p.model_name.includes('RandomForest'))).length} models available with SHAP values
            </p>
          </div>

          {/* Top N Features Selector */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Top N Features
            </label>
            <select
              value={topN}
              onChange={(e) => setTopN(Number(e.target.value))}
              disabled={!selectedPredictionId}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100"
            >
              <option value={5}>Top 5 features</option>
              <option value={10}>Top 10 features</option>
              <option value={15}>Top 15 features</option>
              <option value={20}>Top 20 features</option>
            </select>
          </div>
        </div>
      </div>

      {/* Loading State */}
      {isLoading && (
        <div className="bg-white rounded-lg shadow p-12 flex items-center justify-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-800 font-medium">Failed to load SHAP explanation</p>
          <p className="text-sm text-red-600 mt-1">
            Please check the prediction ID and try again.
          </p>
        </div>
      )}

      {/* SHAP Results */}
      {shapData && (
        <>
          {/* Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <div className="bg-white rounded-lg shadow p-6">
              <p className="text-sm text-gray-500">Prediction ID</p>
              <p className="text-2xl font-bold text-gray-900">#{shapData.prediction_id}</p>
            </div>

            <div className="bg-white rounded-lg shadow p-6">
              <p className="text-sm text-gray-500">Predicted Value</p>
              <p className="text-2xl font-bold text-blue-600">${shapData.predicted_value.toFixed(2)}</p>
            </div>

            <div className="bg-white rounded-lg shadow p-6">
              <p className="text-sm text-gray-500">Base Value</p>
              <p className="text-2xl font-bold text-gray-600">${shapData.base_value.toFixed(2)}</p>
            </div>

            <div className="bg-white rounded-lg shadow p-6">
              <p className="text-sm text-gray-500">Total SHAP Value</p>
              <p className="text-2xl font-bold text-purple-600">${shapData.total_shap_value.toFixed(2)}</p>
            </div>
          </div>

          {/* SHAP Waterfall Chart */}
          <ShapWaterfall
            data={waterfallData}
            baseValue={shapData.base_value}
            finalValue={shapData.predicted_value}
            title={`SHAP Waterfall Plot - ${selectedTimestamp || selectedPredictionId ? `Timestamp: ${selectedTimestamp} (Prediction #${selectedPredictionId})` : 'Selected Prediction'}`}
            height={450}
          />

          {/* Charts Row */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Feature Contributions Bar Chart */}
            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">
                Feature Contributions (Top {topN})
              </h3>
              <ResponsiveContainer width="100%" height={350}>
                <BarChart data={shapData.feature_contributions || []} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" tickFormatter={(value) => `$${value.toFixed(2)}`} />
                  <YAxis dataKey="feature" type="category" width={100} />
                  <Tooltip
                    formatter={(value: number, name: string, props: any) => [
                      props.payload.value > 0 ? '+' : '',
                      `$${value.toFixed(2)}`,
                      props.payload.feature,
                    ]}
                  />
                  <Bar dataKey="value" name="Contribution">
                    {(shapData.feature_contributions || []).map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={entry.value > 0 ? '#10b981' : '#ef4444'}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Feature Importance Pie Chart */}
            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">
                Feature Importance Distribution
              </h3>
              <ResponsiveContainer width="100%" height={350}>
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value: number) => `$${value.toFixed(2)}`} />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Impact Summary */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-green-50 border border-green-200 rounded-lg p-6">
              <h4 className="text-lg font-semibold text-green-800 mb-2">Positive Impact</h4>
              <p className="text-3xl font-bold text-green-600">
                ${(shapData.feature_contributions?.filter((f) => f.value > 0).reduce((sum, f) => sum + f.value, 0) || 0).toFixed(2)}
              </p>
              <p className="text-sm text-green-700 mt-2">
                Features that increased the prediction
              </p>
            </div>
            <div className="bg-red-50 border border-red-200 rounded-lg p-6">
              <h4 className="text-lg font-semibold text-red-800 mb-2">Negative Impact</h4>
              <p className="text-3xl font-bold text-red-600">
                ${Math.abs(shapData.feature_contributions?.filter((f) => f.value < 0).reduce((sum, f) => sum + f.value, 0) || 0).toFixed(2)}
              </p>
              <p className="text-sm text-red-700 mt-2">
                Features that decreased the prediction
              </p>
            </div>
          </div>

          {/* Sortable Feature Table */}
          <div className="bg-white rounded-lg shadow overflow-hidden">
            <div className="px-6 py-4 border-b border-gray-200">
              <h3 className="text-lg font-semibold text-gray-800">Feature Details</h3>
            </div>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th
                      onClick={() => handleSort('feature')}
                      className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase cursor-pointer hover:bg-gray-100 transition-colors"
                    >
                      Feature {sortField === 'feature' && (sortOrder === 'asc' ? '↑' : '↓')}
                    </th>
                    <th
                      onClick={() => handleSort('value')}
                      className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase cursor-pointer hover:bg-gray-100 transition-colors"
                    >
                      Contribution {sortField === 'value' && (sortOrder === 'asc' ? '↑' : '↓')}
                    </th>
                    <th
                      onClick={() => handleSort('importance')}
                      className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase cursor-pointer hover:bg-gray-100 transition-colors"
                    >
                      Abs Value {sortField === 'importance' && (sortOrder === 'asc' ? '↑' : '↓')}
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                      Percentage
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                      Impact
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {tableData.map((row, index) => (
                    <tr key={index} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        {row.feature}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">
                        <span className={row.direction === 'positive' ? 'text-green-600' : 'text-red-600'}>
                          {row.value > 0 ? '+' : ''}${row.value.toFixed(2)}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        ${row.importance.toFixed(2)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        <div className="flex items-center">
                          <div className="w-24 bg-gray-200 rounded-full h-2 mr-2">
                            <div
                              className="bg-blue-600 h-2 rounded-full"
                              style={{ width: `${Math.min(row.percentage, 100)}%` }}
                            ></div>
                          </div>
                          <span>{row.percentage.toFixed(1)}%</span>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">
                        <span
                          className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                            row.direction === 'positive'
                              ? 'bg-green-100 text-green-800'
                              : 'bg-red-100 text-red-800'
                          }`}
                        >
                          {row.direction === 'positive' ? 'Positive' : 'Negative'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Text Summary */}
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">Explanation Summary</h3>
            <div className="bg-gray-50 rounded-lg p-4">
              <pre className="text-sm text-gray-800 whitespace-pre-wrap font-mono whitespace-pre">
                {shapData.summary}
              </pre>
            </div>
          </div>
        </>
      )}

      {/* Empty State */}
      {!selectedPredictionId && !isLoading && !error && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-12 text-center">
          <svg
            className="mx-auto h-16 w-16 text-blue-400 mb-4"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          <h3 className="text-xl font-semibold text-blue-900 mb-2">Select a Prediction</h3>
          <p className="text-blue-700 mb-4">
            Choose a validated prediction from the dropdown above to see its SHAP explanation
          </p>
          <div className="text-left max-w-md mx-auto bg-white rounded-lg p-4">
            <p className="text-sm font-medium text-gray-900 mb-2">How to use:</p>
            <ol className="list-decimal list-inside text-sm text-gray-600 space-y-1">
              <li>Go to the Validation page</li>
              <li>Validate some predictions with actual sales data</li>
              <li>Come back here and select a prediction to explain</li>
            </ol>
          </div>
        </div>
      )}
    </div>
  );
};
