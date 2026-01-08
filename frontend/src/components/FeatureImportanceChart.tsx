/**
 * Economic Indicator Sensitivity Component
 * Displays sensitivity of retail sales to key economic indicators
 */

import { FC } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';

interface FeatureImportanceChartProps {
  height?: number;
}

interface SensitivityData {
  feature_name: string;
  prediction_range: number;
  min_prediction: number;
  max_prediction: number;
  elasticity?: number;
}

export const FeatureImportanceChart: FC<FeatureImportanceChartProps> = ({ height = 350 }) => {
  // Fetch sensitivity data for key indicators
  const { data: sensitivityData, isLoading } = useQuery({
    queryKey: ['economic-sensitivity'],
    queryFn: async () => {
      const indicators = [
        { name: 'UNRATE', display: 'Unemployment Rate', min: 3.0, max: 7.0, color: '#ef4444' },
        { name: 'FEDFUNDS', display: 'Fed Funds Rate', min: 0.0, max: 6.0, color: '#3b82f6' },
        { name: 'CPI', display: 'Inflation (CPI)', min: 1.0, max: 6.0, color: '#f59e0b' },
        { name: 'GDP', display: 'GDP Growth', min: -2.0, max: 4.0, color: '#10b981' },
        { name: 'PAYEMS', display: 'Nonfarm Payrolls', min: -200, max: 500, color: '#8b5cf6' },
      ];

      const results = await Promise.all(
        indicators.map(async (indicator) => {
          try {
            const response = await fetch('http://localhost:8000/api/scenarios/sensitivity', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                category: 'total_sales',
                feature_name: indicator.name,
                min_value: indicator.min,
                max_value: indicator.max,
                num_steps: 2, // Just need range for importance
              })
            });

            if (response.ok) {
              const data = await response.json();
              return {
                feature: indicator.display,
                importance: Math.abs(data.prediction_range) / 1000, // Normalize to thousands
                range: data.prediction_range,
                color: indicator.color,
                elasticity: data.elasticity,
              };
            }
            return null;
          } catch (error) {
            console.error(`Error fetching sensitivity for ${indicator.name}:`, error);
            return null;
          }
        })
      );

      // Filter successful results and sort by importance
      const successfulResults = results.filter(r => r !== null);
      return successfulResults.sort((a, b) => b.importance - a.importance);
    },
  });

  if (isLoading) {
    return (
      <div className="bg-white dark:bg-slate-800 rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-800 dark:text-white mb-4">Economic Indicator Sensitivity</h3>
        <div className="flex items-center justify-center" style={{ height }}>
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
        </div>
      </div>
    );
  }

  const data = sensitivityData || [];

  return (
    <div className="bg-white dark:bg-slate-800 rounded-lg shadow p-6">
      <h3 className="text-lg font-semibold text-gray-800 dark:text-white mb-2">
        Economic Indicator Sensitivity
      </h3>
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
        Impact of economic indicators on retail sales forecasts
      </p>

      {data.length > 0 ? (
        <>
          <ResponsiveContainer width="100%" height={height}>
            <BarChart data={data} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.3} />
              <XAxis
                type="number"
                tickFormatter={(value) => `$${(value).toFixed(0)}K range`}
                tick={{ fill: '#64748b' }}
              />
              <YAxis
                dataKey="feature"
                type="category"
                width={150}
                tick={{ fill: '#64748b', fontSize: 12 }}
              />
              <Tooltip
                formatter={(value: number, name: string) => {
                  if (name === 'range') {
                    return [`$${(value).toFixed(0)}`, 'Prediction Range'];
                  }
                  if (name === 'elasticity') {
                    return [value.toFixed(2), 'Elasticity'];
                  }
                  return [value, name];
                }}
                contentStyle={{ backgroundColor: 'rgba(255, 255, 255, 0.95)', border: '1px solid #ccc' }}
              />
              <Bar dataKey="importance" fill="#3b82f6">
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>

          <div className="mt-4 p-3 bg-amber-50 dark:bg-amber-900/20 rounded-lg">
            <p className="text-sm text-amber-900 dark:text-amber-100">
              <strong>Interpretation:</strong> Shows the range of forecast values across indicator extremes. Higher bars indicate greater sensitivity to that economic indicator.
            </p>
          </div>
        </>
      ) : (
        <div className="flex items-center justify-center" style={{ height }}>
          <p className="text-gray-500 dark:text-gray-400">No sensitivity data available</p>
        </div>
      )}
    </div>
  );
};

export default FeatureImportanceChart;
