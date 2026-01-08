/**
 * Forecast Chart Component
 * Displays historical sales data (through Dec 2025) with forecasts (starting Jan 2026)
 */

import { FC } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';

interface ForecastChartProps {
  height?: number;
}

export const ForecastChart: FC<ForecastChartProps> = ({ height = 400 }) => {
  // Fetch historical sales data (through Dec 2025)
  const { data: historicalData, isLoading } = useQuery({
    queryKey: ['historical-sales', 'total_sales'],
    queryFn: () => fetch('http://localhost:8000/api/historical-sales?category=total_sales&days_back=365')
      .then(res => res.json())
      .then(data => data.data || []),
  });

  // Fetch baseline scenario for 2026 forecasts
  const { data: scenarioData } = useQuery({
    queryKey: ['baseline-scenario-2026'],
    queryFn: () => fetch('http://localhost:8000/api/scenarios/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        scenario_type: 'baseline',
        category: 'total_sales'
      })
    }).then(res => res.json())
      .then(data => data),
    enabled: historicalData && historicalData.length > 0,
  });

  // Generate 2026 forecasts based on scenario analysis
  const forecastData = (() => {
    if (!historicalData || historicalData.length === 0) return [];

    // Get baseline value from scenario API or historical average
    const recentData = historicalData.slice(-3);
    const historicalAvg = recentData.reduce((sum, d) => sum + d.value, 0) / recentData.length;
    const baselineValue = scenarioData?.prediction || historicalAvg;

    // Get confidence interval from scenario
    const confidenceInterval = scenarioData?.confidence_interval || [baselineValue * 0.95, baselineValue * 1.05];

    // Generate 12 months of forecasts for 2026 with gradual growth
    const forecasts = [];
    const monthlyGrowthRate = 0.01 / 12; // 1% annual growth, distributed monthly

    for (let month = 0; month < 12; month++) {
      const forecastDate = new Date(2026, month, 1);
      const seasonality = Math.sin((month / 12) * 2 * Math.PI) * 0.05; // ±5% seasonal variation
      const predictedValue = baselineValue * (1 + monthlyGrowthRate * (month + 1) + seasonality);

      forecasts.push({
        date: forecastDate.toISOString().split('T')[0],
        forecast: Math.max(0, predictedValue),
        actual: null,
      });
    }

    return forecasts;
  })();

  // Combine and format data
  const chartData = () => {
    const historical = historicalData || [];

    // Combine historical (through Dec 2025) with forecasts (starting Jan 2026)
    const combined = [
      ...historical
        .filter(item => item.date <= '2025-12-31')  // Historical through December 2025
        .map(item => ({
          date: item.date,
          actual: item.value,
          forecast: null,
        })),
      ...forecastData
    ];

    // Sort by date and take last 24 months (12 months historical + 12 months forecast)
    return combined
      .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime())
      .slice(-24);
  };

  if (isLoading) {
    return (
      <div className="bg-white dark:bg-slate-800 rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-800 dark:text-white mb-4">Retail Sales Forecast</h3>
        <div className="flex items-center justify-center" style={{ height }}>
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
        </div>
      </div>
    );
  }

  const data = chartData();

  // Don't render chart if no data
  if (!data || data.length === 0) {
    return (
      <div className="bg-white dark:bg-slate-800 rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-800 dark:text-white mb-4">Retail Sales Forecast</h3>
        <div className="flex items-center justify-center" style={{ height }}>
          <p className="text-gray-500 dark:text-gray-400">No data available</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-slate-800 rounded-lg shadow p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-800 dark:text-white">Retail Sales Forecast</h3>
        <div className="flex items-center gap-4 text-sm">
          <div className="flex items-center">
            <div className="w-3 h-3 bg-green-500 mr-2 rounded-full"></div>
            <span className="text-gray-600 dark:text-gray-400">Historical (through Dec 2025)</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 bg-blue-500 mr-2 rounded-full"></div>
            <span className="text-gray-600 dark:text-gray-400">Forecast (Jan 2026+)</span>
          </div>
        </div>
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.3} />
          <XAxis
            dataKey="date"
            tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'short', year: '2-digit' })}
            tick={{ fill: '#64748b' }}
          />
          <YAxis
            tickFormatter={(value) => '$' + (value / 1000).toFixed(0) + 'K'}
            tick={{ fill: '#64748b' }}
          />
          <Tooltip
            formatter={(value: number | null, _name: string, item: any) => {
              if (value == null) return null;

              const key = item?.dataKey; // "actual" or "forecast"
              const label = key === 'actual' ? 'Actual' : 'Forecast';

              return [`$${value.toFixed(2)}`, label]; // [value, name]
            }}
            labelFormatter={(value) =>
              new Date(value).toLocaleDateString('en-US', { month: 'long', year: 'numeric' })
            }
          />

          <Legend />
          <ReferenceLine x="2025-12-01" stroke="#94a3b8" strokeDasharray="3 3" label="Forecast Start" />
          <Line
            type="monotone"
            dataKey="actual"
            stroke="#10b981"
            strokeWidth={2}
            name="Actual Sales"
            connectNulls={false}
            dot={{ r: 3 }}
            activeDot={{ r: 5 }}
          />
          <Line
            type="monotone"
            dataKey="forecast"
            stroke="#3b82f6"
            strokeWidth={2}
            strokeDasharray="5 5"
            name="2026 Forecast"
            connectNulls={false}
            dot={{ r: 3 }}
            activeDot={{ r: 5 }}
          />
        </LineChart>
      </ResponsiveContainer>
      <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <p className="text-sm text-blue-900 dark:text-blue-100">
          <strong>Historical Data:</strong> Shown through December 2025. <strong>Forecast:</strong> 2026 predictions based on economic scenario analysis (baseline: +1% annual growth with seasonal variation).
        </p>
      </div>
    </div>
  );
};

export default ForecastChart;
