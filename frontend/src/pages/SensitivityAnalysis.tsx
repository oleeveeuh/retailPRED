/**
 * Sensitivity Analysis Page
 * Interactive analysis of retail sales sensitivity to economic indicators
 */

import type { FC } from 'react';
import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import { scenariosApi } from '../api/unifiedApi';
import {
  Sliders,
  AlertCircle,
  Activity,
  BarChart3,
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart as RechartsBarChart,
  Bar,
} from 'recharts';

const KEY_INDICATORS = [
  { name: 'UNRATE', display: 'Unemployment Rate', category: 'Labor Market', unit: '%', min: 3.0, max: 8.0, step: 0.1 },
  { name: 'FEDFUNDS', display: 'Federal Funds Rate', category: 'Monetary Policy', unit: '%', min: 0.0, max: 6.0, step: 0.25 },
  { name: 'CPI', display: 'Consumer Price Index', category: 'Consumer', unit: '%', min: 1.0, max: 6.0, step: 0.1 },
  { name: 'GDP', display: 'GDP Growth', category: 'Production', unit: '%', min: -2.0, max: 4.0, step: 0.25 },
  { name: 'PAYEMS', display: 'Nonfarm Payrolls', category: 'Labor Market', unit: 'K', min: -500, max: 500, step: 50 },
];

const STRESS_SCENARIOS = [
  { name: 'Mild Recession', description: 'Slight economic downturn', icon: '⚠️' },
  { name: 'Severe Recession', description: 'Major economic crisis', icon: '🔴' },
  { name: 'Boom Times', description: 'Strong economic growth', icon: '🚀' },
  { name: 'Stagflation', description: 'High inflation, low growth', icon: '📉' },
];

interface SensitivityData {
  feature_name: string;
  values: number[];
  predictions: number[];
  min_prediction: number;
  max_prediction: number;
  prediction_range: number;
  prediction_mean: number;
  elasticity: number | null;
  baseline_value: number;
}

interface TornadoItem {
  indicator: string;
  low: number;
  high: number;
  range: number;
}

export const SensitivityAnalysis: FC = () => {
  const [category, setCategory] = useState('total_sales');
  const [selectedIndicator] = useState(KEY_INDICATORS[0]);
  const [indicatorValues, setIndicatorValues] = useState<Record<string, number>>({
    UNRATE: 3.7,
    FEDFUNDS: 5.25,
    CPI: 3.2,
    GDP: 2.5,
    PAYEMS: 250,
  });

  // Fetch sensitivity data for selected indicator
  const { data: sensitivityData, isLoading } = useQuery({
    queryKey: ['sensitivity', category, selectedIndicator.name],
    queryFn: async () => {
      const indicator = KEY_INDICATORS.find(i => i.name === selectedIndicator.name);
      if (!indicator) return null;

      return await scenariosApi.analyzeSensitivity({
        category,
        feature_name: selectedIndicator.name,
        min_value: indicator.min,
        max_value: indicator.max,
        num_steps: 10,
      }) as SensitivityData;
    },
  });

  // Fetch all sensitivities for tornado chart
  const { data: tornadoData } = useQuery({
    queryKey: ['tornado-data', category],
    queryFn: async () => {
      const tornadoItems: TornadoItem[] = [];

      for (const indicator of KEY_INDICATORS) {
        const data = await scenariosApi.analyzeSensitivity({
          category,
          feature_name: indicator.name,
          min_value: indicator.min,
          max_value: indicator.max,
          num_steps: 2, // Just min/max for tornado
        });

        tornadoItems.push({
          indicator: indicator.display,
          low: data.min_prediction,
          high: data.max_prediction,
          range: data.prediction_range,
        });
      }

      return tornadoItems.sort((a, b) => b.range - a.range);
    },
  });

  // Prepare chart data
  const chartData = sensitivityData?.values.map((value, idx) => ({
    value: value.toFixed(2),
    prediction: sensitivityData.predictions[idx],
  })) || [];

  // Calculate percentage change from baseline
  const baselinePrediction = sensitivityData?.predictions[Math.floor(sensitivityData.predictions.length / 2)] || 0;
  const chartDataWithPct = chartData.map(d => ({
    ...d,
    pctChange: ((d.prediction - baselinePrediction) / baselinePrediction) * 100,
  }));

  const handleIndicatorChange = (name: string, value: number) => {
    setIndicatorValues(prev => ({ ...prev, [name]: value }));
  };

  const applyStressScenario = (scenarioName: string) => {
    // Apply preset stress scenario values
    switch (scenarioName) {
      case 'Mild Recession':
        setIndicatorValues({
          UNRATE: 5.5,
          FEDFUNDS: 4.0,
          CPI: 2.5,
          GDP: 0.5,
          PAYEMS: 100,
        });
        break;
      case 'Severe Recession':
        setIndicatorValues({
          UNRATE: 7.5,
          FEDFUNDS: 3.0,
          CPI: 1.5,
          GDP: -1.5,
          PAYEMS: -200,
        });
        break;
      case 'Boom Times':
        setIndicatorValues({
          UNRATE: 3.0,
          FEDFUNDS: 5.0,
          CPI: 2.0,
          GDP: 4.0,
          PAYEMS: 400,
        });
        break;
      case 'Stagflation':
        setIndicatorValues({
          UNRATE: 5.0,
          FEDFUNDS: 6.0,
          CPI: 5.0,
          GDP: 0.0,
          PAYEMS: 50,
        });
        break;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 p-8">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header */}
        <div>
          <h1 className="text-4xl font-bold text-slate-900 dark:text-white mb-2">
            Economic Factor Sensitivity Analysis
          </h1>
          <p className="text-slate-600 dark:text-slate-400">
            Understand how changes in economic indicators impact retail sales forecasts
          </p>
        </div>

        {/* Stress Test Scenarios */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white dark:bg-slate-800 rounded-lg shadow-lg p-6"
        >
          <h2 className="text-xl font-semibold text-slate-900 dark:text-white mb-4 flex items-center">
            <AlertCircle className="w-5 h-5 mr-2 text-orange-600" />
            Stress Test Scenarios
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {STRESS_SCENARIOS.map((scenario) => (
              <button
                key={scenario.name}
                onClick={() => applyStressScenario(scenario.name)}
                className="p-4 rounded-lg border-2 border-slate-200 dark:border-slate-700 hover:border-orange-500 hover:bg-orange-50 dark:hover:bg-orange-900/20 transition-all"
              >
                <div className="text-2xl mb-2">{scenario.icon}</div>
                <div className="font-semibold text-slate-900 dark:text-white mb-1">
                  {scenario.name}
                </div>
                <div className="text-xs text-slate-600 dark:text-slate-400">
                  {scenario.description}
                </div>
              </button>
            ))}
          </div>
        </motion.div>

        {/* Interactive Indicators */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-white dark:bg-slate-800 rounded-lg shadow-lg p-6"
        >
          <h2 className="text-xl font-semibold text-slate-900 dark:text-white mb-4 flex items-center">
            <Sliders className="w-5 h-5 mr-2 text-blue-600" />
            Economic Indicators
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {KEY_INDICATORS.map((indicator) => (
              <div key={indicator.name} className="space-y-2">
                <div className="flex justify-between items-center">
                  <label className="text-sm font-medium text-slate-700 dark:text-slate-300">
                    {indicator.display}
                  </label>
                  <span className="text-sm font-semibold text-slate-900 dark:text-white">
                    {indicatorValues[indicator.name]} {indicator.unit}
                  </span>
                </div>
                <input
                  type="range"
                  min={indicator.min}
                  max={indicator.max}
                  step={indicator.step}
                  value={indicatorValues[indicator.name] || 0}
                  onChange={(e) => handleIndicatorChange(indicator.name, parseFloat(e.target.value))}
                  className="w-full h-2 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-600"
                />
                <div className="flex justify-between text-xs text-slate-600 dark:text-slate-400">
                  <span>{indicator.min} {indicator.unit}</span>
                  <span>{indicator.max} {indicator.unit}</span>
                </div>
                <div className="text-xs text-slate-500 dark:text-slate-400">
                  Category: {indicator.category}
                </div>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Sensitivity Chart */}
        {sensitivityData && !isLoading && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="grid grid-cols-1 lg:grid-cols-2 gap-6"
          >
            {/* Line Chart */}
            <div className="bg-white dark:bg-slate-800 rounded-lg shadow-lg p-6">
              <h3 className="text-xl font-semibold text-slate-900 dark:text-white mb-4 flex items-center">
                <Activity className="w-5 h-5 mr-2 text-green-600" />
                Sensitivity to {selectedIndicator.display}
              </h3>

              <div className="mb-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm text-slate-600 dark:text-slate-400">
                    Baseline ({selectedIndicator.display} = {sensitivityData.baseline_value.toFixed(2)})
                  </span>
                  <span className="font-semibold text-slate-900 dark:text-white">
                    ${baselinePrediction.toLocaleString()}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-slate-600 dark:text-slate-400">
                    Range of Forecasts
                  </span>
                  <span className="font-semibold text-slate-900 dark:text-white">
                    ${sensitivityData.min_prediction.toLocaleString()} - ${sensitivityData.max_prediction.toLocaleString()}
                  </span>
                </div>
              </div>

              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={chartDataWithPct}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="value"
                    label={{ value: `${selectedIndicator.name} (${selectedIndicator.unit})`, position: 'insideBottom', offset: -5 }}
                  />
                  <YAxis
                    label={{ value: 'Forecast ($)', angle: -90, position: 'insideLeft' }}
                    tickFormatter={(value) => `$${(value / 1000).toFixed(0)}K`}
                  />
                  <Tooltip
                    formatter={(value: number | undefined) => `$${(value || 0).toFixed(2)}`}
                    labelFormatter={(value) => `${selectedIndicator.name}: ${value}${selectedIndicator.unit}`}
                  />
                  <Line
                    type="monotone"
                    dataKey="prediction"
                    stroke="#3b82f6"
                    strokeWidth={2}
                    dot={{ r: 4 }}
                  />
                </LineChart>
              </ResponsiveContainer>

              {sensitivityData.elasticity && (
                <div className="mt-4 p-3 bg-slate-50 dark:bg-slate-900 rounded-lg">
                  <div className="text-sm text-slate-600 dark:text-slate-400">
                    <span className="font-medium">Elasticity:</span> {sensitivityData.elasticity.toFixed(2)}
                  </div>
                  <div className="text-xs text-slate-500 dark:text-slate-500 mt-1">
                    A 1% change in {selectedIndicator.display.toLowerCase()} leads to a {sensitivityData.elasticity.toFixed(1)}% change in retail sales
                  </div>
                </div>
              )}
            </div>

            {/* Tornado Chart */}
            <div className="bg-white dark:bg-slate-800 rounded-lg shadow-lg p-6">
              <h3 className="text-xl font-semibold text-slate-900 dark:text-white mb-4 flex items-center">
                <BarChart3 className="w-5 h-5 mr-2 text-purple-600" />
                Factor Importance (Tornado Chart)
              </h3>
              <p className="text-sm text-slate-600 dark:text-slate-400 mb-4">
                Shows the range of forecast values across indicator extremes
              </p>

              <ResponsiveContainer width="100%" height={400}>
                <RechartsBarChart
                  data={tornadoData || []}
                  layout="vertical"
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    type="number"
                    tickFormatter={(value) => `$${(value / 1000).toFixed(0)}K`}
                  />
                  <YAxis type="category" dataKey="indicator" width={120} />
                  <Tooltip
                    formatter={(value: number | undefined) => `$${(value || 0).toLocaleString()}`}
                  />
                  <Bar dataKey="high" fill="#10b981" name="High Scenario" />
                  <Bar dataKey="low" fill="#ef4444" name="Low Scenario" />
                </RechartsBarChart>
              </ResponsiveContainer>

              <div className="mt-4 space-y-2">
                <div className="flex items-center text-sm">
                  <div className="w-4 h-4 bg-green-500 mr-2"></div>
                  <span className="text-slate-700 dark:text-slate-300">High value scenario</span>
                </div>
                <div className="flex items-center text-sm">
                  <div className="w-4 h-4 bg-red-500 mr-2"></div>
                  <span className="text-slate-700 dark:text-slate-300">Low value scenario</span>
                </div>
              </div>
            </div>
          </motion.div>
        )}

        {/* Category Selector */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-white dark:bg-slate-800 rounded-lg shadow-lg p-6"
        >
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
            Analyze Category
          </h3>
          <select
            value={category}
            onChange={(e) => setCategory(e.target.value)}
            className="w-full max-w-xs px-4 py-2 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 text-slate-900 dark:text-white"
          >
            <option value="total_sales">Total Retail Sales</option>
            <option value="general_merchandise">General Merchandise</option>
            <option value="food_beverage">Food & Beverage</option>
            <option value="automobile_dealers">Automobile Dealers</option>
            <option value="building_materials">Building Materials</option>
          </select>
        </motion.div>
      </div>
    </div>
  );
};

export default SensitivityAnalysis;
