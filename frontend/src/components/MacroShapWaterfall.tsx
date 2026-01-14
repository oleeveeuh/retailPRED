/**
 * Macro SHAP Waterfall Component
 * Color-coded SHAP waterfall chart grouped by data source (FRED, MRTS, Yahoo Finance)
 */

import { FC } from 'react';
import { motion } from 'framer-motion';
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
import {
  Info,
  Database,
  TrendingUp,
} from 'lucide-react';

interface MacroShapData {
  feature: string;
  displayName: string;
  value: number;
  contribution: number;
  source: 'FRED' | 'MRTS' | 'Yahoo Finance';
  category: string;
  explanation: string;
}

interface MacroShapWaterfallProps {
  data: MacroShapData[];
  baseValue: number;
  finalValue: number;
  height?: number;
}

// Data source colors
const SOURCE_COLORS = {
  'FRED': { bg: '#3b82f6', border: '#2563eb', light: 'bg-blue-50 dark:bg-blue-900/20' },      // Blue
  'MRTS': { bg: '#10b981', border: '#059669', light: 'bg-green-50 dark:bg-green-900/20' },   // Green
  'Yahoo Finance': { bg: '#f59e0b', border: '#d97706', light: 'bg-amber-50 dark:bg-amber-900/20' }, // Orange
};

const SOURCE_ICONS = {
  'FRED': '📊',
  'MRTS': '🛒',
  'Yahoo Finance': '📈',
};

export const MacroShapWaterfall: FC<MacroShapWaterfallProps> = ({
  data,
  baseValue,
  finalValue,
  height = 400,
}) => {
  // Prepare chart data with running total
  let runningTotal = baseValue;
  const chartData = data.map((item) => {
    runningTotal += item.contribution;
    return {
      ...item,
      runningTotal,
      isPositive: item.contribution >= 0,
    };
  });

  // Get top features by absolute contribution
  const topFeatures = [...data]
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
    .slice(0, 10);

  // Group by source
  const bySource = topFeatures.reduce((acc, item) => {
    if (!acc[item.source]) acc[item.source] = [];
    acc[item.source].push(item);
    return acc;
  }, {} as Record<string, MacroShapData[]>);

  // Custom tooltip
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-white dark:bg-slate-800 p-4 rounded-lg shadow-lg border border-slate-200 dark:border-slate-700">
          <div className="font-semibold text-slate-900 dark:text-white mb-2">
            {data.displayName}
          </div>
          <div className="space-y-1 text-sm">
            <div className="flex justify-between">
              <span className="text-slate-600 dark:text-slate-400">Contribution:</span>
              <span className={`font-semibold ${data.contribution >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {data.contribution >= 0 ? '+' : ''}${data.contribution.toFixed(2)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-600 dark:text-slate-400">Source:</span>
              <span className="font-semibold flex items-center">
                {SOURCE_ICONS[data.source]} {data.source}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-600 dark:text-slate-400">Category:</span>
              <span className="text-slate-900 dark:text-white">{data.category}</span>
            </div>
          </div>
          <div className="mt-2 pt-2 border-t border-slate-200 dark:border-slate-700">
            <p className="text-xs text-slate-600 dark:text-slate-400">
              {data.explanation}
            </p>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="space-y-6">
      {/* Legend */}
      <div className="flex flex-wrap gap-4 text-sm">
        {Object.entries(SOURCE_COLORS).map(([source, colors]) => (
          <div key={source} className="flex items-center">
            <div
              className={`w-4 h-4 mr-2 rounded`}
              style={{ backgroundColor: colors.bg }}
            ></div>
            <span className="text-slate-700 dark:text-slate-300">
              {SOURCE_ICONS[source]} {source}
            </span>
          </div>
        ))}
      </div>

      {/* Waterfall Chart */}
      <ResponsiveContainer width="100%" height={height}>
        <BarChart
          data={chartData}
          layout="vertical"
          margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.3} />
          <XAxis
            type="number"
            tickFormatter={(value) => `$${(value / 1000).toFixed(0)}K`}
          />
          <YAxis
            type="category"
            dataKey="displayName"
            width={150}
            tick={{ fontSize: 12 }}
          />
          <Tooltip content={<CustomTooltip />} />

          {/* Bars color-coded by source and direction */}
          {chartData.map((entry, index) => (
            <Bar
              key={`bar-${index}`}
              dataKey="contribution"
              stackId="stack"
            >
              <Cell
                fill={entry.isPositive ? SOURCE_COLORS[entry.source].bg : '#ef4444'}
                stroke={SOURCE_COLORS[entry.source].border}
                strokeWidth={2}
              />
            </Bar>
          ))}
        </BarChart>
      </ResponsiveContainer>

      {/* Feature Details by Source */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {Object.entries(bySource).map(([source, items]) => (
          <motion.div
            key={source}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className={`rounded-lg p-4 border-2 ${SOURCE_COLORS[source as keyof typeof SOURCE_COLORS].light}`}
          >
            <div className="flex items-center mb-3">
              <span className="text-2xl mr-2">{SOURCE_ICONS[source]}</span>
              <h3 className="text-lg font-semibold text-slate-900 dark:text-white">
                {source}
              </h3>
            </div>

            <div className="space-y-2">
              {items.map((item, idx) => (
                <div
                  key={idx}
                  className="p-2 bg-white dark:bg-slate-800 rounded border border-slate-200 dark:border-slate-700"
                >
                  <div className="flex justify-between items-start mb-1">
                    <span className="text-sm font-medium text-slate-900 dark:text-white">
                      {item.displayName}
                    </span>
                    <span
                      className={`text-sm font-semibold ${
                        item.contribution >= 0 ? 'text-green-600' : 'text-red-600'
                      }`}
                    >
                      {item.contribution >= 0 ? '+' : ''}${item.contribution.toFixed(0)}
                    </span>
                  </div>
                  <div className="text-xs text-slate-600 dark:text-slate-400">
                    {item.category}
                  </div>
                </div>
              ))}
            </div>

            {/* Summary */}
            <div className="mt-3 pt-3 border-t border-slate-200 dark:border-slate-700">
              <div className="text-xs text-slate-600 dark:text-slate-400">
                <strong>{items.length} indicators</strong> from {source}
              </div>
              <div className="text-xs text-slate-600 dark:text-slate-400 mt-1">
                Total impact: ${items.reduce((sum, i) => sum + i.contribution, 0).toFixed(0)}
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Plain Language Explanations */}
      <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
        <h4 className="font-semibold text-slate-900 dark:text-white mb-2 flex items-center">
          <Info className="w-4 h-4 mr-2 text-primary-600" />
          Understanding the Economic Factors
        </h4>
        <div className="space-y-2 text-sm text-slate-700 dark:text-slate-300">
          {topFeatures.slice(0, 5).map((item, idx) => (
            <div key={idx} className="flex items-start">
              <span className="mr-2">{idx + 1}.</span>
              <span>
                <strong>{item.displayName}</strong> ({item.category}): {item.explanation}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Source Information */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
        <div className="bg-white dark:bg-slate-800 rounded p-3 border border-blue-200 dark:border-blue-900">
          <div className="font-semibold text-blue-900 dark:text-blue-100 mb-1 flex items-center">
            <Database className="w-3 h-3 mr-1" />
            FRED (Federal Reserve Economic Data)
          </div>
          <p className="text-slate-600 dark:text-slate-400">
            800,000+ economic indicators from the Federal Reserve including unemployment, GDP, inflation, and interest rates
          </p>
        </div>

        <div className="bg-white dark:bg-slate-800 rounded p-3 border border-green-200 dark:border-green-900">
          <div className="font-semibold text-green-900 dark:text-green-100 mb-1 flex items-center">
            <TrendingUp className="w-3 h-3 mr-1" />
            MRTS (Monthly Retail Trade Survey)
          </div>
          <p className="text-slate-600 dark:text-slate-400">
            Advance monthly sales estimates for retail trade from the U.S. Census Bureau
          </p>
        </div>

        <div className="bg-white dark:bg-slate-800 rounded p-3 border border-amber-200 dark:border-amber-900">
          <div className="font-semibold text-amber-900 dark:text-amber-100 mb-1 flex items-center">
            <TrendingUp className="w-3 h-3 mr-1" />
            Yahoo Finance
          </div>
          <p className="text-slate-600 dark:text-slate-400">
            Financial market data including S&P 500, Dow Jones, and volatility indices
          </p>
        </div>
      </div>
    </div>
  );
};

export default MacroShapWaterfall;
