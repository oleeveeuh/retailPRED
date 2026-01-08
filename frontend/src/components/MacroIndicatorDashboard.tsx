/**
 * Macro Indicator Dashboard Component
 * Displays current economic snapshot with visual status indicators
 */

import { FC } from 'react';
import { useQuery } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import {
  Activity,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  CheckCircle,
  Info,
  Sparkles,
} from 'lucide-react';

interface Indicator {
  name: string;
  display: string;
  value: number;
  previousValue: number;
  unit: string;
  category: string;
  source: string;
  lead_lag: 'leading' | 'coincident' | 'lagging';
  status: 'healthy' | 'warning' | 'alert';
  date: string;
}

const INDICATOR_CATEGORIES = {
  'Labor Market': { icon: '👥', color: 'blue' },
  'Monetary Policy': { icon: '💰', color: 'green' },
  'Consumer': { icon: '🛒', color: 'purple' },
  'Housing': { icon: '🏠', color: 'orange' },
  'Production': { icon: '🏭', color: 'red' },
  'Financial': { icon: '📈', color: 'cyan' },
};

const STATUS_CONFIG = {
  healthy: { icon: CheckCircle, color: 'green', bg: 'bg-green-50 dark:bg-green-900/20', text: 'text-green-600 dark:text-green-400' },
  warning: { icon: AlertTriangle, color: 'yellow', bg: 'bg-yellow-50 dark:bg-yellow-900/20', text: 'text-yellow-600 dark:text-yellow-400' },
  alert: { icon: TrendingDown, color: 'red', bg: 'bg-red-50 dark:bg-red-900/20', text: 'text-red-600 dark:text-red-400' },
};

export const MacroIndicatorDashboard: FC = () => {
  // Fetch current indicators
  const { data: indicatorsData, isLoading } = useQuery({
    queryKey: ['current-indicators-detailed'],
    queryFn: async () => {
      const response = await fetch('http://localhost:8000/api/economic-indicators/current');
      if (!response.ok) throw new Error('Failed to fetch indicators');
      const data = await response.json();
      return data.indicators || [];
    },
  });

  // Calculate status based on indicator values
  const calculateStatus = (indicator: string, value: number): 'healthy' | 'warning' | 'alert' => {
    // Define thresholds for each indicator
    const thresholds: Record<string, { healthy: number[], warning: number[], alert: number[] }> = {
      UNRATE: { healthy: [3.5, 5.0], warning: [5.0, 6.5], alert: [6.5, 100] }, // Unemployment
      FEDFUNDS: { healthy: [1.5, 4.0], warning: [4.0, 5.5], alert: [5.5, 100] }, // Fed Funds Rate
      CPI: { healthy: [1.5, 2.5], warning: [2.5, 4.0], alert: [4.0, 100] }, // Inflation
      GDP: { healthy: [2.0, 100], warning: [0.5, 2.0], alert: [-100, 0.5] }, // GDP Growth
      PAYEMS: { healthy: [150000, 1000000], warning: [50000, 150000], alert: [-1000000, 50000] }, // Payrolls
      UMCSENT: { healthy: [75, 100], warning: [60, 75], alert: [0, 60] }, // Consumer Sentiment
      HOUST: { healthy: [1200, 2000], warning: [900, 1200], alert: [0, 900] }, // Housing Starts
      DGS10: { healthy: [2.0, 4.0], warning: [4.0, 5.5], alert: [5.5, 100] }, // 10-Year Treasury
      SP500: { healthy: [4000, 100000], warning: [3500, 4000], alert: [0, 3500] }, // S&P 500
    };

    const range = thresholds[indicator];
    if (!range) return 'healthy';

    if (value >= range.healthy[0] && value <= range.healthy[1]) return 'healthy';
    if (value >= range.warning[0] && value <= range.warning[1]) return 'warning';
    return 'alert';
  };

  const getChangeIcon = (current: number, previous: number) => {
    if (current > previous) return TrendingUp;
    if (current < previous) return TrendingDown;
    return Activity;
  };

  const getChangeColor = (current: number, previous: number) => {
    if (current > previous) return 'text-green-600';
    if (current < previous) return 'text-red-600';
    return 'text-gray-600';
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  // Group indicators by category
  const indicatorsByCategory = indicatorsData?.reduce((acc: Record<string, Indicator[]>, ind: any) => {
    const category = ind.category || 'Other';
    if (!acc[category]) acc[category] = [];
    acc[category].push({
      ...ind,
      status: calculateStatus(ind.name, ind.value),
      lead_lag: ind.name === 'SP500' || ind.name === 'HOUST' ? 'leading' :
                 ind.name === 'UNRATE' || ind.name === 'CPI' ? 'lagging' : 'coincident',
    });
    return acc;
  }, {}) || {};

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-semibold text-slate-900 dark:text-white flex items-center">
            <Sparkles className="w-6 h-6 mr-2 text-purple-600" />
            Current Economic Snapshot
          </h2>
          <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
            Latest values from FRED, MRTS, and Yahoo Finance
          </p>
        </div>
      </div>

      {/* Overall Economic Health */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg p-6"
      >
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center">
            <div className="text-sm text-slate-600 dark:text-slate-400 mb-2">Healthy Indicators</div>
            <div className="text-3xl font-bold text-green-600">
              {indicatorsData?.filter((i: any) => calculateStatus(i.name, i.value) === 'healthy').length || 0}
            </div>
          </div>
          <div className="text-center">
            <div className="text-sm text-slate-600 dark:text-slate-400 mb-2">Warning Indicators</div>
            <div className="text-3xl font-bold text-yellow-600">
              {indicatorsData?.filter((i: any) => calculateStatus(i.name, i.value) === 'warning').length || 0}
            </div>
          </div>
          <div className="text-center">
            <div className="text-sm text-slate-600 dark:text-slate-400 mb-2">Alert Indicators</div>
            <div className="text-3xl font-bold text-red-600">
              {indicatorsData?.filter((i: any) => calculateStatus(i.name, i.value) === 'alert').length || 0}
            </div>
          </div>
        </div>
      </motion.div>

      {/* Indicators by Category */}
      {Object.entries(INDICATOR_CATEGORIES).map(([category, config]) => {
        const categoryIndicators = indicatorsByCategory[category] || [];
        if (categoryIndicators.length === 0) return null;

        return (
          <motion.div
            key={category}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="bg-white dark:bg-slate-800 rounded-lg shadow-lg p-6"
          >
            <div className="flex items-center mb-4">
              <span className="text-2xl mr-2">{config.icon}</span>
              <h3 className="text-xl font-semibold text-slate-900 dark:text-white">
                {category}
              </h3>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {categoryIndicators.map((indicator, idx) => {
                const StatusIcon = STATUS_CONFIG[indicator.status].icon;
                const ChangeIcon = getChangeIcon(indicator.value, indicator.previousValue);
                const changeColor = getChangeColor(indicator.value, indicator.previousValue);
                const changePct = ((indicator.value - indicator.previousValue) / indicator.previousValue * 100);

                return (
                  <div
                    key={idx}
                    className={`p-4 rounded-lg border-2 ${
                      indicator.status === 'healthy' ? 'border-green-200 dark:border-green-900' :
                      indicator.status === 'warning' ? 'border-yellow-200 dark:border-yellow-900' :
                      'border-red-200 dark:border-red-900'
                    }`}
                  >
                    {/* Status Header */}
                    <div className="flex items-center justify-between mb-2">
                      <div className="text-sm font-medium text-slate-700 dark:text-slate-300">
                        {indicator.display}
                      </div>
                      <StatusIcon className={`w-5 h-5 ${STATUS_CONFIG[indicator.status].text}`} />
                    </div>

                    {/* Value */}
                    <div className="text-2xl font-bold text-slate-900 dark:text-white mb-1">
                      {indicator.value.toFixed(2)}{indicator.unit !== 'K' && indicator.unit !== 'N/A' && indicator.unit !== 'index' ? indicator.unit : ''}
                      {indicator.unit === 'K' ? 'K' : ''}
                    </div>

                    {/* Change */}
                    <div className={`flex items-center text-xs ${changeColor} mb-2`}>
                      <ChangeIcon className="w-3 h-3 mr-1" />
                      {changePct > 0 ? '+' : ''}{changePct.toFixed(1)}% from previous
                    </div>

                    {/* Meta Info */}
                    <div className="space-y-1 text-xs text-slate-600 dark:text-slate-400">
                      <div className="flex items-center">
                        <Info className="w-3 h-3 mr-1" />
                        {indicator.source} • {indicator.lead_lag}
                      </div>
                      <div>As of {new Date(indicator.date).toLocaleDateString()}</div>
                    </div>

                    {/* Status Badge */}
                    <div className={`mt-2 inline-block px-2 py-1 rounded text-xs font-medium ${STATUS_CONFIG[indicator.status].bg} ${STATUS_CONFIG[indicator.status].text}`}>
                      {indicator.status.charAt(0).toUpperCase() + indicator.status.slice(1)}
                    </div>
                  </div>
                );
              })}
            </div>
          </motion.div>
        );
      })}

      {/* Legend */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="bg-white dark:bg-slate-800 rounded-lg shadow p-4"
      >
        <h4 className="font-semibold text-slate-900 dark:text-white mb-3">Indicator Status Guide</h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
          <div className="flex items-center">
            <CheckCircle className="w-5 h-5 mr-2 text-green-600" />
            <div>
              <div className="font-medium text-slate-900 dark:text-white">Healthy</div>
              <div className="text-slate-600 dark:text-slate-400">Within normal historical range</div>
            </div>
          </div>
          <div className="flex items-center">
            <AlertTriangle className="w-5 h-5 mr-2 text-yellow-600" />
            <div>
              <div className="font-medium text-slate-900 dark:text-white">Warning</div>
              <div className="text-slate-600 dark:text-slate-400">Approaching extreme levels</div>
            </div>
          </div>
          <div className="flex items-center">
            <TrendingDown className="w-5 h-5 mr-2 text-red-600" />
            <div>
              <div className="font-medium text-slate-900 dark:text-white">Alert</div>
              <div className="text-slate-600 dark:text-slate-400">At extreme historical levels</div>
            </div>
          </div>
        </div>
        <div className="mt-3 pt-3 border-t border-slate-200 dark:border-slate-700">
          <div className="text-xs text-slate-600 dark:text-slate-400">
            <strong className="text-slate-900 dark:text-white">Leading indicators:</strong> Predict future economic activity (e.g., Housing Starts, S&P 500)
          </div>
          <div className="text-xs text-slate-600 dark:text-slate-400">
            <strong className="text-slate-900 dark:text-white">Coincident indicators:</strong> Reflect current economic activity (e.g., GDP, Retail Sales)
          </div>
          <div className="text-xs text-slate-600 dark:text-slate-400">
            <strong className="text-slate-900 dark:text-white">Lagging indicators:</strong> Confirm past economic activity (e.g., Unemployment, CPI)
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default MacroIndicatorDashboard;
