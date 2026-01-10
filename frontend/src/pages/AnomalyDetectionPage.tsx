/**
 * Anomaly Detection Page
 * Displays unusual predictions with economic context explanations
 */

import { FC, useState } from 'react';
import { motion } from 'framer-motion';
import {
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Filter,
  Calendar,
  BarChart3,
} from 'lucide-react';
import { AnomalyExplanation } from '../components/AnomalyExplanation';
import { useAnomalyDetection } from '../hooks/useAnomalyDetection';
import { useQuery } from '@tanstack/react-query';
import { predictionsApi } from '../api/unifiedApi';

const AnomalyDetectionPage: FC = () => {
  const [selectedCategory, setSelectedCategory] = useState('total_sales');
  const [severityFilter, setSeverityFilter] = useState<'all' | 'moderate' | 'severe'>('all');
  const [typeFilter, setTypeFilter] = useState<'all' | 'surge' | 'decline'>('all');

  // Fetch predictions for anomaly detection
  const { data: predictions, isLoading } = useQuery({
    queryKey: ['predictions', selectedCategory],
    queryFn: () => predictionsApi.getPredictions(selectedCategory, 100),
  });

  // Detect anomalies from actual prediction data
  const anomalies = (() => {
    if (!predictions || predictions.length === 0) return [];

    const hasPredictedValue = predictions[0].predicted_value !== undefined;

    return predictions.filter((p, i) => {
      if (i === 0) return false;
      const prev = predictions[i - 1];
      const currentValue = hasPredictedValue ? p.predicted_value : p.value;
      const previousValue = hasPredictedValue ? prev.predicted_value : prev.value;

      if (!currentValue || !previousValue) return false;

      const change = Math.abs(((currentValue - previousValue) / previousValue) * 100);
      return change > 5;
    }).map((p, i, arr) => {
      const predIndex = predictions.indexOf(p);
      const prev = predictions[predIndex - 1];
      const currentValue = hasPredictedValue ? p.predicted_value : p.value;
      const previousValue = hasPredictedValue ? prev.predicted_value : prev.value;

      const change = ((currentValue - previousValue) / previousValue) * 100;
      const changeMagnitude = Math.abs(change);
      const severity = changeMagnitude > 10 ? 'severe' : 'moderate';
      const type = change > 0 ? 'surge' : 'decline';

      return {
        id: `${p.date}-${selectedCategory}`,
        date: p.date || p.prediction_date,
        predicted_value: currentValue,
        actual_value: p.actual_value,
        change_percent: change,
        model_name: p.model_name || p.model_type || 'LGBM',
        category: selectedCategory,
        severity,
        type,
      };
    });
  })();

  // Apply filters
  const filteredAnomalies = anomalies.filter((a: any) => {
    if (severityFilter !== 'all' && a.severity !== severityFilter) return false;
    if (typeFilter !== 'all' && a.type !== typeFilter) return false;
    return true;
  });

  const stats = {
    total: anomalies.length,
    surges: anomalies.filter((a: any) => a.type === 'surge').length,
    declines: anomalies.filter((a: any) => a.type === 'decline').length,
    severe: anomalies.filter((a: any) => a.severity === 'severe').length,
  };

  const categories = [
    { value: 'total_sales', label: 'Total Retail Sales' },
    { value: 'automobile_dealers', label: 'Automobile Dealers' },
    { value: 'building_materials', label: 'Building Materials' },
    { value: 'clothing_accessories', label: 'Clothing & Accessories' },
    { value: 'electronics_appliances', label: 'Electronics & Appliances' },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100">
            Anomaly Detection
          </h1>
          <p className="text-slate-600 dark:text-slate-400 mt-1">
            Unusual predictions explained with economic context
          </p>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-xl p-6 border border-blue-200 dark:border-blue-800"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-blue-600 dark:text-blue-400">
                Total Anomalies
              </p>
              <p className="text-3xl font-bold text-blue-900 dark:text-blue-100 mt-2">
                {stats.total}
              </p>
            </div>
            <div className="p-3 bg-blue-500 rounded-lg">
              <AlertTriangle className="w-6 h-6 text-white" />
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 rounded-xl p-6 border border-green-200 dark:border-green-800"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-green-600 dark:text-green-400">
                Sales Surges
              </p>
              <p className="text-3xl font-bold text-green-900 dark:text-green-100 mt-2">
                {stats.surges}
              </p>
            </div>
            <div className="p-3 bg-green-500 rounded-lg">
              <TrendingUp className="w-6 h-6 text-white" />
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-gradient-to-br from-red-50 to-red-100 dark:from-red-900/20 dark:to-red-800/20 rounded-xl p-6 border border-red-200 dark:border-red-800"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-red-600 dark:text-red-400">
                Sales Declines
              </p>
              <p className="text-3xl font-bold text-red-900 dark:text-red-100 mt-2">
                {stats.declines}
              </p>
            </div>
            <div className="p-3 bg-red-500 rounded-lg">
              <TrendingDown className="w-6 h-6 text-white" />
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-gradient-to-br from-orange-50 to-orange-100 dark:from-orange-900/20 dark:to-orange-800/20 rounded-xl p-6 border border-orange-200 dark:border-orange-800"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-orange-600 dark:text-orange-400">
                Severe Events
              </p>
              <p className="text-3xl font-bold text-orange-900 dark:text-orange-100 mt-2">
                {stats.severe}
              </p>
            </div>
            <div className="p-3 bg-orange-500 rounded-lg">
              <BarChart3 className="w-6 h-6 text-white" />
            </div>
          </div>
        </motion.div>
      </div>

      {/* Filters */}
      <div className="bg-white dark:bg-slate-800 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
        <div className="flex flex-wrap gap-4 items-center">
          <div className="flex items-center gap-2">
            <Filter className="w-4 h-4 text-slate-500" />
            <span className="text-sm font-medium text-slate-700 dark:text-slate-300">
              Filters:
            </span>
          </div>

          {/* Category Filter */}
          <select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100 text-sm"
          >
            {categories.map((cat) => (
              <option key={cat.value} value={cat.value}>
                {cat.label}
              </option>
            ))}
          </select>

          {/* Severity Filter */}
          <select
            value={severityFilter}
            onChange={(e) => setSeverityFilter(e.target.value as any)}
            className="px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100 text-sm"
          >
            <option value="all">All Severities</option>
            <option value="moderate">Moderate (5-10%)</option>
            <option value="severe">Severe (&gt;10%)</option>
          </select>

          {/* Type Filter */}
          <select
            value={typeFilter}
            onChange={(e) => setTypeFilter(e.target.value as any)}
            className="px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100 text-sm"
          >
            <option value="all">All Types</option>
            <option value="surge">Surges</option>
            <option value="decline">Declines</option>
          </select>
        </div>
      </div>

      {/* Anomalies List */}
      <div className="space-y-4">
        {isLoading ? (
          <div className="text-center py-12">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            <p className="text-slate-600 dark:text-slate-400 mt-4">Loading anomalies...</p>
          </div>
        ) : filteredAnomalies.length === 0 ? (
          <div className="text-center py-12 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700">
            <AlertTriangle className="w-16 h-16 text-slate-400 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-2">
              No Anomalies Found
            </h3>
            <p className="text-slate-600 dark:text-slate-400">
              No unusual predictions detected for the selected filters.
            </p>
          </div>
        ) : (
          filteredAnomalies.map((anomaly: any, index: number) => (
            <motion.div
              key={anomaly.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.05 }}
            >
              <AnomalyExplanation
                date={anomaly.date}
                predictionChange={anomaly.change_percent}
                economicContext={anomaly.economicContext}
              />
            </motion.div>
          ))
        )}
      </div>

      {/* Info Box */}
      <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-xl p-4">
        <div className="flex gap-3">
          <AlertTriangle className="w-5 h-5 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <h4 className="font-semibold text-blue-900 dark:text-blue-100 mb-1">
              About Anomaly Detection
            </h4>
            <p className="text-sm text-blue-700 dark:text-blue-300">
              Anomalies are detected when predictions change by more than 5% from the previous period.
              Economic context is provided to help interpret these unusual changes. The economic indicators
              shown are for context only and are not used in model predictions.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnomalyDetectionPage;
