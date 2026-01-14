/**
 * Validation Page - Comprehensive Performance Dashboard
 * Professional ML model validation and monitoring interface
 */

import type { FC } from 'react';
import { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useQuery } from '@tanstack/react-query';
import toast from 'react-hot-toast';
import {
  CheckCircle2,
  XCircle,
  Clock,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  Filter,
  Download,
  Brain,
  Crosshair,
  Activity,
  BarChart3,
  FileText,
  RefreshCw,
  ChevronLeft,
  ChevronRight,
  X,
  Sparkles,
  Info,
  Zap,
} from 'lucide-react';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
  AreaChart,
} from 'recharts';
import { predictionsApi } from '../api/unifiedApi';

// Types
interface Prediction {
  id: number;
  prediction_date: string;
  model_name: string;
  store_id?: number;
  product_id?: number;
  predicted_value: number;
  actual_value?: number;
  error_absolute?: number;
  error_percentage?: number;
  is_validated: boolean;
  confidence_score?: number;
  confidence_interval_lower?: number;
  confidence_interval_upper?: number;
  created_at: string;
}

interface ValidationMetrics {
  overall_accuracy: number;
  avg_error_rate: number;
  predictions_validated: number;
  model_confidence: number;
  accuracy_trend: 'up' | 'down' | 'stable';
}

// Mock data (replace with actual API)
// const mockPredictions: Prediction[] = [
//   { id: 1, prediction_date: '2025-01-01', model_name: 'LightGBM_Auto', store_id: 1, product_id: 1, predicted_value: 45000, actual_value: 45234, error_absolute: 234, error_percentage: 0.52, is_validated: true, confidence_score: 0.95, created_at: '2025-01-01T00:00:00' },
//   { id: 2, prediction_date: '2025-01-02', model_name: 'LightGBM_Auto', store_id: 1, product_id: 1, predicted_value: 45100, actual_value: 44890, error_absolute: 210, error_percentage: 0.47, is_validated: true, confidence_score: 0.94, created_at: '2025-01-02T00:00:00' },
//   { id: 3, prediction_date: '2025-01-03', model_name: 'LightGBM_Auto', store_id: 1, product_id: 1, predicted_value: 45200, actual_value: 45678, error_absolute: 478, error_percentage: 1.05, is_validated: true, confidence_score: 0.92, created_at: '2025-01-03T00:00:00' },
//   { id: 4, prediction_date: '2025-01-04', model_name: 'LightGBM_Auto', store_id: 1, product_id: 1, predicted_value: 45300, is_validated: false, confidence_score: 0.91, created_at: '2025-01-04T00:00:00' },
//   { id: 5, prediction_date: '2025-01-05', model_name: 'LightGBM_Auto', store_id: 1, product_id: 1, predicted_value: 45400, actual_value: 44234, error_absolute: 1166, error_percentage: 2.57, is_validated: true, confidence_score: 0.88, created_at: '2025-01-05T00:00:00' },
//   { id: 6, prediction_date: '2025-01-06', model_name: 'LightGBM_Auto', store_id: 1, product_id: 1, predicted_value: 45500, actual_value: 45789, error_absolute: 289, error_percentage: 0.63, is_validated: true, confidence_score: 0.93, created_at: '2025-01-06T00:00:00' },
//   { id: 7, prediction_date: '2025-01-07', model_name: 'RandomForest_v2', store_id: 1, product_id: 1, predicted_value: 45600, actual_value: 45345, error_absolute: 255, error_percentage: 0.56, is_validated: true, confidence_score: 0.90, created_at: '2025-01-07T00:00:00' },
//   { id: 8, prediction_date: '2025-01-08', model_name: 'RandomForest_v2', store_id: 1, product_id: 1, predicted_value: 45700, actual_value: 46123, error_absolute: 423, error_percentage: 0.92, is_validated: true, confidence_score: 0.89, created_at: '2025-01-08T00:00:00' },
//   { id: 9, prediction_date: '2025-01-09', model_name: 'RandomForest_v2', store_id: 1, product_id: 1, predicted_value: 45800, actual_value: 45456, error_absolute: 344, error_percentage: 0.75, is_validated: true, confidence_score: 0.91, created_at: '2025-01-09T00:00:00' },
//   { id: 10, prediction_date: '2025-01-10', model_name: 'XGBoost_Pro', store_id: 1, product_id: 1, predicted_value: 45900, actual_value: 45234, error_absolute: 666, error_percentage: 1.45, is_validated: true, confidence_score: 0.92, created_at: '2025-01-10T00:00:00' },
// ];

type DateRangeType = '7d' | '30d' | '90d' | 'all';
type ModelFilterType = string[];

export const ValidationPage: FC = () => {
  const [dateRange, setDateRange] = useState<DateRangeType>('all');  // Show all predictions by default
  const [selectedModels, setSelectedModels] = useState<ModelFilterType>(['all']);
  const [selectedPrediction, setSelectedPrediction] = useState<Prediction | null>(null);
  const [timelineScroll, setTimelineScroll] = useState(0);
  const [isAutoValidating, setIsAutoValidating] = useState(false);

  // Fetch predictions
  const { data: predictionsData, isLoading, refetch } = useQuery({
    queryKey: ['predictions', dateRange, selectedModels],
    queryFn: async () => {
      // Calculate start date based on dateRange selection
      let startDate: string | undefined;
      if (dateRange !== 'all') {
        const days = parseInt(dateRange);
        const date = new Date();
        date.setDate(date.getDate() - days);
        startDate = date.toISOString().split('T')[0]; // YYYY-MM-DD format
      }

      const data = await predictionsApi.getHistory({
        start_date: startDate,
        limit: 15000, // Get all predictions including validated ones from 2025
      });
      return data;
    },
  });

  const predictions = predictionsData?.predictions || [];

  // Filter predictions
  const filteredPredictions = useMemo(() => {
    let filtered = [...predictions];

    // Date range filter
    if (dateRange !== 'all') {
      const days = parseInt(dateRange);
      const cutoffDate = new Date();
      cutoffDate.setDate(cutoffDate.getDate() - days);
      filtered = filtered.filter(p => new Date(p.prediction_date) >= cutoffDate);
    }

    // Model filter
    if (!selectedModels.includes('all')) {
      filtered = filtered.filter(p => selectedModels.includes(p.model_name));
    }

    // Data is already sorted by date ASC from backend, just return filtered
    return filtered;
  }, [predictions, dateRange, selectedModels]);

  // Calculate metrics
  const metrics = useMemo((): ValidationMetrics => {
    const validated = filteredPredictions.filter(p => p.is_validated && p.actual_value);

    if (validated.length === 0) {
      return {
        overall_accuracy: 0,
        avg_error_rate: 0,
        predictions_validated: 0,
        model_confidence: 0,
        accuracy_trend: 'stable',
      };
    }

    const accuracy = 100 - (validated.reduce((sum, p) => sum + (p.error_percentage || 0), 0) / validated.length);
    // @ts-ignore
    const avgError = validated.reduce((sum, p) => sum + ((p as any).error_absolute || 0), 0) / validated.length;
    // @ts-ignore
    const confidence = validated.reduce((sum, p) => sum + ((p as any).confidence_score || 0), 0) / validated.length;

    // Calculate trend
    const halfway = Math.floor(validated.length / 2);
    const firstHalf = validated.slice(0, halfway);
    const secondHalf = validated.slice(halfway);
    const firstHalfAcc = 100 - (firstHalf.reduce((sum, p) => sum + (p.error_percentage || 0), 0) / firstHalf.length);
    const secondHalfAcc = 100 - (secondHalf.reduce((sum, p) => sum + (p.error_percentage || 0), 0) / secondHalf.length);

    let trend: 'up' | 'down' | 'stable' = 'stable';
    if (secondHalfAcc > firstHalfAcc + 1) trend = 'up';
    else if (secondHalfAcc < firstHalfAcc - 1) trend = 'down';

    return {
      overall_accuracy: accuracy,
      avg_error_rate: avgError,
      predictions_validated: validated.length,
      model_confidence: confidence,
      accuracy_trend: trend,
    };
  }, [filteredPredictions]);

  // Prepare timeline data
  const timelineData = useMemo(() => {
    return filteredPredictions.map(p => ({
      ...p,
      status: !p.is_validated ? 'pending' : (p.error_percentage && p.error_percentage < 2 ? 'accurate' : 'inaccurate'),
    }));
  }, [filteredPredictions]);

  // Prepare error distribution data
  const errorDistribution = useMemo(() => {
    const validated = filteredPredictions.filter(p => p.is_validated && p.error_percentage);
    const bins = [0, 0, 0, 0, 0]; // 0-1%, 1-2%, 2-3%, 3-4%, >4%

    validated.forEach(p => {
      const error = p.error_percentage || 0;
      if (error < 1) bins[0]++;
      else if (error < 2) bins[1]++;
      else if (error < 3) bins[2]++;
      else if (error < 4) bins[3]++;
      else bins[4]++;
    });

    return [
      { range: '0-1%', count: bins[0] },
      { range: '1-2%', count: bins[1] },
      { range: '2-3%', count: bins[2] },
      { range: '3-4%', count: bins[3] },
      { range: '>4%', count: bins[4] },
    ];
  }, [filteredPredictions]);

  // Prepare scatter plot data with both validated and unvalidated predictions
  const scatterData = useMemo(() => {
    const validated = filteredPredictions
      .filter(p => p.is_validated && p.actual_value)
      .map(p => ({
        predicted: p.predicted_value,
        actual: p.actual_value!,
        model: p.model_name,
        validated: true,
        date: p.prediction_date,
      }));

    const unvalidated = filteredPredictions
      .filter(p => !p.is_validated || !p.actual_value)
      .map(p => ({
        predicted: p.predicted_value,
        actual: null,  // No actual value yet
        model: p.model_name,
        validated: false,
        date: p.prediction_date,
        // For unvalidated, estimate actual as predicted for positioning
        estimatedActual: p.predicted_value,
      }));

    return [...validated, ...unvalidated];
  }, [filteredPredictions]);

  // Worst predictions
  const worstPredictions = useMemo(() => {
    return [...filteredPredictions]
      .filter(p => p.is_validated && p.error_percentage)
      .sort((a, b) => (b.error_percentage || 0) - (a.error_percentage || 0))
      .slice(0, 5);
  }, [filteredPredictions]);

  // Accuracy degradation over time
  const accuracyOverTime = useMemo(() => {
    const validated = filteredPredictions.filter(p => p.is_validated && p.error_percentage);
    const grouped: Record<string, number[]> = {};

    validated.forEach(p => {
      const date = p.prediction_date.substring(0, 7); // YYYY-MM
      if (!grouped[date]) grouped[date] = [];
      grouped[date].push(100 - (p.error_percentage || 0));
    });

    return Object.entries(grouped)
      .map(([date, accuracies]) => ({
        date,
        accuracy: accuracies.reduce((sum, acc) => sum + acc, 0) / accuracies.length,
      }))
      .sort((a, b) => a.date.localeCompare(b.date));
  }, [filteredPredictions]);

  // Trend chart data
  const trendChartData = useMemo(() => {
    return filteredPredictions
      .filter(p => p.is_validated && p.error_percentage)
      .map(p => ({
        date: p.prediction_date,
        accuracy: 100 - (p.error_percentage || 0),
      }))
      .sort((a, b) => a.date.localeCompare(b.date));
  }, [filteredPredictions]);

  // Model health status (only consider production models: RandomForest and LGBM)
  const modelHealth = useMemo(() => {
    // Filter to only production models
    const productionPredictions = filteredPredictions.filter(p =>
      p.is_validated &&
      (p.model_name?.includes('randomforest') || p.model_name?.includes('lgbm'))
    );

    // Calculate accuracy per model, then average the models
    const modelAccuracies = productionPredictions.reduce((acc, p) => {
      const modelName = p.model_name || 'unknown';
      if (!acc[modelName]) acc[modelName] = [];
      acc[modelName].push(100 - (p.error_percentage || 0));
      return acc;
    }, {} as Record<string, number[]>);

    // Average each model's predictions, then average across models
    const avgModelAccuracies = Object.values(modelAccuracies).map(accuracies =>
      accuracies.reduce((sum, acc) => sum + acc, 0) / accuracies.length
    );

    const avgAccuracy = avgModelAccuracies.length > 0
      ? avgModelAccuracies.reduce((sum, acc) => sum + acc, 0) / avgModelAccuracies.length
      : 0;

    const threshold = 90;

    if (avgAccuracy < threshold && avgAccuracy > 0) {
      return {
        status: 'drift',
        message: `Production model accuracy (${avgAccuracy.toFixed(1)}%) below threshold (${threshold}%)`,
        severity: 'high',
      };
    }

    // Check for declining trend (last 3 time periods)
    const productionByDate = productionPredictions.reduce((acc, p) => {
      const date = p.prediction_date;
      if (!acc[date]) acc[date] = [];
      acc[date].push(p);
      return acc;
    }, {} as Record<string, typeof productionPredictions>);

    const recentTrend = Object.entries(productionByDate)
      .map(([date, preds]) => ({
        date,
        accuracy: preds.reduce((sum, p) => sum + (100 - (p.error_percentage || 0)), 0) / preds.length,
      }))
      .sort((a, b) => a.date.localeCompare(b.date))
      .slice(-3);

    if (recentTrend.length >= 3) {
      const decreasing = recentTrend.every((acc, i) =>
        i === 0 ? true : acc.accuracy < recentTrend[i - 1].accuracy
      );
      if (decreasing) {
        return {
          status: 'degrading',
          message: 'Production model accuracy showing declining trend',
          severity: 'medium',
        };
      }
    }

    return {
      status: 'healthy',
      message: `Production models performing well (${avgAccuracy.toFixed(1)}% average accuracy)`,
      severity: 'low',
    };
  }, [filteredPredictions]);

  // Get unique models
  const uniqueModels = useMemo(() => {
    return ['all', ...Array.from(new Set(predictions.map(p => p.model_name)))];
  }, [predictions]);

  // Handle export
  const handleExport = () => {
    toast.loading('Generating PDF report...', { id: 'export' });
    setTimeout(() => {
      toast.dismiss('export');
      toast.success('Report exported successfully!', {
        icon: <CheckCircle2 className="w-5 h-5 text-emerald-500" />,
        duration: 3000,
      });
    }, 2000);
  };

  const handleAutoValidate = async () => {
    setIsAutoValidating(true);
    try {
      toast.loading('Fetching actual values and validating predictions...', { id: 'auto-validate' });

      // @ts-ignore
      const results = await predictionsApi.autoValidate({
        category_id: '4400', // Total Retail Sales
        days_back: 90,
      });

      toast.dismiss('auto-validate');

      // @ts-ignore
      if (results.length === 0) {
        toast('No pending predictions found to validate', {
          icon: <Info className="w-5 h-5 text-blue-500" />,
          duration: 3000,
        });
      } else {
        // @ts-ignore
        toast.success(`Successfully validated ${results.length} predictions!`, {
          icon: <CheckCircle2 className="w-5 h-5 text-emerald-500" />,
          duration: 4000,
        });

        // Calculate average error
        // @ts-ignore
        const avgError = results.reduce((sum: any, r: any) => sum + (r.error_percentage || 0), 0) / results.length;
        // @ts-ignore
        const accurateCount = results.filter((r: any) => (r.error_percentage || 0) < 2).length;

        toast(`Average error: ${avgError.toFixed(2)}% | ${accurateCount}/${results.length} within 2%`, {
          icon: <Crosshair className="w-5 h-5 text-blue-500" />,
          duration: 5000,
        });
      }

      // Refetch predictions to show updated validation status
      await refetch();
    } catch (error: any) {
      toast.dismiss('auto-validate');
      toast.error(`Failed to auto-validate: ${error.message || 'Unknown error'}`, {
        icon: <XCircle className="w-5 h-5 text-red-500" />,
        duration: 4000,
      });
    } finally {
      setIsAutoValidating(false);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between"
      >
        <div>
          <h1 className="text-5xl font-bold gradient-text animate-gradient">
            Model Validation Dashboard
          </h1>
          <p className="text-slate-600 dark:text-slate-400 text-lg mt-2">
            Monitor model performance, track accuracy, and detect drift
          </p>
        </div>
        <div className="flex gap-3">
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={handleAutoValidate}
            disabled={isAutoValidating}
            className="px-6 py-3 bg-gradient-to-r from-emerald-600 to-teal-600 text-white rounded-xl font-medium shadow-lg shadow-emerald-500/50 flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isAutoValidating ? (
              <>
                <RefreshCw className="w-5 h-5 animate-spin" />
                Validating...
              </>
            ) : (
              <>
                <Zap className="w-5 h-5" />
                Auto-Validate
              </>
            )}
          </motion.button>
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={handleExport}
            className="px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-xl font-medium shadow-lg shadow-blue-500/50 flex items-center gap-2"
          >
            <Download className="w-5 h-5" />
            Export Report
          </motion.button>
        </div>
      </motion.div>

      {/* Interactive Filters */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="glass-card p-6"
      >
        <div className="flex items-center gap-3 mb-4">
          <Filter className="w-5 h-5 text-slate-600 dark:text-slate-400" />
          <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">Filters</h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Date Range */}
          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Date Range
            </label>
            <div className="flex gap-2">
              {(['7d', '30d', '90d', 'all'] as DateRangeType[]).map((range) => (
                <button
                  key={range}
                  onClick={() => setDateRange(range)}
                  className={`px-4 py-2 rounded-lg font-medium transition-all ${
                    dateRange === range
                      ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/50'
                      : 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700'
                  }`}
                >
                  {range === 'all' ? 'All Time' : `Last ${range.replace('d', ' days')}`}
                </button>
              ))}
            </div>
          </div>

          {/* Model Filter */}
          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Models
            </label>
            <select
              value={selectedModels.includes('all') ? 'all' : selectedModels[0] || 'all'}
              onChange={(e) => {
                if (e.target.value === 'all') {
                  setSelectedModels(['all']);
                } else {
                  setSelectedModels([e.target.value]);
                }
              }}
              className="w-full px-4 py-2 bg-slate-100 dark:bg-slate-800 border-0 rounded-lg text-slate-900 dark:text-slate-100 focus:ring-2 focus:ring-blue-500"
            >
              <option value="all">All Models</option>
              {uniqueModels.filter(m => m !== 'all').map(model => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))}
            </select>
          </div>
        </div>
      </motion.div>

      {/* Performance Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Overall Accuracy */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="glass-card p-6 relative overflow-hidden"
        >
          <div className="flex items-start justify-between">
            <div>
              <p className="text-sm text-slate-600 dark:text-slate-400 font-medium">Overall Accuracy</p>
              <div className="flex items-center gap-2 mt-2">
                <motion.h3
                  key={metrics.overall_accuracy}
                  initial={{ scale: 0.8, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  className="text-4xl font-bold text-slate-900 dark:text-slate-100"
                >
                  {metrics.overall_accuracy.toFixed(1)}%
                </motion.h3>
                {metrics.accuracy_trend === 'up' && (
                  <TrendingUp className="w-6 h-6 text-emerald-500" />
                )}
                {metrics.accuracy_trend === 'down' && (
                  <TrendingDown className="w-6 h-6 text-red-500" />
                )}
                {metrics.accuracy_trend === 'stable' && (
                  <Activity className="w-6 h-6 text-blue-500" />
                )}
              </div>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-2">
                {metrics.accuracy_trend === 'up' && 'Improving'}
                {metrics.accuracy_trend === 'down' && 'Declining'}
                {metrics.accuracy_trend === 'stable' && 'Stable'}
              </p>
            </div>
            <div className="p-3 bg-emerald-100 dark:bg-emerald-900/20 rounded-xl">
              <Crosshair className="w-8 h-8 text-emerald-600" />
            </div>
          </div>

          {/* Mini trend chart */}
          <div className="mt-4 h-16">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={trendChartData.slice(-10)}>
                <defs>
                  <linearGradient id="colorAccuracy" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <Area type="monotone" dataKey="accuracy" stroke="#10b981" fill="url(#colorAccuracy)" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* Average Error Rate */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.25 }}
          className="glass-card p-6 relative overflow-hidden"
        >
          <div className="flex items-start justify-between">
            <div>
              <p className="text-sm text-slate-600 dark:text-slate-400 font-medium">Avg Error (MAE)</p>
              <motion.h3
                key={metrics.avg_error_rate}
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                className="text-4xl font-bold text-slate-900 dark:text-slate-100 mt-2"
              >
                ${metrics.avg_error_rate.toFixed(2)}
              </motion.h3>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-2">
                Mean absolute error
              </p>
            </div>
            <div className="p-3 bg-amber-100 dark:bg-amber-900/20 rounded-xl">
              <BarChart3 className="w-8 h-8 text-amber-600" />
            </div>
          </div>

          {/* Mini trend chart */}
          <div className="mt-4 h-16">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={trendChartData.slice(-10)}>
                <defs>
                  <linearGradient id="colorError" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#f59e0b" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <Area type="monotone" dataKey={(d) => 100 - d.accuracy} stroke="#f59e0b" fill="url(#colorError)" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* Predictions Validated */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="glass-card p-6 relative overflow-hidden"
        >
          <div className="flex items-start justify-between">
            <div>
              <p className="text-sm text-slate-600 dark:text-slate-400 font-medium">Validated This Month</p>
              <motion.h3
                key={metrics.predictions_validated}
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                className="text-4xl font-bold text-slate-900 dark:text-slate-100 mt-2"
              >
                {metrics.predictions_validated}
              </motion.h3>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-2">
                {filteredPredictions.filter(p => !p.is_validated).length} pending
              </p>
            </div>
            <div className="p-3 bg-blue-100 dark:bg-blue-900/20 rounded-xl">
              <CheckCircle2 className="w-8 h-8 text-blue-600" />
            </div>
          </div>

          {/* Mini progress bar */}
          <div className="mt-4 h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${(metrics.predictions_validated / filteredPredictions.length) * 100}%` }}
              transition={{ delay: 0.5, duration: 0.5 }}
              className="h-full bg-blue-600 rounded-full"
            />
          </div>
        </motion.div>

        {/* Model Confidence */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.35 }}
          className="glass-card p-6 relative overflow-hidden"
        >
          <div className="flex items-start justify-between">
            <div>
              <p className="text-sm text-slate-600 dark:text-slate-400 font-medium">Model Confidence</p>
              <motion.h3
                key={metrics.model_confidence}
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                className="text-4xl font-bold text-slate-900 dark:text-slate-100 mt-2"
              >
                {metrics.model_confidence.toFixed(1)}%
              </motion.h3>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-2">
                Average confidence score
              </p>
            </div>
            <div className="p-3 bg-purple-100 dark:bg-purple-900/20 rounded-xl">
              <Brain className="w-8 h-8 text-purple-600" />
            </div>
          </div>

          {/* Mini gauge */}
          <div className="mt-4 h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${metrics.model_confidence}%` }}
              transition={{ delay: 0.5, duration: 0.5 }}
              className="h-full bg-purple-600 rounded-full"
            />
          </div>
        </motion.div>
      </div>

      {/* Timeline View */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="glass-card p-6"
      >
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-2xl font-bold text-slate-900 dark:text-slate-100">
              Prediction Timeline
            </h2>
            <p className="text-slate-600 dark:text-slate-400 text-sm mt-1">
              Click on any dot to view prediction details
            </p>
          </div>
          <div className="flex items-center gap-4 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-slate-300 dark:bg-slate-600" />
              <span className="text-slate-600 dark:text-slate-400">Pending</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-emerald-500" />
              <span className="text-slate-600 dark:text-slate-400">Accurate (&lt;2%)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-red-500" />
              <span className="text-slate-600 dark:text-slate-400">Inaccurate (&gt;2%)</span>
            </div>
          </div>
        </div>

        {/* Horizontal Timeline */}
        <div className="relative">
          {/* Scroll buttons */}
          <button
            onClick={() => setTimelineScroll(Math.max(0, timelineScroll - 200))}
            className="absolute left-0 top-1/2 -translate-y-1/2 z-10 p-2 bg-white dark:bg-slate-800 rounded-lg shadow-lg hover:shadow-xl transition-shadow"
          >
            <ChevronLeft className="w-5 h-5 text-slate-600 dark:text-slate-400" />
          </button>
          <button
            onClick={() => setTimelineScroll(Math.min(timelineScroll + 200, timelineData.length * 60 - 800))}
            className="absolute right-0 top-1/2 -translate-y-1/2 z-10 p-2 bg-white dark:bg-slate-800 rounded-lg shadow-lg hover:shadow-xl transition-shadow"
          >
            <ChevronRight className="w-5 h-5 text-slate-600 dark:text-slate-400" />
          </button>

          {/* Timeline */}
          <div className="overflow-hidden mx-10">
            <motion.div
              animate={{ x: -timelineScroll }}
              transition={{ type: 'spring', stiffness: 300, damping: 30 }}
              className="flex gap-6"
            >
              {timelineData.map((prediction) => (
                <motion.button
                  key={prediction.id}
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  onClick={() => setSelectedPrediction(prediction)}
                  className={`flex-shrink-0 w-12 h-12 rounded-full border-4 transition-all ${
                    prediction.status === 'pending'
                      ? 'bg-slate-300 dark:bg-slate-600 border-slate-400 dark:border-slate-500'
                      : prediction.status === 'accurate'
                      ? 'bg-emerald-500 border-emerald-300 shadow-lg shadow-emerald-500/50'
                      : 'bg-red-500 border-red-300 shadow-lg shadow-red-500/50'
                  } ${selectedPrediction?.id === prediction.id ? 'ring-4 ring-blue-500 ring-offset-2' : ''}`}
                  title={`${prediction.prediction_date}\n${prediction.model_name}\n${prediction.status === 'pending' ? 'Pending validation' : `Error: ${prediction.error_percentage?.toFixed(2)}%`}`}
                >
                  <span className="sr-only">{prediction.prediction_date}</span>
                </motion.button>
              ))}
            </motion.div>
          </div>

          {/* Date labels */}
          <div className="overflow-hidden mx-10 mt-2">
            <motion.div
              animate={{ x: -timelineScroll }}
              transition={{ type: 'spring', stiffness: 300, damping: 30 }}
              className="flex gap-6"
            >
              {timelineData.map((prediction) => (
                <div key={prediction.id} className="flex-shrink-0 w-12 text-center">
                  <span className="text-xs text-slate-500 dark:text-slate-400">
                    {new Date(prediction.prediction_date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                  </span>
                </div>
              ))}
            </motion.div>
          </div>
        </div>
      </motion.div>

      {/* Prediction Detail Side Panel */}
      <AnimatePresence>
        {selectedPrediction && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
            onClick={() => setSelectedPrediction(null)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              className="glass-card max-w-md w-full p-6 relative"
            >
              <button
                onClick={() => setSelectedPrediction(null)}
                className="absolute top-4 right-4 p-2 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-lg transition-colors"
              >
                <X className="w-5 h-5 text-slate-600 dark:text-slate-400" />
              </button>

              <div className="space-y-4">
                <div>
                  <p className="text-sm text-slate-600 dark:text-slate-400">Prediction ID</p>
                  <p className="text-2xl font-bold text-slate-900 dark:text-slate-100">#{selectedPrediction.id}</p>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-slate-600 dark:text-slate-400">Date</p>
                    <p className="text-lg font-semibold text-slate-900 dark:text-slate-100">
                      {new Date(selectedPrediction.prediction_date).toLocaleDateString()}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-slate-600 dark:text-slate-400">Model</p>
                    <p className="text-lg font-semibold text-slate-900 dark:text-slate-100">
                      {selectedPrediction.model_name}
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-slate-600 dark:text-slate-400">Predicted</p>
                    <p className="text-xl font-bold text-blue-600">
                      ${selectedPrediction.predicted_value.toFixed(2)}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-slate-600 dark:text-slate-400">Actual</p>
                    <p className="text-xl font-bold text-emerald-600">
                      {selectedPrediction.actual_value ? `$${selectedPrediction.actual_value.toFixed(2)}` : 'Pending'}
                    </p>
                  </div>
                </div>

                {selectedPrediction.is_validated && selectedPrediction.error_percentage && (
                  <div className="p-4 bg-slate-50 dark:bg-slate-800/50 rounded-xl">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-sm text-slate-600 dark:text-slate-400">Error (Abs)</p>
                        <p className="text-lg font-semibold text-slate-900 dark:text-slate-100">
                          ${selectedPrediction.error_absolute?.toFixed(2)}
                        </p>
                      </div>
                      <div>
                        <p className="text-sm text-slate-600 dark:text-slate-400">Error (%)</p>
                        <p className={`text-lg font-semibold ${
                          selectedPrediction.error_percentage < 2 ? 'text-emerald-600' : 'text-red-600'
                        }`}>
                          {selectedPrediction.error_percentage.toFixed(2)}%
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                <div>
                  <p className="text-sm text-slate-600 dark:text-slate-400">Confidence Score</p>
                  <div className="flex items-center gap-3">
                    <div className="flex-1 h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${(selectedPrediction.confidence_score || 0) * 100}%` }}
                        className="h-full bg-purple-600 rounded-full"
                      />
                    </div>
                    <span className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                      {((selectedPrediction.confidence_score || 0) * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Error Analysis Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Error Distribution Histogram */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="glass-card p-6"
        >
          <h3 className="text-xl font-bold text-slate-900 dark:text-slate-100 mb-4">
            Error Distribution
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={errorDistribution}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="range" stroke="#64748b" />
              <YAxis stroke="#64748b" label={{ value: 'Count', angle: -90, position: 'insideLeft' }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(15, 23, 42, 0.95)',
                  border: 'none',
                  borderRadius: '8px',
                  color: '#fff',
                }}
              />
              <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </motion.div>

        {/* Predicted vs Actual Scatter Plot */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.55 }}
          className="glass-card p-6"
        >
          <h3 className="text-xl font-bold text-slate-900 dark:text-slate-100 mb-4">
            Predicted vs Actual
          </h3>
          <div className="flex items-center gap-4 mb-4 text-sm">
            <div className="flex items-center">
              <div className="w-3 h-3 bg-blue-500 mr-2 rounded-full"></div>
              <span className="text-slate-600 dark:text-slate-400">Validated</span>
            </div>
            <div className="flex items-center">
              <div className="w-3 h-3 bg-gray-400 mr-2 rounded-full"></div>
              <span className="text-slate-600 dark:text-slate-400">Pending Validation</span>
            </div>
            <div className="flex items-center">
              <div className="w-8 h-0.5 bg-red-500 mr-2" style={{transform: 'rotate(-45deg)'}}></div>
              <span className="text-slate-600 dark:text-slate-400">Perfect Prediction</span>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid stroke="#e2e8f0" />
              <XAxis
                type="number"
                dataKey="predicted"
                name="Predicted"
                stroke="#64748b"
                label={{ value: 'Predicted $', position: 'insideBottom', offset: -5 }}
              />
              <YAxis
                type="number"
                dataKey="actual"
                name="Actual"
                stroke="#64748b"
                label={{ value: 'Actual $', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip
                cursor={{ strokeDasharray: '3 3' }}
                contentStyle={{
                  backgroundColor: '#1e293b',
                  border: '1px solid #475569',
                  borderRadius: '8px',
                  padding: '12px',
                }}
                itemStyle={{
                  color: '#f1f5f9',
                }}
                labelStyle={{
                  color: '#f1f5f9',
                  fontWeight: 'bold',
                  marginBottom: '8px',
                }}
                formatter={(_value: number | undefined, _name: string | undefined, props: any) => {
                  // Recharts Scatter tooltip gives us the active point
                  const payload = props?.payload;
                  if (!payload) return null;

                  // Get all data from the payload
                  const dataPoint = payload;
                  const date = dataPoint.date || 'N/A';
                  const predicted = dataPoint.predicted || dataPoint.payload?.predicted || 0;
                  const actual = dataPoint.actual || dataPoint.payload?.actual;

                  // Build a rich tooltip with all information
                  const result = [
                    <div key="tooltip" style={{ width: '200px', color: '#f1f5f9' }}>
                      <div style={{ marginBottom: '8px', fontWeight: 'bold', fontSize: '12px', color: '#f1f5f9' }}>
                        {date}
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px', color: '#f1f5f9' }}>
                        <span>Predicted:</span>
                        <span style={{ fontWeight: 'bold', color: '#f1f5f9' }}>${predicted?.toFixed(2) || '0.00'}</span>
                      </div>
                      {actual !== null && actual !== undefined ? (
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px', color: '#f1f5f9' }}>
                          <span>Actual:</span>
                          <span style={{ fontWeight: 'bold', color: '#f1f5f9' }}>${actual?.toFixed(2) || '0.00'}</span>
                        </div>
                      ) : (
                        <div style={{ fontSize: '11px', fontStyle: 'italic', color: '#fbbf24' }}>
                          Pending validation
                        </div>
                      )}
                    </div>
                  ];

                  return result;
                }}
                labelFormatter={() => ''}  // We handle the label in formatter
              />
              <Scatter name="Validated" data={scatterData.filter(d => d.validated)} fill="#3b82f6" />
              <Scatter name="Pending" data={scatterData.filter(d => !d.validated).map(d => ({...d, actual: (d as any).estimatedActual ?? d.predicted}))} fill="#9ca3af" />
              {/* Perfect prediction line */}
              <Line
                type="monotone"
                dataKey="predicted"
                stroke="#ef4444"
                strokeWidth={2}
                strokeDasharray="5 5"
                dot={false}
              />
              <Legend />
            </ScatterChart>
          </ResponsiveContainer>
        </motion.div>
      </div>

      {/* Worst Predictions Table */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
        className="glass-card overflow-hidden"
      >
        <div className="p-6 border-b border-slate-200 dark:border-slate-700">
          <h3 className="text-xl font-bold text-slate-900 dark:text-slate-100">
            Worst Predictions
          </h3>
          <p className="text-slate-600 dark:text-slate-400 text-sm mt-1">
            Top 5 predictions with highest error rates
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-slate-50 dark:bg-slate-800/50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-semibold text-slate-700 dark:text-slate-300 uppercase">
                  Date
                </th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-slate-700 dark:text-slate-300 uppercase">
                  Model
                </th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-slate-700 dark:text-slate-300 uppercase">
                  Predicted
                </th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-slate-700 dark:text-slate-300 uppercase">
                  Actual
                </th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-slate-700 dark:text-slate-300 uppercase">
                  Error %
                </th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-slate-700 dark:text-slate-300 uppercase">
                  Explanation
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 dark:divide-slate-700">
              {worstPredictions.length > 0 ? worstPredictions.map((prediction) => (
                <tr key={prediction.id} className="hover:bg-slate-50 dark:hover:bg-slate-800/30 transition-colors">
                  <td className="px-6 py-4 text-sm text-slate-900 dark:text-slate-100">
                    {new Date(prediction.prediction_date).toLocaleDateString()}
                  </td>
                  <td className="px-6 py-4 text-sm text-slate-900 dark:text-slate-100">
                    {prediction.model_name}
                  </td>
                  <td className="px-6 py-4 text-sm text-blue-600 font-semibold">
                    ${prediction.predicted_value.toFixed(2)}
                  </td>
                  <td className="px-6 py-4 text-sm text-emerald-600 font-semibold">
                    ${prediction.actual_value?.toFixed(2)}
                  </td>
                  <td className="px-6 py-4">
                    <span className="px-3 py-1 bg-red-100 dark:bg-red-900/20 text-red-700 dark:text-red-400 text-sm font-semibold rounded-full">
                      {prediction.error_percentage?.toFixed(2)}%
                    </span>
                  </td>
                  <td className="px-6 py-4 text-sm text-slate-600 dark:text-slate-400">
                    {prediction.error_percentage && prediction.error_percentage > 3
                      ? 'Significant market deviation'
                      : 'Seasonal variation'}
                  </td>
                </tr>
              )) : (
                <tr>
                  <td colSpan={6} className="px-6 py-12 text-center text-slate-500 dark:text-slate-400">
                    No validated predictions found
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </motion.div>

      {/* Model Health Monitor */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.65 }}
        className={`glass-card p-6 border-l-4 ${
          modelHealth.severity === 'high'
            ? 'border-red-500'
            : modelHealth.severity === 'medium'
            ? 'border-amber-500'
            : 'border-emerald-500'
        }`}
      >
        <div className="flex items-start justify-between">
          <div className="flex items-start gap-4">
            <div className={`p-3 rounded-xl ${
              modelHealth.severity === 'high'
                ? 'bg-red-100 dark:bg-red-900/20'
                : modelHealth.severity === 'medium'
                ? 'bg-amber-100 dark:bg-amber-900/20'
                : 'bg-emerald-100 dark:bg-emerald-900/20'
            }`}>
              {modelHealth.severity === 'high' ? (
                <AlertTriangle className="w-8 h-8 text-red-600" />
              ) : modelHealth.severity === 'medium' ? (
                <Clock className="w-8 h-8 text-amber-600" />
              ) : (
                <CheckCircle2 className="w-8 h-8 text-emerald-600" />
              )}
            </div>

            <div>
              <h3 className="text-xl font-bold text-slate-900 dark:text-slate-100">
                Model Health: {modelHealth.status === 'healthy' ? 'Good' : modelHealth.status === 'degrading' ? 'Warning' : 'Critical'}
              </h3>
              <p className="text-slate-600 dark:text-slate-400 mt-1">
                {modelHealth.message}
              </p>
            </div>
          </div>
        </div>

        {/* Accuracy degradation chart */}
        {accuracyOverTime.length > 1 && (
          <div className="mt-6">
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={accuracyOverTime}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="date" stroke="#64748b" />
                <YAxis
                  stroke="#64748b"
                  domain={[90, 100]}
                  label={{ value: 'Accuracy %', angle: -90, position: 'insideLeft' }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(15, 23, 42, 0.95)',
                    border: 'none',
                    borderRadius: '8px',
                    color: '#fff',
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="accuracy"
                  stroke={modelHealth.severity === 'high' ? '#ef4444' : modelHealth.severity === 'medium' ? '#f59e0b' : '#10b981'}
                  strokeWidth={3}
                  dot={{ fill: modelHealth.severity === 'high' ? '#ef4444' : modelHealth.severity === 'medium' ? '#f59e0b' : '#10b981', r: 5 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </motion.div>

      {/* Empty State */}
      {filteredPredictions.length === 0 && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="glass-card p-16 text-center"
        >
          <div className="w-24 h-24 bg-slate-100 dark:bg-slate-800 rounded-full flex items-center justify-center mx-auto mb-6">
            <FileText className="w-12 h-12 text-slate-400" />
          </div>
          <h3 className="text-2xl font-bold text-slate-900 dark:text-slate-100 mb-2">
            No Predictions Found
          </h3>
          <p className="text-slate-600 dark:text-slate-400 mb-6">
            {dateRange !== 'all' || !selectedModels.includes('all')
              ? 'Try adjusting your filters to see more predictions'
              : 'Generate predictions from the Predictions page to get started'}
          </p>
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            className="px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-xl font-medium shadow-lg shadow-blue-500/50 inline-flex items-center gap-2"
          >
            <Sparkles className="w-5 h-5" />
            Go to Predictions Page
          </motion.button>
        </motion.div>
      )}
    </div>
  );
};
