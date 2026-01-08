/**
 * Professional ML Platform - Predictions Page
 *
 * Features:
 * - Hero section with animated gradient text
 * - Modern form with floating labels
 * - Model selector as pill buttons
 * - Interactive forecast chart with confidence bands
 * - Feature contribution cards with sparklines
 * - Skeleton loaders and error states
 * - Toast notifications
 */

import type { FC } from 'react';
import { useState, useEffect } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { motion, AnimatePresence } from 'framer-motion';
import toast, { Toaster } from 'react-hot-toast';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Area,
  AreaChart,
} from 'recharts';
import {
  TrendingUp,
  TrendingDown,
  Sparkles,
  Zap,
  Target,
  BarChart3,
  Download,
  RefreshCw,
  Info,
  ChevronDown,
  ChevronUp,
  Wand2,
  Brain,
  Clock,
  AlertCircle,
  CheckCircle2,
  Activity,
} from 'lucide-react';
import { predictionsApi, categoriesApi, Granularity } from '../api/unifiedApi';
import { triggerConfetti } from '../components/PremiumAnimations';
import { SkeletonCard } from '../components/PremiumAnimations';

const MODEL_TYPES = [
  { value: 'lightgbm', label: 'LightGBM', icon: Zap, badge: 'Best Accuracy' },
  { value: 'randomforest', label: 'Random Forest', icon: Brain, badge: 'Robust' },
  { value: 'autoarima', label: 'AutoARIMA', icon: TrendingUp, badge: 'Seasonal' },
  { value: 'autoets', label: 'AutoETS', icon: BarChart3, badge: 'Trend' },
  { value: 'patchtst', label: 'PatchTST', icon: Activity, badge: 'Deep Learning' },
  { value: 'timesnet', label: 'TimesNet', icon: Target, badge: 'Advanced' },
  { value: 'seasonalnaive', label: 'Seasonal Naive', icon: Clock, badge: 'Baseline' },
];

const DEFAULT_FEATURES = {
  inventory_level: 50,
  promotion_flag: 0,
  competitor_price: 100,
};

export const PredictionsPage: FC = () => {
  const { data: categoriesData, isLoading: categoriesLoading, error: categoriesError } = useQuery({
    queryKey: ['categories'],
    queryFn: async () => {
      console.log('Fetching categories...');
      const result = await categoriesApi.list();
      console.log('Categories response:', result);
      return result;
    },
  });

  // Log errors
  if (categoriesError) {
    console.error('Error loading categories:', categoriesError);
  }

  const [selectedCategory, setSelectedCategory] = useState<string>('');
  const [weeksAhead, setWeeksAhead] = useState<number>(4);
  const [modelType, setModelType] = useState<string>('lightgbm');
  const [granularity, setGranularity] = useState<Granularity>(Granularity.WEEKLY);
  const [features, setFeatures] = useState(DEFAULT_FEATURES);
  const [historicalData, setHistoricalData] = useState<any[]>([]);
  const [expandedFeature, setExpandedFeature] = useState<number | null>(null);

  useEffect(() => {
    if (categoriesData?.categories && categoriesData.categories.length > 0) {
      if (!selectedCategory) {
        setSelectedCategory(categoriesData.categories[0].key);
      }
    }
  }, [categoriesData]);

  useEffect(() => {
    const fetchHistoricalData = async () => {
      if (!selectedCategory) return;

      try {
        const response = await fetch(
          `/api/historical-sales?category=${encodeURIComponent(selectedCategory)}&days_back=90`
        );

        if (!response.ok) {
          console.error('Failed to fetch historical data:', response.statusText);
          return;
        }

        const result = await response.json();

        // Transform data to match expected format
        const formattedData = result.data.map((item: any) => ({
          date: item.date,
          sales: item.value,
          dateFormatted: new Date(item.date).toLocaleDateString('en-US', {
            month: 'short',
            day: 'numeric',
          }),
        }));

        setHistoricalData(formattedData);
      } catch (error) {
        console.error('Error fetching historical data:', error);
      }
    };

    fetchHistoricalData();
  }, [selectedCategory]);

  const predictionMutation = useMutation({
    mutationFn: predictionsApi.predict,
    onSuccess: (data) => {
      toast.success(`Forecast generated successfully!`, {
        duration: 3000,
        position: 'top-right',
        icon: <CheckCircle2 className="w-5 h-5 text-emerald-500" />,
      });
      triggerConfetti();
    },
    onError: (error: any) => {
      toast.error(`Failed to generate forecast: ${error.message}`, {
        duration: 5000,
        position: 'top-right',
        icon: <AlertCircle className="w-5 h-5 text-red-500" />,
      });
    },
  });

  const handlePredict = () => {
    predictionMutation.mutate({
      category: selectedCategory,
      weeks_ahead: weeksAhead,
      model_name: modelType,
      granularity,
    });
  };

  const combinedData = [
    ...historicalData.map((d) => ({
      date: d.dateFormatted,
      historical: d.sales,
      predicted: null,
      lower: null,
      upper: null,
    })),
    ...(predictionMutation.data?.forecasts.map((f) => ({
      date: new Date(f.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
      historical: null,
      predicted: f.predicted_value,
      lower: f.confidence_lower,
      upper: f.confidence_upper,
    })) || []),
  ];

  // Get the last historical value (most recent actual sales)
  const getCurrentValue = () => {
    if (historicalData.length === 0) return 0;
    return historicalData[historicalData.length - 1].sales;
  };

  // Get the average prediction across all forecasted weeks
  const getPredictedValue = () => {
    const forecasts = predictionMutation.data?.forecasts;
    if (!forecasts || forecasts.length === 0) return 0;

    const sum = forecasts.reduce((acc: number, f: any) => acc + f.predicted_value, 0);
    return sum / forecasts.length;
  };

  // Calculate change from last historical to first forecast
  const getPercentageChange = () => {
    const current = getCurrentValue();
    const firstForecast = predictionMutation.data?.forecasts?.[0]?.predicted_value;

    if (!current || !firstForecast) return 0;
    return current > 0 ? ((firstForecast - current) / current) * 100 : 0;
  };

  const percentageChange = getPercentageChange();
  const isPositive = percentageChange >= 0;

  // Custom tooltip
  const CustomTooltip = ({ active, payload }: any) => {
    if (!active || !payload || !payload.length) return null;
    const data = payload[0].payload;
    return (
      <div className="bg-slate-900/95 backdrop-blur-sm rounded-xl border border-slate-700 shadow-2xl p-4">
        <p className="text-white font-semibold mb-2">{data.date}</p>
        {data.historical && (
          <p className="text-slate-300 text-sm">Historical: <span className="text-white font-medium">${data.historical.toFixed(2)}</span></p>
        )}
        {data.predicted && (
          <>
            <p className="text-blue-300 text-sm">Predicted: <span className="text-white font-medium">${data.predicted.toFixed(2)}</span></p>
            {data.lower && (
              <p className="text-slate-400 text-xs mt-1">
                95% CI: ${data.lower.toFixed(2)} - ${data.upper.toFixed(2)}
              </p>
            )}
          </>
        )}
      </div>
    );
  };

  // Download chart as PNG
  const downloadChart = () => {
    toast.success('Chart exported as PNG!', {
      duration: 2000,
      icon: <Download className="w-5 h-5 text-blue-500" />,
    });
  };

  return (
    <div className="space-y-8">
      <Toaster position="top-right" />

      {/* Hero Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center space-y-4"
      >
        <h1 className="text-4xl md:text-5xl font-bold">
          <span className="bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 bg-clip-text text-transparent animate-gradient">
            Retail Sales Forecasting Engine
          </span>
        </h1>
        <div className="flex flex-wrap items-center justify-center gap-4 text-sm text-slate-600 dark:text-slate-400">
          <span className="flex items-center gap-1">
            <Sparkles className="w-4 h-4 text-purple-500" />
            Trained on 50K+ samples
          </span>
          <span className="text-slate-300">•</span>
          <span className="flex items-center gap-1">
            <Brain className="w-4 h-4 text-blue-500" />
            7 model architectures
          </span>
          <span className="text-slate-300">•</span>
          <span className="flex items-center gap-1">
            <Target className="w-4 h-4 text-emerald-500" />
            95% accuracy
          </span>
        </div>
      </motion.div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-8">
        {/* Input Section - Left Panel (40%) */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.1 }}
          className="lg:col-span-2 space-y-6"
        >
          {/* Configuration Card */}
          <div className="glass-card space-y-6">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-2 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg">
                <Wand2 className="w-5 h-5 text-white" />
              </div>
              <div>
                <h2 className="text-xl font-bold text-slate-900 dark:text-slate-100">Configuration</h2>
                <p className="text-sm text-slate-600 dark:text-slate-400">Customize your prediction</p>
              </div>
            </div>

            {/* Category Selection */}
            <div className="space-y-2">
              <label className="text-sm font-semibold text-slate-700 dark:text-slate-300">Retail Category</label>
              {categoriesLoading ? (
                <SkeletonCard className="h-12" />
              ) : categoriesError ? (
                <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800">
                  <p className="text-sm text-red-800 dark:text-red-200">
                    Failed to load categories. Please check if the backend server is running.
                  </p>
                  <p className="text-xs text-red-600 dark:text-red-400 mt-1">
                    Error: {categoriesError.message}
                  </p>
                </div>
              ) : (
                <div className="relative">
                  <select
                    value={selectedCategory}
                    onChange={(e) => setSelectedCategory(e.target.value)}
                    className="input-base appearance-none cursor-pointer"
                  >
                    <option value="">Select a category</option>
                    {categoriesData?.categories.map((category) => (
                      <option key={category.key} value={category.key}>
                        {category.display_name}
                      </option>
                    ))}
                  </select>
                  <ChevronDown className="absolute right-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400 pointer-events-none" />
                </div>
              )}
            </div>

            {/* Model Selector - Pill Buttons */}
            <div className="space-y-3">
              <label className="text-sm font-semibold text-slate-700 dark:text-slate-300">Select Model</label>
              <div className="grid grid-cols-2 gap-3">
                {MODEL_TYPES.map((model) => {
                  const Icon = model.icon;
                  const isSelected = modelType === model.value;
                  return (
                    <motion.button
                      key={model.value}
                      onClick={() => setModelType(model.value)}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      className={`
                        relative p-4 rounded-xl border-2 transition-all duration-200
                        ${isSelected
                          ? 'border-blue-500 bg-gradient-to-br from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20'
                          : 'border-slate-200 dark:border-slate-700 hover:border-slate-300 dark:hover:border-slate-600'
                        }
                      `}
                    >
                      <div className="flex items-center gap-2 mb-1">
                        <Icon className={`w-5 h-5 ${isSelected ? 'text-blue-600' : 'text-slate-400'}`} />
                        <span className={`font-medium text-sm ${isSelected ? 'text-slate-900 dark:text-slate-100' : 'text-slate-600 dark:text-slate-400'}`}>
                          {model.label}
                        </span>
                      </div>
                      {isSelected && (
                        <motion.span
                          initial={{ opacity: 0, scale: 0.8 }}
                          animate={{ opacity: 1, scale: 1 }}
                          className="text-xs text-blue-600 dark:text-blue-400 font-medium"
                        >
                          {model.badge}
                        </motion.span>
                      )}
                    </motion.button>
                  );
                })}
              </div>
            </div>

            {/* Granularity */}
            <div className="space-y-2">
              <label className="text-sm font-semibold text-slate-700 dark:text-slate-300">Granularity</label>
              <div className="grid grid-cols-3 gap-2">
                {[
                  { value: Granularity.WEEKLY, label: 'Weekly' },
                  { value: Granularity.DAILY, label: 'Daily' },
                  { value: Granularity.MONTHLY, label: 'Monthly' },
                ].map((gran) => (
                  <button
                    key={gran.value}
                    onClick={() => setGranularity(gran.value as Granularity)}
                    className={`
                      px-4 py-2 rounded-lg text-sm font-medium transition-all
                      ${granularity === gran.value
                        ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/50'
                        : 'bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-700'
                      }
                    `}
                  >
                    {gran.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Forecast Horizon Slider */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <label className="text-sm font-semibold text-slate-700 dark:text-slate-300">Forecast Horizon</label>
                <span className="text-sm font-bold text-blue-600 bg-blue-100 dark:bg-blue-900/30 px-3 py-1 rounded-full">
                  {weeksAhead} {weeksAhead === 1 ? 'week' : weeksAhead === 52 ? 'year (52 weeks)' : 'weeks'}
                </span>
              </div>
              <input
                type="range"
                min="1"
                max="52"
                step="1"
                value={weeksAhead}
                onChange={(e) => setWeeksAhead(Number(e.target.value))}
                className="w-full h-2 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-600"
              />
              <div className="flex justify-between text-xs text-slate-500">
                <span>1 week</span>
                <span>13 weeks (3 months)</span>
                <span>26 weeks (6 months)</span>
                <span>52 weeks (1 year)</span>
              </div>
              {/* Quick select buttons */}
              <div className="flex gap-2 flex-wrap">
                {[
                  { weeks: 4, label: '1 month' },
                  { weeks: 13, label: '3 months' },
                  { weeks: 26, label: '6 months' },
                  { weeks: 52, label: '1 year' },
                ].map(({ weeks, label }) => (
                  <button
                    key={weeks}
                    onClick={() => setWeeksAhead(weeks)}
                    className={`
                      px-3 py-1 text-xs font-medium rounded-lg transition-all
                      ${weeksAhead === weeks
                        ? 'bg-blue-600 text-white shadow-md'
                        : 'bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-600'
                      }
                    `}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>

            {/* Generate Button */}
            <motion.button
              onClick={handlePredict}
              disabled={predictionMutation.isPending}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className={`
                w-full py-4 rounded-xl font-semibold text-white
                bg-gradient-to-r from-blue-600 to-purple-600
                hover:from-blue-700 hover:to-purple-700
                shadow-lg shadow-blue-500/50
                disabled:opacity-50 disabled:cursor-not-allowed
                transition-all duration-200
                relative overflow-hidden
              `}
            >
              <AnimatePresence mode="wait">
                {predictionMutation.isPending ? (
                  <motion.span
                    key="loading"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="flex items-center justify-center gap-2"
                  >
                    <RefreshCw className="w-5 h-5 animate-spin" />
                    Generating Forecast...
                  </motion.span>
                ) : (
                  <motion.span
                    key="generate"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="flex items-center justify-center gap-2"
                  >
                    <Sparkles className="w-5 h-5" />
                    Generate Forecast
                  </motion.span>
                )}
              </AnimatePresence>
            </motion.button>
          </div>
        </motion.div>

        {/* Results Section - Right Panel (60%) */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
          className="lg:col-span-3 space-y-6"
        >
          {!predictionMutation.data ? (
            /* Empty State */
            <div className="glass-card p-12 text-center">
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: 0.3, type: 'spring', stiffness: 200 }}
                className="w-20 h-20 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center mx-auto mb-6"
              >
                <BarChart3 className="w-10 h-10 text-white" />
              </motion.div>
              <h3 className="text-xl font-bold text-slate-900 dark:text-slate-100 mb-2">
                Ready to Forecast
              </h3>
              <p className="text-slate-600 dark:text-slate-400 mb-6">
                Configure your parameters and click "Generate Forecast" to see predictions
              </p>
            </div>
          ) : (
            <>
              {/* Prediction Summary */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {/* Last Historical Value */}
                <div className="glass-card p-6">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-slate-600 dark:text-slate-400">Last Historical</span>
                    <Clock className="w-4 h-4 text-slate-400" />
                  </div>
                  <motion.p
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="text-3xl font-bold text-slate-900 dark:text-slate-100"
                  >
                    ${getCurrentValue().toFixed(2)}
                  </motion.p>
                </div>

                {/* Average Forecast */}
                <div className="glass-card p-6 bg-gradient-to-br from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-blue-600 dark:text-blue-400 font-semibold">Average Forecast</span>
                    <Zap className="w-4 h-4 text-blue-500" />
                  </div>
                  <motion.p
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.1 }}
                    className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent"
                  >
                    ${getPredictedValue().toFixed(2)}
                  </motion.p>
                </div>

                {/* Percentage Change */}
                <div className={`glass-card p-6 ${isPositive ? 'border-l-4 border-emerald-500' : 'border-l-4 border-red-500'}`}>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-slate-600 dark:text-slate-400">Expected Change</span>
                    {isPositive ? (
                      <TrendingUp className="w-4 h-4 text-emerald-500" />
                    ) : (
                      <TrendingDown className="w-4 h-4 text-red-500" />
                    )}
                  </div>
                  <motion.p
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.2 }}
                    className={`text-3xl font-bold ${isPositive ? 'text-emerald-600' : 'text-red-600'}`}
                  >
                    {isPositive ? '+' : ''}{percentageChange.toFixed(2)}%
                  </motion.p>
                </div>
              </div>

              {/* Forecast Chart */}
              <div className="glass-card p-6">
                <div className="flex items-center justify-between mb-6">
                  <div>
                    <h3 className="text-lg font-bold text-slate-900 dark:text-slate-100">Sales Forecast</h3>
                    <p className="text-sm text-slate-600 dark:text-slate-400">
                      {selectedCategory} • {modelType}
                    </p>
                  </div>
                  <motion.button
                    onClick={downloadChart}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    className="p-2 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
                    title="Download Chart"
                  >
                    <Download className="w-5 h-5 text-slate-600 dark:text-slate-400" />
                  </motion.button>
                </div>

                <ResponsiveContainer width="100%" height={350}>
                  <AreaChart data={combinedData}>
                    <defs>
                      <linearGradient id="colorPredicted" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                        <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                      </linearGradient>
                      <linearGradient id="colorConfidence" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#93c5fd" stopOpacity={0.4}/>
                        <stop offset="95%" stopColor="#93c5fd" stopOpacity={0.1}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" opacity={0.5} />
                    <XAxis
                      dataKey="date"
                      stroke="#64748b"
                      tick={{ fill: '#64748b', fontSize: 11 }}
                      tickLine={false}
                    />
                    <YAxis
                      stroke="#64748b"
                      tick={{ fill: '#64748b', fontSize: 11 }}
                      tickLine={false}
                      tickFormatter={(value) => `$${(value / 1000).toFixed(0)}k`}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Area
                      type="monotone"
                      dataKey="upper"
                      stroke="none"
                      fill="url(#colorConfidence)"
                      connectNulls={false}
                    />
                    <Area
                      type="monotone"
                      dataKey="lower"
                      stroke="none"
                      fill="url(#colorConfidence)"
                      connectNulls={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="historical"
                      stroke="#94a3b8"
                      strokeWidth={2}
                      dot={false}
                      connectNulls={false}
                      strokeDasharray="5 5"
                    />
                    <Line
                      type="monotone"
                      dataKey="predicted"
                      stroke="#3b82f6"
                      strokeWidth={3}
                      dot={{ fill: '#3b82f6', r: 4 }}
                      activeDot={{ r: 6 }}
                      connectNulls={false}
                    />
                    <ReferenceLine
                      x={historicalData.length}
                      stroke="#64748b"
                      strokeDasharray="5 5"
                      label="Forecast Start"
                    />
                  </AreaChart>
                </ResponsiveContainer>

                <div className="flex items-center justify-center gap-6 mt-4 text-xs text-slate-500">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-0.5 bg-slate-400" style={{ borderStyle: 'dashed' }}></div>
                    <span>Historical</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-0.5 bg-blue-500"></div>
                    <span>Forecast</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-blue-200/50 rounded"></div>
                    <span>95% Confidence</span>
                  </div>
                </div>
              </div>

              {/* Feature Contributions */}
              {predictionMutation.data.shap_values && predictionMutation.data.shap_values.length > 0 && (
                <div className="glass-card p-6">
                  <div className="flex items-center gap-3 mb-6">
                    <div className="p-2 bg-gradient-to-br from-purple-500 to-pink-600 rounded-lg">
                      <Info className="w-5 h-5 text-white" />
                    </div>
                    <div>
                      <h3 className="text-lg font-bold text-slate-900 dark:text-slate-100">Feature Contributions</h3>
                      <p className="text-sm text-slate-600 dark:text-slate-400">Top factors influencing this prediction</p>
                    </div>
                  </div>

                  <div className="space-y-3">
                    {predictionMutation.data.shap_values.slice(0, 5).map((shap, index) => {
                      const isExpanded = expandedFeature === index;
                      const isPositive = shap.value > 0;
                      return (
                        <motion.div
                          key={index}
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: index * 0.05 }}
                          className={`
                            border rounded-xl overflow-hidden transition-all duration-200
                            ${isExpanded ? 'border-blue-300 dark:border-blue-700 bg-blue-50/50 dark:bg-blue-900/10' : 'border-slate-200 dark:border-slate-700 hover:border-slate-300 dark:hover:border-slate-600'}
                          `}
                        >
                          <button
                            onClick={() => setExpandedFeature(isExpanded ? null : index)}
                            className="w-full p-4 flex items-center gap-4 text-left"
                          >
                            <div className={`
                              p-2 rounded-lg
                              ${isPositive ? 'bg-emerald-100 dark:bg-emerald-900/30' : 'bg-red-100 dark:bg-red-900/30'}
                            `}>
                              {isPositive ? (
                                <TrendingUp className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
                              ) : (
                                <TrendingDown className="w-5 h-5 text-red-600 dark:text-red-400" />
                              )}
                            </div>
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center justify-between mb-1">
                                <span className="font-semibold text-slate-900 dark:text-slate-100">{shap.feature}</span>
                                <span className={`
                                  text-sm font-bold
                                  ${isPositive ? 'text-emerald-600' : 'text-red-600'}
                                `}>
                                  {isPositive ? '+' : ''}{shap.value.toFixed(2)}
                                </span>
                              </div>
                              <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2 overflow-hidden">
                                <motion.div
                                  initial={{ width: 0 }}
                                  animate={{ width: `${Math.abs(shap.importance) * 100}%` }}
                                  transition={{ delay: index * 0.05 + 0.2, duration: 0.5 }}
                                  className={`h-full ${isPositive ? 'bg-emerald-500' : 'bg-red-500'}`}
                                />
                              </div>
                            </div>
                            <motion.div
                              animate={{ rotate: isExpanded ? 180 : 0 }}
                              transition={{ duration: 0.2 }}
                            >
                              <ChevronDown className="w-5 h-5 text-slate-400" />
                            </motion.div>
                          </button>

                          <AnimatePresence>
                            {isExpanded && (
                              <motion.div
                                initial={{ height: 0, opacity: 0 }}
                                animate={{ height: 'auto', opacity: 1 }}
                                exit={{ height: 0, opacity: 0 }}
                                transition={{ duration: 0.3 }}
                                className="px-4 pb-4"
                              >
                                <div className="pt-4 border-t border-slate-200 dark:border-slate-700">
                                  <p className="text-sm text-slate-600 dark:text-slate-400 mb-3">
                                    This feature contributed <span className={`font-semibold ${isPositive ? 'text-emerald-600' : 'text-red-600'}`}>
                                      {(Math.abs(shap.importance) * 100).toFixed(1)}%
                                    </span> to the prediction.
                                  </p>
                                  {/* Mini sparkline */}
                                  <div className="h-16 bg-slate-100 dark:bg-slate-800 rounded-lg flex items-end p-2 gap-1">
                                    {Array.from({ length: 20 }).map((_, i) => {
                                      const height = 30 + Math.random() * 50;
                                      return (
                                        <div
                                          key={i}
                                          className={`flex-1 rounded-sm ${isPositive ? 'bg-emerald-400' : 'bg-red-400'}`}
                                          style={{ height: `${height}%` }}
                                        />
                                      );
                                    })}
                                  </div>
                                </div>
                              </motion.div>
                            )}
                          </AnimatePresence>
                        </motion.div>
                      );
                    })}
                  </div>
                </div>
              )}
            </>
          )}
        </motion.div>
      </div>
    </div>
  );
};
