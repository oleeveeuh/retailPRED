/**
 * Anomaly Detection Page
 * Displays unusual predictions with economic context explanations
 */

import { FC, useState, useEffect, useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Filter,
  Calendar,
  BarChart3,
} from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import { predictionsApi } from '../api/unifiedApi';

// Helper function to load economic context for demo mode
const loadEconomicContext = async (date: string) => {
  try {
    const response = await fetch('/demo-data/economic-context.json');
    const data = await response.json();

    // Find the closest date match (within same month)
    const targetDate = new Date(date);
    const closestMatch = data.data.reduce((best: any, current: any) => {
      const currentDate = new Date(current.date);
      const bestDate = new Date(best.date);

      const currentDiff = Math.abs(targetDate.getTime() - currentDate.getTime());
      const bestDiff = Math.abs(targetDate.getTime() - bestDate.getTime());

      return currentDiff < bestDiff ? current : best;
    }, data.data[0]);

    return {
      regime: closestMatch.regime,
      indicators: closestMatch.indicators || {
        unemployment: closestMatch.unemployment,
        unemploymentChange: closestMatch.unemployment_change,
        consumerConfidence: closestMatch.consumer_confidence,
        confidenceChange: closestMatch.confidence_change,
        fedRate: closestMatch.fed_rate,
      },
      anomalies: closestMatch.anomalies,
      explanation: closestMatch.explanation,
    };
  } catch (error) {
    console.error('Failed to load economic context:', error);
    return null;
  }
};

const AnomalyDetectionPage: FC = () => {
  const [selectedCategory, setSelectedCategory] = useState('total_sales');
  const [severityFilter, setSeverityFilter] = useState<'all' | 'moderate' | 'severe'>('all');
  const [typeFilter, setTypeFilter] = useState<'all' | 'surge' | 'decline'>('all');
  const [anomaliesWithContext, setAnomaliesWithContext] = useState<any[]>([]);

  // Fetch predictions for anomaly detection
  // Map category names to model name patterns
  const categoryToPattern: Record<string, string> = {
    'total_sales': 'total_retail_sales',
    'automobile_dealers': 'automobile_dealers',
    'building_materials': 'building_materials_garden',
    'clothing_accessories': 'clothing_accessories',
    'electronics_and_appliances': 'electronics_and_appliances',
    'food_beverage_stores': 'food_beverage_stores',
    'furniture_home_furnishings': 'furniture_home_furnishings',
    'gasoline_stations': 'gasoline_stations',
    'general_merchandise': 'general_merchandise',
    'health_personal_care': 'health_personal_care',
    'sporting_goods_hobby': 'sporting_goods_hobby',
  };

  const { data: predictionsResponse, isLoading } = useQuery({
    queryKey: ['predictions', selectedCategory],
    queryFn: () => predictionsApi.getHistory({
      model_name: categoryToPattern[selectedCategory] || selectedCategory,
      limit: 500  // Get more predictions to find anomalies
    }),
  });

  const predictions = predictionsResponse?.predictions || [];

  // Debug logging
  console.log('Anomaly Detection Debug:');
  console.log('Selected Category:', selectedCategory);
  console.log('Model Pattern:', categoryToPattern[selectedCategory]);
  console.log('Predictions fetched:', predictions.length);
  console.log('First prediction:', predictions[0]);

  // Detect anomalies from actual prediction data (use Memo to prevent infinite loop)
  const anomalies = useMemo(() => {
    if (!predictions || predictions.length === 0) return [];

    const hasPredictedValue = predictions[0]?.predicted_value !== undefined;

    // Detect anomalies: week-over-week changes > 5%
    return predictions.filter((p, i) => {
      if (i === 0) return false;
      const prev = predictions[i - 1];
      const currentValue = hasPredictedValue ? p.predicted_value! : (p as any).value;
      const previousValue = hasPredictedValue ? prev.predicted_value! : (prev as any).value;

      if (!currentValue || !previousValue) return false;

      const change = Math.abs(((currentValue - previousValue) / previousValue) * 100);
      return change > 5; // 5% threshold for weekly retail sales anomalies
    }).map((p, i, arr) => {
      const predIndex = predictions.indexOf(p);
      const prev = predictions[predIndex - 1];
      const currentValue = hasPredictedValue ? p.predicted_value! : (p as any).value;
      const previousValue = hasPredictedValue ? prev.predicted_value! : (prev as any).value;

      const change = ((currentValue - previousValue) / previousValue) * 100;
      const changeMagnitude = Math.abs(change);
      const severity = changeMagnitude > 10 ? 'severe' : 'moderate'; // 10%+ is severe
      const type = change > 0 ? 'surge' : 'decline';

      return {
        id: `${p.prediction_date}-${selectedCategory}`,
        date: p.prediction_date,
        predicted_value: currentValue,
        actual_value: p.actual_value,
        change_percent: change,
        model_name: p.model_name || 'LGBM',
        category: selectedCategory,
        severity,
        type,
      };
    });
  }, [predictions, selectedCategory]);

  // Fetch economic context for each anomaly
  useEffect(() => {
    const fetchEconomicContext = async () => {
      if (!anomalies || anomalies.length === 0) {
        setAnomaliesWithContext([]);
        return;
      }

      const anomaliesWithCtx = await Promise.all(
        anomalies.map(async (anomaly) => {
          const context = await loadEconomicContext(anomaly.date);
          return {
            ...anomaly,
            economicContext: context,
          };
        })
      );

      setAnomaliesWithContext(anomaliesWithCtx);
    };

    fetchEconomicContext();
  }, [anomalies]);

  // Apply filters
  const filteredAnomalies = anomaliesWithContext.filter((a: any) => {
    if (severityFilter !== 'all' && a.severity !== severityFilter) return false;
    if (typeFilter !== 'all' && a.type !== typeFilter) return false;
    return true;
  });

  const stats = {
    total: anomaliesWithContext.length,
    surges: anomaliesWithContext.filter((a: any) => a.type === 'surge').length,
    declines: anomaliesWithContext.filter((a: any) => a.type === 'decline').length,
    severe: anomaliesWithContext.filter((a: any) => a.severity === 'severe').length,
  };

  const categories = [
    { value: 'total_sales', label: 'Total Retail Sales' },
    { value: 'automobile_dealers', label: 'Automobile Dealers' },
    { value: 'building_materials', label: 'Building Materials & Garden' },
    { value: 'clothing_accessories', label: 'Clothing & Accessories' },
    { value: 'electronics_and_appliances', label: 'Electronics & Appliances' },
    { value: 'food_beverage_stores', label: 'Food & Beverage Stores' },
    { value: 'furniture_home_furnishings', label: 'Furniture & Home Furnishings' },
    { value: 'gasoline_stations', label: 'Gasoline Stations' },
    { value: 'general_merchandise', label: 'General Merchandise Stores' },
    { value: 'health_personal_care', label: 'Health & Personal Care' },
    { value: 'sporting_goods_hobby', label: 'Sporting Goods & Hobby' },
  ];

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-normal text-gray-900 tracking-tight">
            Anomaly Detection
          </h1>
          <p className="text-gray-500 text-sm font-light mt-1">
            Unusual predictions explained with economic context
          </p>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white border border-gray-200 rounded-sm p-5"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs font-medium text-gray-600 uppercase tracking-wide">
                Total Anomalies
              </p>
              <p className="text-3xl font-normal text-gray-900 mt-2">
                {stats.total}
              </p>
            </div>
            <div className="p-3 bg-[#3A3A6C] rounded-sm">
              <AlertTriangle className="w-5 h-5 text-white" />
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-white border border-gray-200 rounded-sm p-5"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs font-medium text-gray-600 uppercase tracking-wide">
                Sales Surges
              </p>
              <p className="text-3xl font-normal text-green-600 mt-2">
                {stats.surges}
              </p>
            </div>
            <div className="p-3 bg-green-100 rounded-sm">
              <TrendingUp className="w-5 h-5 text-green-600" />
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-white border border-gray-200 rounded-sm p-5"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs font-medium text-gray-600 uppercase tracking-wide">
                Sales Declines
              </p>
              <p className="text-3xl font-normal text-red-600 mt-2">
                {stats.declines}
              </p>
            </div>
            <div className="p-3 bg-red-100 rounded-sm">
              <TrendingDown className="w-5 h-5 text-red-600" />
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-white border border-gray-200 rounded-sm p-5"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs font-medium text-gray-600 uppercase tracking-wide">
                Severe Events
              </p>
              <p className="text-3xl font-normal text-orange-600 mt-2">
                {stats.severe}
              </p>
            </div>
            <div className="p-3 bg-orange-100 rounded-sm">
              <BarChart3 className="w-5 h-5 text-orange-600" />
            </div>
          </div>
        </motion.div>
      </div>

      {/* Filters */}
      <div className="bg-white border border-gray-200 rounded-sm p-4">
        <div className="flex flex-wrap gap-4 items-center">
          <div className="flex items-center gap-2">
            <Filter className="w-4 h-4 text-gray-500" />
            <span className="text-sm font-medium text-gray-700">
              Filters:
            </span>
          </div>

          {/* Category Filter */}
          <select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="px-3 py-2 border border-gray-200 rounded-sm bg-white text-gray-900 text-sm"
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
            className="px-3 py-2 border border-gray-200 rounded-sm bg-white text-gray-900 text-sm"
          >
            <option value="all">All Severities</option>
            <option value="moderate">Moderate (5-10%)</option>
            <option value="severe">Severe (&gt;10%)</option>
          </select>

          {/* Type Filter */}
          <select
            value={typeFilter}
            onChange={(e) => setTypeFilter(e.target.value as any)}
            className="px-3 py-2 border border-gray-200 rounded-sm bg-white text-gray-900 text-sm"
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
          <div className="text-center py-12 bg-white border border-gray-200 rounded-sm">
            <AlertTriangle className="w-16 h-16 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-normal text-gray-900 mb-2">
              No Anomalies Found
            </h3>
            <p className="text-gray-500">
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
              className={`bg-white border border-gray-200 rounded-sm p-6 border-l-2 ${
                anomaly.type === 'surge'
                  ? 'border-green-500'
                  : 'border-red-500'
              } shadow-sm hover:shadow-md transition-shadow`}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-3">
                    <span className={`px-3 py-1 rounded-sm text-xs font-normal ${
                      anomaly.severity === 'severe'
                        ? 'bg-orange-100 text-orange-800'
                        : 'bg-blue-100 text-blue-800'
                    }`}>
                      {anomaly.severity === 'severe' ? 'Severe' : 'Moderate'}
                    </span>
                    <span className={`px-3 py-1 rounded-sm text-xs font-normal ${
                      anomaly.type === 'surge'
                        ? 'bg-green-100 text-green-800'
                        : 'bg-red-100 text-red-800'
                    }`}>
                      {anomaly.type === 'surge' ? 'Sales Surge' : 'Sales Decline'}
                    </span>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-3">
                    <div>
                      <p className="text-xs text-gray-500 mb-1">Date</p>
                      <p className="text-sm font-normal text-gray-900">
                        {anomaly.date}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-500 mb-1">Model</p>
                      <p className="text-sm font-normal text-gray-900">
                        {anomaly.model_name}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-500 mb-1">Predicted Value</p>
                      <p className="text-sm font-normal text-gray-900">
                        ${anomaly.predicted_value?.toLocaleString('en-US', { maximumFractionDigits: 0 })}
                      </p>
                    </div>
                  </div>

                  <div className={`flex items-center gap-2 text-lg font-normal ${
                    anomaly.type === 'surge'
                      ? 'text-green-600'
                      : 'text-red-600'
                  }`}>
                    {anomaly.type === 'surge' ? (
                      <TrendingUp className="w-5 h-5" />
                    ) : (
                      <TrendingDown className="w-5 h-5" />
                    )}
                    <span>
                      {anomaly.change_percent != null && !isNaN(anomaly.change_percent)
                        ? `${anomaly.change_percent > 0 ? '+' : ''}${anomaly.change_percent.toFixed(1)}%`
                        : 'N/A'}
                    </span>
                    <span className="text-sm font-normal text-gray-500">
                      from previous period
                    </span>
                  </div>

                  {/* Economic Context */}
                  {anomaly.economicContext && (
                    <div className="mt-4 pt-4 border-t border-gray-200">
                      <div className="flex items-center gap-2 mb-2">
                        <span className={`px-2 py-0.5 rounded text-xs font-normal ${
                          anomaly.economicContext.regime === 'crisis'
                            ? 'bg-red-100 text-red-800'
                            : anomaly.economicContext.regime === 'recession'
                            ? 'bg-orange-100 text-orange-800'
                            : anomaly.economicContext.regime === 'expansion'
                            ? 'bg-green-100 text-green-800'
                            : 'bg-blue-100 text-blue-800'
                        }`}>
                          {anomaly.economicContext.regime.charAt(0).toUpperCase() + anomaly.economicContext.regime.slice(1)}
                        </span>
                        <span className="text-xs font-normal text-gray-600">
                          Economic Context
                        </span>
                      </div>

                      <div className="grid grid-cols-3 gap-2 mb-2 text-xs">
                        <div>
                          <span className="text-gray-500">Unemployment: </span>
                          <span className="font-normal text-gray-900">
                            {anomaly.economicContext.indicators?.unemployment != null ? `${anomaly.economicContext.indicators.unemployment}%` : 'N/A'}
                            {anomaly.economicContext.indicators?.unemploymentChange != null && anomaly.economicContext.indicators.unemploymentChange !== 0 && (
                              <span className={`ml-1 ${anomaly.economicContext.indicators.unemploymentChange > 0 ? 'text-red-600' : 'text-green-600'}`}>
                                ({anomaly.economicContext.indicators.unemploymentChange > 0 ? '+' : ''}{anomaly.economicContext.indicators.unemploymentChange.toFixed(1)}%)
                              </span>
                            )}
                          </span>
                        </div>
                        <div>
                          <span className="text-gray-500">Confidence: </span>
                          <span className="font-normal text-gray-900">
                            {anomaly.economicContext.indicators?.consumerConfidence != null ? anomaly.economicContext.indicators.consumerConfidence : 'N/A'}
                            {anomaly.economicContext.indicators?.confidenceChange != null && anomaly.economicContext.indicators.confidenceChange !== 0 && (
                              <span className={`ml-1 ${anomaly.economicContext.indicators.confidenceChange < 0 ? 'text-red-600' : 'text-green-600'}`}>
                                ({anomaly.economicContext.indicators.confidenceChange > 0 ? '+' : ''}{anomaly.economicContext.indicators.confidenceChange.toFixed(1)})
                              </span>
                            )}
                          </span>
                        </div>
                        <div>
                          <span className="text-gray-500">Fed Rate: </span>
                          <span className="font-normal text-gray-900">
                            {anomaly.economicContext.indicators?.fedRate != null ? `${anomaly.economicContext.indicators.fedRate}%` : 'N/A'}
                          </span>
                        </div>
                      </div>

                      <p className="text-xs text-gray-600 leading-relaxed">
                        {anomaly.economicContext.explanation}
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          ))
        )}
      </div>

      {/* Info Box */}
      <div className="border border-blue-200 rounded-sm p-4" style={{ backgroundColor: '#eff6ff' }}>
        <div className="flex gap-3">
          <AlertTriangle className="w-5 h-5 text-[#3A3A6C] flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <h4 className="font-normal mb-1" style={{ color: '#1e3a8a' }}>
              About Anomaly Detection
            </h4>
            <p className="text-sm" style={{ color: '#1e40af' }}>
              Anomalies are detected when predictions change by more than 5% from the previous period.
              Severity is classified as moderate (5-10%) or severe (&gt;10%). This helps identify unusual
              patterns that may require further investigation.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnomalyDetectionPage;
