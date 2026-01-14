/**
 * Business Dashboard - Executive View with Tableau Integration
 * Stakeholder-friendly dashboard with KPIs and Tableau visualizations
 */

import type { FC } from 'react';
import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  BarChart3,
  Download,
  ExternalLink,
  Info,
  TrendingUp,
  DollarSign,
  Crosshair,
  Calendar,
  Loader2,
  AlertCircle,
  FileText,
  Users,
  ArrowRight,
} from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import { predictionsApi, trainingMetricsApi } from '../api/unifiedApi';
import { TableauEmbed } from '../components/TableauEmbed';

type TabType = 'tableau' | 'export' | 'guide';

interface KPICardProps {
  title: string;
  value: string | number;
  change?: string;
  icon: React.ReactNode;
  color: string;
  trend?: 'up' | 'down' | 'neutral';
}

const KPICard: FC<KPICardProps> = ({ title, value, change, icon, color, trend }) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    className={`bg-white rounded-lg shadow-lg p-6 border-l-4 ${color}`}
  >
    <div className="flex items-start justify-between">
      <div className="flex-1">
        <p className="text-sm font-medium text-gray-600 mb-1">{title}</p>
        <p className="text-3xl font-bold text-gray-900">{value}</p>
        {change && (
          <p className={`text-sm mt-2 flex items-center ${
            trend === 'up' ? 'text-green-600' : trend === 'down' ? 'text-red-600' : 'text-gray-600'
          }`}>
            {trend === 'up' && <TrendingUp className="w-4 h-4 mr-1" />}
            {change}
          </p>
        )}
      </div>
      <div className="ml-4 p-3 bg-gray-50 rounded-lg">
        {icon}
      </div>
    </div>
  </motion.div>
);

export const BusinessDashboard: FC = () => {
  const [activeTab, setActiveTab] = useState<TabType>('tableau');
  const [exportLoading, setExportLoading] = useState(false);

  // Fetch summary stats (fetch all predictions to get accurate total count)
  const { data: statsData, isLoading: statsLoading } = useQuery({
    queryKey: ['business-stats'],
    queryFn: async () => {
      // Fetch with high limit to get total_count accurately
      const data = await predictionsApi.getHistory({ limit: 15000 });
      return data;
    },
  });

  // Fetch model count
  const { data: modelsData } = useQuery({
    queryKey: ['models-list'],
    queryFn: async () => {
      return await trainingMetricsApi.getModels();
    },
  });

  // Calculate KPIs
  const calculateKPIs = () => {
    if (!statsData) return null;

    const predictions = statsData.predictions || [];
    const validated = predictions.filter(p => p.is_validated && p.actual_value);
    const totalPredictions = statsData.total_count || 0;

    // Get active models count
    const activeModels = modelsData?.total_count || 0;

    if (validated.length === 0) {
      return {
        totalPredictions,
        avgAccuracy: 'N/A',
        totalSales: '$0',
        activeModels,
        forecastRange: 'N/A',
      };
    }

    const avgError = validated.reduce((sum, p) => sum + (p.error_percentage || 0), 0) / validated.length;
    const avgAccuracy = (100 - avgError).toFixed(1);

    const totalSales = validated
      .reduce((sum, p) => sum + (p.actual_value || 0), 0)
      .toFixed(0);

    const dates = predictions.map(p => p.prediction_date).sort();
    const forecastRange = dates.length > 0
      ? `${dates[0]} - ${dates[dates.length - 1]}`
      : 'N/A';

    return {
      totalPredictions,
      avgAccuracy: `${avgAccuracy}%`,
      totalSales: `$${Number(totalSales).toLocaleString()}`,
      activeModels,
      forecastRange,
    };
  };

  const kpis = calculateKPIs();

  const handleExportCSV = async () => {
    setExportLoading(true);
    try {
      const apiBaseUrl = import.meta.env.VITE_API_URL !== undefined
        ? import.meta.env.VITE_API_URL
        : 'http://localhost:8000';
      const response = await fetch(`${apiBaseUrl}/api/export/predictions-csv`);
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `retail_predictions_${new Date().toISOString().split('T')[0]}.csv`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      }
    } catch (error) {
      console.error('Export failed:', error);
      alert('Export failed. Please try again.');
    } finally {
      setExportLoading(false);
    }
  };

  const tableauEmbedUrl = import.meta.env.VITE_TABLEAU_EMBED_URL || '';

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 p-8">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-4xl font-bold text-gray-900 flex items-center">
              <BarChart3 className="w-10 h-10 mr-3 text-blue-600" />
              Business Dashboard
            </h1>
            <p className="text-gray-600 mt-2">
              Executive view of retail forecasting performance and insights
            </p>
          </div>
          <a
            href="/"
            className="px-4 py-2 bg-white text-gray-700 rounded-lg shadow hover:shadow-md transition-shadow text-sm font-medium"
          >
            Switch to Technical View
          </a>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6 mb-8">
        {statsLoading ? (
          <>
            {[1, 2, 3, 4, 5].map((i) => (
              <div key={i} className="bg-white rounded-lg shadow p-6 border-l-4 border-blue-500 animate-pulse">
                <div className="h-20 bg-gray-200 rounded"></div>
              </div>
            ))}
          </>
        ) : kpis ? (
          <>
            <KPICard
              title="Total Predictions"
              value={kpis.totalPredictions.toLocaleString()}
              icon={<Calendar className="w-6 h-6 text-blue-600" />}
              color="border-blue-500"
            />
            <KPICard
              title="Forecast Accuracy"
              value={kpis.avgAccuracy}
              icon={<Crosshair className="w-6 h-6 text-green-600" />}
              color="border-green-500"
            />
            <KPICard
              title="Total Sales Forecast"
              value={kpis.totalSales}
              icon={<DollarSign className="w-6 h-6 text-purple-600" />}
              color="border-purple-500"
            />
            <KPICard
              title="Forecast Period"
              value={kpis.forecastRange !== 'N/A' ? kpis.forecastRange.split(' - ')[1] : 'N/A'}
              icon={<TrendingUp className="w-6 h-6 text-orange-600" />}
              color="border-orange-500"
            />
            <KPICard
              title="Active Models"
              value={kpis.activeModels}
              icon={<Users className="w-6 h-6 text-indigo-600" />}
              color="border-indigo-500"
            />
          </>
        ) : null}
      </div>

      {/* Tab Navigation */}
      <div className="bg-white rounded-lg shadow-lg mb-6">
        <div className="border-b border-gray-200">
          <nav className="flex -mb-px">
            <button
              onClick={() => setActiveTab('tableau')}
              className={`py-4 px-6 border-b-2 font-medium text-sm transition-colors ${
                activeTab === 'tableau'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              <FileText className="w-4 h-4 mr-2 inline" />
              Executive Summary
            </button>
            <button
              onClick={() => setActiveTab('export')}
              className={`py-4 px-6 border-b-2 font-medium text-sm transition-colors ${
                activeTab === 'export'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              <Download className="w-4 h-4 mr-2 inline" />
              Export Data
            </button>
            <button
              onClick={() => setActiveTab('guide')}
              className={`py-4 px-6 border-b-2 font-medium text-sm transition-colors ${
                activeTab === 'guide'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              <Info className="w-4 h-4 mr-2 inline" />
              Dashboard Guide
            </button>
          </nav>
        </div>

        {/* Tab Content */}
        <div className="p-6">
          <AnimatePresence mode="wait">
            {activeTab === 'tableau' && (
              <motion.div
                key="tableau"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
              >
                {tableauEmbedUrl ? (
                  <div className="space-y-4">
                    <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                      <div className="flex items-start">
                        <Info className="w-5 h-5 text-blue-600 mr-3 mt-0.5" />
                        <div>
                          <p className="text-sm text-blue-800 font-medium">
                            Interactive Dashboard
                          </p>
                          <p className="text-sm text-blue-700 mt-1">
                            View visualizations, trends, and insights powered by Tableau.
                          </p>
                        </div>
                      </div>
                    </div>

                    <div className="relative" style={{ height: '600px' }}>
                      <TableauEmbed url={tableauEmbedUrl} height={600} />
                    </div>

                    <div className="flex justify-end">
                      <a
                        href={tableauEmbedUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium"
                      >
                        Open in Tableau Public
                        <ExternalLink className="w-4 h-4 ml-2" />
                      </a>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-16">
                    <AlertCircle className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                    <h3 className="text-xl font-semibold text-gray-900 mb-2">
                      Tableau Dashboard Not Configured
                    </h3>
                    <p className="text-gray-600 mb-6 max-w-md mx-auto">
                      To display the Tableau dashboard, you need to:
                    </p>
                    <ol className="text-left text-sm text-gray-600 max-w-lg mx-auto space-y-2 mb-8">
                      <li>1. Export data using the "Export Data" tab</li>
                      <li>2. Create a dashboard in Tableau Desktop or Tableau Public</li>
                      <li>3. Publish to Tableau Public and get the embed URL</li>
                      <li>4. Add <code className="bg-gray-100 px-2 py-1 rounded">VITE_TABLEAU_EMBED_URL</code> to your .env file</li>
                    </ol>
                    <a
                      href="https://public.tableau.com/"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium"
                    >
                      Go to Tableau Public
                      <ExternalLink className="w-4 h-4 ml-2" />
                    </a>
                  </div>
                )}
              </motion.div>
            )}

            {activeTab === 'export' && (
              <motion.div
                key="export"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
              >
                <div className="space-y-6">
                  <div>
                    <h3 className="text-xl font-semibold text-gray-900 mb-4">
                      Export Prediction Data
                    </h3>
                    <p className="text-gray-600 mb-6">
                      Download all predictions in CSV format optimized for Tableau and other BI tools.
                    </p>
                  </div>

                  <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
                    <h4 className="font-semibold text-blue-900 mb-4">Included Fields</h4>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                      <div className="flex items-start">
                        <div className="w-2 h-2 bg-blue-600 rounded-full mt-1.5 mr-2"></div>
                        <span className="text-blue-800">Prediction Date</span>
                      </div>
                      <div className="flex items-start">
                        <div className="w-2 h-2 bg-blue-600 rounded-full mt-1.5 mr-2"></div>
                        <span className="text-blue-800">Store/Product</span>
                      </div>
                      <div className="flex items-start">
                        <div className="w-2 h-2 bg-blue-600 rounded-full mt-1.5 mr-2"></div>
                        <span className="text-blue-800">Predicted Sales</span>
                      </div>
                      <div className="flex items-start">
                        <div className="w-2 h-2 bg-blue-600 rounded-full mt-1.5 mr-2"></div>
                        <span className="text-blue-800">Actual Sales</span>
                      </div>
                      <div className="flex items-start">
                        <div className="w-2 h-2 bg-blue-600 rounded-full mt-1.5 mr-2"></div>
                        <span className="text-blue-800">Error %</span>
                      </div>
                      <div className="flex items-start">
                        <div className="w-2 h-2 bg-blue-600 rounded-full mt-1.5 mr-2"></div>
                        <span className="text-blue-800">Model Name</span>
                      </div>
                    </div>
                  </div>

                  <div className="bg-gray-50 border border-gray-200 rounded-lg p-6">
                    <h4 className="font-semibold text-gray-900 mb-4">Export Options</h4>
                    <div className="space-y-3">
                      <button
                        onClick={handleExportCSV}
                        disabled={exportLoading}
                        className="w-full flex items-center justify-center px-6 py-4 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        {exportLoading ? (
                          <>
                            <Loader2 className="w-5 h-5 mr-3 animate-spin" />
                            Exporting...
                          </>
                        ) : (
                          <>
                            <Download className="w-5 h-5 mr-3" />
                            Download CSV
                          </>
                        )}
                      </button>
                      <p className="text-sm text-gray-500 text-center">
                        CSV file will include all predictions with formatting optimized for Tableau
                      </p>
                    </div>
                  </div>

                  <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                    <div className="flex items-start">
                      <Info className="w-5 h-5 text-yellow-600 mr-3 mt-0.5 flex-shrink-0" />
                      <div className="text-sm text-yellow-800">
                        <p className="font-medium mb-1">Tip for Tableau</p>
                        <p>
                          After downloading, open the CSV in Tableau Desktop or upload to Tableau Public.
                          Use "Prediction Date" as a date dimension and create visualizations like:
                          line charts for trends, bar charts for comparisons, and KPI cards for metrics.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}

            {activeTab === 'guide' && (
              <motion.div
                key="guide"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                className="prose max-w-4xl"
              >
                <div className="space-y-8">
                  <div>
                    <h3 className="text-2xl font-bold text-gray-900 mb-4">
                      Understanding the Dashboard Metrics
                    </h3>
                    <div className="space-y-4">
                      <div className="bg-white border border-gray-200 rounded-lg p-6">
                        <h4 className="font-semibold text-gray-900 mb-2 flex items-center">
                          <Crosshair className="w-5 h-5 mr-2 text-green-600" />
                          Forecast Accuracy
                        </h4>
                        <p className="text-gray-700 mb-2">
                          Percentage of how close our predictions are to actual sales values.
                        </p>
                        <div className="bg-green-50 border-l-4 border-green-500 p-4">
                          <p className="text-sm">
                            <strong>Good:</strong> 95%+ accuracy (error less than 5%)<br/>
                            <strong>Fair:</strong> 90-95% accuracy<br/>
                            <strong>Needs Attention:</strong> Below 90% accuracy
                          </p>
                        </div>
                      </div>

                      <div className="bg-white border border-gray-200 rounded-lg p-6">
                        <h4 className="font-semibold text-gray-900 mb-2 flex items-center">
                          <Calendar className="w-5 h-5 mr-2 text-blue-600" />
                          Total Predictions
                        </h4>
                        <p className="text-gray-700">
                          Number of forecasts made across all retail categories and time periods.
                          More predictions = better trend analysis.
                        </p>
                      </div>

                      <div className="bg-white border border-gray-200 rounded-lg p-6">
                        <h4 className="font-semibold text-gray-900 mb-2 flex items-center">
                          <TrendingUp className="w-5 h-5 mr-2 text-purple-600" />
                          Total Sales Forecast
                        </h4>
                        <p className="text-gray-700">
                          Sum of all predicted sales values for the forecast period.
                          Helps with revenue planning and budget allocation.
                        </p>
                      </div>
                    </div>
                  </div>

                  <div>
                    <h3 className="text-2xl font-bold text-gray-900 mb-4">
                      How to Use This Dashboard
                    </h3>
                    <div className="space-y-3">
                      <div className="flex items-start">
                        <div className="flex-shrink-0 w-8 h-8 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center font-bold mr-4">
                          1
                        </div>
                        <div className="flex-1">
                          <p className="text-gray-700">
                            Review the KPI cards at the top for quick insights into forecast performance
                          </p>
                        </div>
                      </div>
                      <div className="flex items-start">
                        <div className="flex-shrink-0 w-8 h-8 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center font-bold mr-4">
                          2
                        </div>
                        <div className="flex-1">
                          <p className="text-gray-700">
                            Explore the Tableau dashboard for visual trends and patterns (when configured)
                          </p>
                        </div>
                      </div>
                      <div className="flex items-start">
                        <div className="flex-shrink-0 w-8 h-8 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center font-bold mr-4">
                          3
                        </div>
                        <div className="flex-1">
                          <p className="text-gray-700">
                            Export data to CSV for deeper analysis in Excel or Tableau Desktop
                          </p>
                        </div>
                      </div>
                      <div className="flex items-start">
                        <div className="flex-shrink-0 w-8 h-8 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center font-bold mr-4">
                          4
                        </div>
                        <div className="flex-1">
                          <p className="text-gray-700">
                            Use the insights for inventory planning, staffing, and budget decisions
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg p-6">
                    <h4 className="font-semibold text-gray-900 mb-2">Need More Details?</h4>
                    <p className="text-gray-700 mb-4">
                      Switch to the Technical View for detailed model performance, feature importance,
                      and advanced analytics tools.
                    </p>
                    <a
                      href="/"
                      className="inline-flex items-center px-4 py-2 bg-white text-blue-600 rounded-lg hover:bg-gray-50 transition-colors font-medium text-sm border border-blue-300"
                    >
                      Go to Technical View
                      <ArrowRight className="w-4 h-4 ml-2" />
                    </a>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
};

export default BusinessDashboard;
