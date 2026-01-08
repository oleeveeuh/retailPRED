/**
 * Economic Scenario Analysis Page
 * Analyzes retail sales forecasts under different macroeconomic scenarios
 */

import { FC, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import { scenariosApi, economicIndicatorsApi, predictionsApi } from '../api/unifiedApi';
import {
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  CheckCircle,
  Info,
  BarChart3,
  Activity,
  DollarSign,
} from 'lucide-react';

interface Scenario {
  type: string;
  name: string;
  description: string;
  icon: FC<{ className?: string }>;
  color: string;
}

interface ScenarioResult {
  scenario_type: string;
  scenario_name: string;
  description: string;
  prediction: number;
  confidence_interval: [number, number];
  impact_summary: Array<{
    indicator: string;
    category: string;
    source: string;
    base_value: number;
    scenario_value: number;
    change: number;
    change_pct: number;
  }>;
}

const SCENARIOS: Scenario[] = [
  {
    type: 'recession',
    name: 'Recession',
    description: 'Economic downturn with elevated unemployment and negative GDP growth',
    icon: TrendingDown,
    color: 'red',
  },
  {
    type: 'rate_hike',
    name: 'Rate Hike Cycle',
    description: 'Tightening monetary policy with higher interest rates',
    icon: TrendingUp,
    color: 'orange',
  },
  {
    type: 'inflation_surge',
    name: 'Inflation Surge',
    description: 'High inflation environment with elevated consumer prices',
    icon: AlertTriangle,
    color: 'yellow',
  },
  {
    type: 'recovery',
    name: 'Economic Recovery',
    description: 'Strong growth with falling unemployment and rising confidence',
    icon: CheckCircle,
    color: 'green',
  },
  {
    type: 'baseline',
    name: 'Baseline',
    description: 'Continue current economic conditions with no changes',
    icon: Activity,
    color: 'blue',
  },
];

export const EconomicScenarioAnalysis: FC = () => {
  const [selectedScenario, setSelectedScenario] = useState<string>('baseline');
  const [category, setCategory] = useState<string>('total_sales');

  // Fetch current economic indicators
  const { data: currentIndicators, isLoading: loadingIndicators } = useQuery({
    queryKey: ['current-indicators'],
    queryFn: async () => {
      return await economicIndicatorsApi.getCurrent();
    },
  });

  // Fetch scenario predictions
  const { data: scenarioResults, isLoading: loadingScenarios } = useQuery({
    queryKey: ['scenario-results', category, selectedScenario],
    queryFn: async () => {
      const scenarioType = selectedScenario === 'baseline' ? 'baseline' :
                         selectedScenario === 'optimistic' ? 'optimistic' : 'pessimistic';
      return await scenariosApi.analyzeScenario({
        scenario_type: scenarioType,
        category
      });
    },
    enabled: !!category && !!selectedScenario,
  });

  // Fetch historical similar periods
  const { data: similarPeriods } = useQuery({
    queryKey: ['similar-periods', category],
    queryFn: async () => {
      return await scenariosApi.getSimilarPeriods(category, 5);
    },
  });

  // Fetch regime detection
  const { data: currentRegime } = useQuery({
    queryKey: ['current-regime', category],
    queryFn: async () => {
      return await scenariosApi.getRegime(category);
    },
  });

  // Fetch model comparisons - using predict API with model names
  const { data: modelPredictions } = useQuery({
    queryKey: ['model-predictions', category, selectedScenario],
    queryFn: async () => {
      const models = ['LGBM', 'RandomForest', 'PatchTST', 'TimesNet'];
      const predictions = await Promise.all(
        models.map(async (model) => {
          try {
            const data = await predictionsApi.predict({
              category,
              model_name: model,
              weeks_ahead: 4
            });
            return {
              name: model,
              value: data.forecasts[0]?.predicted_value || 0,
              color: model === 'LGBM' ? 'bg-blue-500' :
                     model === 'RandomForest' ? 'bg-green-500' :
                     model === 'PatchTST' ? 'bg-purple-500' :
                     'bg-orange-500'
            };
          } catch {
            return null;
          }
        })
      );
      return predictions.filter((p): p is NonNullable<typeof p> => p !== null);
    },
    enabled: !!category && !!selectedScenario,
  });

  const selectedScenarioInfo = SCENARIOS.find(s => s.type === selectedScenario);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 p-8">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header */}
        <div>
          <h1 className="text-4xl font-bold text-slate-900 dark:text-white mb-2">
            Economic Scenario Analysis
          </h1>
          <p className="text-slate-600 dark:text-slate-400">
            Analyze retail sales forecasts under different macroeconomic scenarios
          </p>
        </div>

        {/* Current Economic Regime */}
        {currentRegime && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white dark:bg-slate-800 rounded-lg shadow-lg p-6"
          >
            <h2 className="text-2xl font-semibold text-slate-900 dark:text-white mb-4 flex items-center">
              <Activity className="w-6 h-6 mr-2 text-blue-600" />
              Current Economic Regime
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="text-center">
                <div className="text-3xl font-bold text-blue-600 dark:text-blue-400 mb-2">
                  {currentRegime.regime}
                </div>
                <div className="text-sm text-slate-600 dark:text-slate-400">
                  Detected Regime
                </div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-green-600 dark:text-green-400 mb-2">
                  {(currentRegime.confidence * 100).toFixed(0)}%
                </div>
                <div className="text-sm text-slate-600 dark:text-slate-400">
                  Confidence
                </div>
              </div>
              <div className="text-center">
                <div className="text-sm text-slate-600 dark:text-slate-400 flex items-center justify-center">
                  <Info className="w-4 h-4 mr-2" />
                  {currentRegime.description}
                </div>
              </div>
            </div>
          </motion.div>
        )}

        {/* Scenario Selector */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-white dark:bg-slate-800 rounded-lg shadow-lg p-6"
        >
          <h2 className="text-2xl font-semibold text-slate-900 dark:text-white mb-4">
            Select Economic Scenario
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-5 gap-4 mb-6">
            {SCENARIOS.map((scenario) => {
              const Icon = scenario.icon;
              const isSelected = selectedScenario === scenario.type;

              return (
                <button
                  key={scenario.type}
                  onClick={() => setSelectedScenario(scenario.type)}
                  className={`relative p-4 rounded-lg border-2 transition-all ${
                    isSelected
                      ? `border-${scenario.color}-500 bg-${scenario.color}-50 dark:bg-${scenario.color}-900/20`
                      : 'border-slate-200 dark:border-slate-700 hover:border-slate-300'
                  }`}
                >
                  <Icon className={`w-8 h-8 mb-2 text-${scenario.color}-600`} />
                  <div className="font-semibold text-slate-900 dark:text-white mb-1">
                    {scenario.name}
                  </div>
                  <div className="text-xs text-slate-600 dark:text-slate-400">
                    {scenario.description}
                  </div>
                </button>
              );
            })}
          </div>

          {/* Category Selector */}
          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Retail Category
            </label>
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
          </div>
        </motion.div>

        {/* Scenario Results */}
        {scenarioResults && !loadingScenarios && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="grid grid-cols-1 lg:grid-cols-2 gap-6"
          >
            {/* Prediction Under Scenario */}
            <div className="bg-white dark:bg-slate-800 rounded-lg shadow-lg p-6">
              <h3 className="text-xl font-semibold text-slate-900 dark:text-white mb-4 flex items-center">
                <DollarSign className="w-5 h-5 mr-2 text-green-600" />
                Forecast Under {selectedScenarioInfo?.name}
              </h3>

              <div className="text-center mb-6">
                <div className="text-4xl font-bold text-slate-900 dark:text-white mb-2">
                  ${scenarioResults.prediction.toLocaleString()}
                </div>
                <div className="text-sm text-slate-600 dark:text-slate-400">
                  Predicted Monthly Retail Sales
                </div>
              </div>

              {/* Confidence Interval */}
              <div className="mb-6">
                <div className="flex justify-between text-sm text-slate-600 dark:text-slate-400 mb-2">
                  <span>Confidence Interval (95%)</span>
                </div>
                <div className="relative h-8 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                  <div
                    className="absolute h-full bg-blue-600 dark:bg-blue-400"
                    style={{
                      left: `${((scenarioResults.confidence_interval[0] - 40000) / 40000) * 100}%`,
                      width: `${((scenarioResults.confidence_interval[1] - scenarioResults.confidence_interval[0]) / 40000) * 100}%`,
                    }}
                  />
                  <div className="absolute inset-0 flex items-center justify-center text-sm font-semibold text-slate-900 dark:text-white">
                    ${scenarioResults.confidence_interval[0].toLocaleString()} - ${scenarioResults.confidence_interval[1].toLocaleString()}
                  </div>
                </div>
              </div>

              {/* Scenario Description */}
              <div className="bg-slate-50 dark:bg-slate-900 rounded-lg p-4">
                <p className="text-sm text-slate-700 dark:text-slate-300">
                  {selectedScenarioInfo?.description}
                </p>
              </div>
            </div>

            {/* Macro Factor Attribution */}
            <div className="bg-white dark:bg-slate-800 rounded-lg shadow-lg p-6">
              <h3 className="text-xl font-semibold text-slate-900 dark:text-white mb-4 flex items-center">
                <BarChart3 className="w-5 h-5 mr-2 text-blue-600" />
                Macro Factor Impact
              </h3>

              <div className="space-y-3">
                {scenarioResults.impact_summary.slice(0, 6).map((impact, idx) => (
                  <div key={idx} className="border-b border-slate-200 dark:border-slate-700 pb-3 last:border-0">
                    <div className="flex justify-between items-start mb-1">
                      <div>
                        <div className="font-medium text-slate-900 dark:text-white">
                          {impact.indicator}
                        </div>
                        <div className="text-xs text-slate-600 dark:text-slate-400">
                          {impact.category} • {impact.source}
                        </div>
                      </div>
                      <div className="text-right">
                        <div className={`font-semibold ${
                          impact.change_pct > 0 ? 'text-green-600' : 'text-red-600'
                        }`}>
                          {impact.change_pct > 0 ? '+' : ''}{impact.change_pct.toFixed(1)}%
                        </div>
                      </div>
                    </div>
                    <div className="flex justify-between text-xs text-slate-600 dark:text-slate-400">
                      <span>{impact.base_value.toFixed(2)} → {impact.scenario_value.toFixed(2)}</span>
                    </div>
                  </div>
                ))}
              </div>

              {/* Model Predictions Chart */}
              <div className="mt-6 pt-6 border-t border-slate-200 dark:border-slate-700">
                <h4 className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-4">
                  Model Predictions Under This Scenario
                </h4>
                {modelPredictions && modelPredictions.length > 0 ? (
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    {modelPredictions.map((model) => (
                      <div key={model.name} className="bg-slate-50 dark:bg-slate-900 rounded-lg p-3">
                        <div className="text-xs text-slate-600 dark:text-slate-400 mb-1">{model.name}</div>
                        <div className="text-lg font-bold text-slate-900 dark:text-white">
                          ${(model.value / 1000).toFixed(1)}K
                        </div>
                        <div className="mt-2 h-1.5 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                          <div
                            className={`h-full ${model.color} rounded-full`}
                            style={{
                              width: `${(model.value / (Math.max(...modelPredictions.map(m => m.value)) * 1.05)) * 100}%`
                            }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-sm text-slate-500 dark:text-slate-400 text-center py-4">
                    Loading model predictions...
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        )}

        {/* Historical Pattern Matching */}
        {similarPeriods && similarPeriods.periods && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="bg-white dark:bg-slate-800 rounded-lg shadow-lg p-6"
          >
            <h3 className="text-xl font-semibold text-slate-900 dark:text-white mb-4">
              Historical Pattern Matching
            </h3>
            <p className="text-sm text-slate-600 dark:text-slate-400 mb-6">
              Finding similar economic periods in history can help predict future outcomes
            </p>

            <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
              {similarPeriods.periods.map((period: any, idx: number) => (
                <div key={idx} className="border border-slate-200 dark:border-slate-700 rounded-lg p-4">
                  <div className="text-center mb-3">
                    <div className="text-lg font-bold text-slate-900 dark:text-white mb-1">
                      {new Date(period.date).toLocaleDateString('en-US', { month: 'short', year: 'numeric' })}
                    </div>
                    <div className="text-sm text-slate-600 dark:text-slate-400">
                      Similarity: {(period.similarity_score * 100).toFixed(0)}%
                    </div>
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-slate-600 dark:text-slate-400">Unemployment</span>
                      <span className="font-medium text-slate-900 dark:text-white">
                        {period.indicators.UNRATE?.toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-slate-600 dark:text-slate-400">GDP Growth</span>
                      <span className="font-medium text-slate-900 dark:text-white">
                        {period.indicators.GDP?.toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-slate-600 dark:text-slate-400">Retail Sales</span>
                      <span className="font-medium text-green-600">
                        ${(period.retail_sales / 1000).toFixed(0)}K
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        )}

        {/* Loading State */}
        {(loadingIndicators || loadingScenarios) && (
          <div className="flex justify-center py-12">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
          </div>
        )}
      </div>
    </div>
  );
};

export default EconomicScenarioAnalysis;
