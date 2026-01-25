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
    type: 'expansion',
    name: 'Economic Expansion',
    description: 'Strong expansion with robust growth, low unemployment, high confidence',
    icon: TrendingUp,
    color: 'emerald',
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
      return await scenariosApi.analyzeScenario({
        scenario_type: selectedScenario,
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

  // Fetch model comparisons - using scenario API with model names
  const { data: modelPredictions } = useQuery({
    queryKey: ['model-predictions', category, selectedScenario],
    queryFn: async () => {
      const models = ['LGBM', 'RandomForest'];
      const predictions = await Promise.all(
        models.map(async (model) => {
          try {
            const data = await scenariosApi.analyzeModelScenario({
              category,
              model_name: model,
              scenario_type: selectedScenario
            });
            return {
              name: model,
              value: data.prediction || 0,
              color: model === 'LGBM' ? 'bg-primary' : 'bg-green-500'
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
    <div className="min-h-screen bg-white p-8">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div>
          <h1 className="text-2xl font-normal text-gray-900 tracking-tight">
            Economic Scenario Analysis
          </h1>
          <p className="text-gray-500 text-sm font-light mt-1">
            Analyze retail sales forecasts under different macroeconomic scenarios
          </p>
        </div>

        {/* Current Economic Regime */}
        {currentRegime && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white border border-gray-200 rounded-sm p-5"
          >
            <h2 className="text-sm font-medium text-gray-900 uppercase tracking-wide mb-4 flex items-center">
              <Activity className="w-4 h-4 mr-2 text-[#3A3A6C]" />
              Current Economic Regime
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
              <div className="text-center">
                <div className="text-2xl font-normal text-[#3A3A6C] mb-2">
                  {currentRegime.regime.charAt(0).toUpperCase() + currentRegime.regime.slice(1)}
                </div>
                <div className="text-xs text-gray-500 uppercase tracking-wide">
                  Detected Regime
                </div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-normal text-green-600 mb-2">
                  {(currentRegime.confidence * 100).toFixed(0)}%
                </div>
                <div className="text-xs text-gray-500 uppercase tracking-wide">
                  Confidence
                </div>
              </div>
              <div className="text-center">
                <div className="text-xs text-gray-500 flex items-center justify-center">
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
          className="bg-white border border-gray-200 rounded-sm p-5"
        >
          <h2 className="text-sm font-medium text-gray-900 uppercase tracking-wide mb-4">
            Select Economic Scenario
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-6 gap-4 mb-5">
            {SCENARIOS.map((scenario) => {
              const Icon = scenario.icon;
              const isSelected = selectedScenario === scenario.type;

              return (
                <button
                  key={scenario.type}
                  onClick={() => setSelectedScenario(scenario.type)}
                  className={`relative p-4 rounded-sm border-2 transition-all ${
                    isSelected
                      ? `border-[#3A3A6C] bg-[#3A3A6C]/5`
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                >
                  <Icon className={`w-6 h-6 mb-2 ${isSelected ? 'text-[#3A3A6C]' : `text-gray-400`}`} />
                  <div className="font-normal text-sm text-gray-900 mb-1">
                    {scenario.name}
                  </div>
                  <div className="text-xs text-gray-500">
                    {scenario.description}
                  </div>
                </button>
              );
            })}
          </div>

          {/* Category Selector */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Retail Category
            </label>
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              className="w-full max-w-xs px-4 py-2 rounded-sm border border-gray-200 bg-white text-gray-900"
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
            className="grid grid-cols-1 lg:grid-cols-2 gap-5"
          >
            {/* Prediction Under Scenario */}
            <div className="bg-white border border-gray-200 rounded-sm p-5">
              <h3 className="text-sm font-medium text-gray-900 uppercase tracking-wide mb-4 flex items-center">
                <DollarSign className="w-4 h-4 mr-2 text-green-600" />
                Forecast Under {selectedScenarioInfo?.name}
              </h3>

              <div className="text-center mb-5">
                <div className="text-3xl font-normal text-gray-900 mb-2">
                  ${scenarioResults.prediction.toLocaleString()}
                </div>
                <div className="text-xs text-gray-500 uppercase tracking-wide">
                  Predicted Monthly Retail Sales
                </div>
              </div>

              {/* Confidence Interval */}
              <div className="mb-5">
                <div className="flex justify-between text-xs text-gray-500 mb-2 uppercase tracking-wide">
                  <span>Confidence Interval (95%)</span>
                </div>
                <div className="relative h-8 bg-gray-200 rounded-sm overflow-hidden">
                  <div
                    className="absolute h-full bg-[#81C1AC]"
                    style={{
                      left: `${((scenarioResults.confidence_interval[0] - 40000) / 40000) * 100}%`,
                      width: `${((scenarioResults.confidence_interval[1] - scenarioResults.confidence_interval[0]) / 40000) * 100}%`,
                    }}
                  />
                  <div className="absolute inset-0 flex items-center justify-center text-xs font-normal text-gray-900">
                    ${scenarioResults.confidence_interval[0].toLocaleString()} - ${scenarioResults.confidence_interval[1].toLocaleString()}
                  </div>
                </div>
              </div>

              {/* Scenario Description */}
              <div className="bg-gray-50 rounded-sm p-4">
                <p className="text-xs text-gray-700">
                  {selectedScenarioInfo?.description}
                </p>
              </div>
            </div>

            {/* Macro Factor Attribution */}
            <div className="bg-white border border-gray-200 rounded-sm p-5">
              <h3 className="text-sm font-medium text-gray-900 uppercase tracking-wide mb-4 flex items-center">
                <BarChart3 className="w-4 h-4 mr-2 text-[#3A3A6C]" />
                Macro Factor Impact
              </h3>

              <div className="space-y-3">
                {scenarioResults.impact_summary.slice(0, 6).map((impact, idx) => (
                  <div key={idx} className="border-b border-gray-200 pb-3 last:border-0">
                    <div className="flex justify-between items-start mb-1">
                      <div>
                        <div className="font-normal text-xs text-gray-900">
                          {impact.indicator}
                        </div>
                        <div className="text-xs text-gray-500">
                          {impact.category} • {impact.source}
                        </div>
                      </div>
                      <div className="text-right">
                        <div className={`font-normal text-xs ${
                          impact.change_pct > 0 ? 'text-green-600' : 'text-red-600'
                        }`}>
                          {impact.change_pct > 0 ? '+' : ''}{impact.change_pct.toFixed(1)}%
                        </div>
                      </div>
                    </div>
                    <div className="flex justify-between text-xs text-gray-500">
                      <span>{impact.base_value.toFixed(2)} → {impact.scenario_value.toFixed(2)}</span>
                    </div>
                  </div>
                ))}
              </div>

              {/* Model Predictions Chart */}
              <div className="mt-5 pt-5 border-t border-gray-200">
                <h4 className="text-xs font-medium text-gray-700 uppercase tracking-wide mb-4">
                  Model Predictions Under This Scenario
                </h4>
                {modelPredictions && modelPredictions.length > 0 ? (
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    {modelPredictions.map((model) => (
                      <div key={model.name} className="bg-gray-50 rounded-sm p-3">
                        <div className="text-xs text-gray-500 mb-1">{model.name}</div>
                        <div className="text-sm font-normal text-gray-900">
                          ${(model.value / 1000).toFixed(1)}K
                        </div>
                        <div className="mt-2 h-1.5 bg-gray-200 rounded-sm overflow-hidden">
                          <div
                            className={`h-full ${model.color} rounded-sm`}
                            style={{
                              width: `${(model.value / (Math.max(...modelPredictions.map(m => m.value)) * 1.05)) * 100}%`
                            }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-xs text-gray-500 text-center py-4">
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
            className="bg-white border border-gray-200 rounded-sm p-5"
          >
            <h3 className="text-sm font-medium text-gray-900 uppercase tracking-wide mb-4">
              Historical Pattern Matching
            </h3>
            <p className="text-xs text-gray-500 mb-5">
              Finding similar economic periods in history can help predict future outcomes
            </p>

            <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
              {similarPeriods.periods.map((period: any, idx: number) => (
                <div key={idx} className="border border-gray-200 rounded-sm p-4">
                  <div className="text-center mb-3">
                    <div className="text-sm font-normal text-gray-900 mb-1">
                      {new Date(period.date).toLocaleDateString('en-US', { month: 'short', year: 'numeric' })}
                    </div>
                    <div className="text-xs text-gray-500">
                      Similarity: {(period.similarity_score * 100).toFixed(0)}%
                    </div>
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-500">Unemployment</span>
                      <span className="font-normal text-gray-900">
                        {period.indicators.UNRATE?.toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-500">GDP Growth</span>
                      <span className="font-normal text-gray-900">
                        {period.indicators.GDP?.toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-500">Retail Sales</span>
                      <span className="font-normal text-green-600">
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
