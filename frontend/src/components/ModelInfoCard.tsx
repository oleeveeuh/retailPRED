/**
 * Model Info Card Component
 * Displays key information about a trained model
 */

import { FC } from 'react';

export interface ModelInfoCardProps {
  model: {
    id: string;
    model_name: string;
    model_type: string;
    category: string;
    metrics: {
      MASE?: number | { mean: number };
      MAPE?: number | { mean: number };
      SMAPE?: number | { mean: number };
      RMSE?: number;
      MAE?: number;
      r2?: number;
    };
    training_date: string;
    is_active: boolean;
  };
}

export const ModelInfoCard: FC<ModelInfoCardProps> = ({ model }) => {
  const getMetricColor = (value: number, lowerIsBetter: boolean = true) => {
    if (lowerIsBetter) {
      if (value <= 0.05) return 'text-green-600';
      if (value <= 0.10) return 'text-yellow-600';
      return 'text-red-600';
    } else {
      if (value >= 0.95) return 'text-green-600';
      if (value >= 0.90) return 'text-yellow-600';
      return 'text-red-600';
    }
  };

  const getMetricValue = (metric: number | { mean: number } | undefined): number => {
    if (typeof metric === 'number') return metric;
    if (typeof metric === 'object' && metric?.mean) return metric.mean;
    return 0;
  };

  const mase = getMetricValue(model.metrics.MASE);
  const mape = getMetricValue(model.metrics.MAPE);
  const smape = getMetricValue(model.metrics.SMAPE);

  return (
    <div className="bg-white dark:bg-slate-800 rounded-lg shadow p-6 hover:shadow-lg transition-shadow">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">{model.model_type}</h3>
          <p className="text-xs text-gray-500 dark:text-gray-400">{model.category}</p>
        </div>
        {model.is_active && (
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-100">
            Active
          </span>
        )}
      </div>

      <div className="space-y-2">
        <div className="flex justify-between items-center">
          <span className="text-sm text-gray-500 dark:text-gray-400">MASE</span>
          <span className={`text-sm font-semibold ${getMetricColor(mase)}`}>
            {mase.toFixed(3)}
          </span>
        </div>

        <div className="flex justify-between items-center">
          <span className="text-sm text-gray-500 dark:text-gray-400">MAPE</span>
          <span className={`text-sm font-semibold ${getMetricColor(mape)}`}>
            {mape.toFixed(2)}%
          </span>
        </div>

        <div className="flex justify-between items-center">
          <span className="text-sm text-gray-500 dark:text-gray-400">sMAPE</span>
          <span className={`text-sm font-semibold ${getMetricColor(smape)}`}>
            {smape.toFixed(2)}%
          </span>
        </div>

        <div className="pt-2 mt-2 border-t border-gray-200 dark:border-slate-700">
          <p className="text-xs text-gray-500 dark:text-gray-400">
            Trained: {new Date(model.training_date).toLocaleDateString()}
          </p>
        </div>
      </div>
    </div>
  );
};

export default ModelInfoCard;

