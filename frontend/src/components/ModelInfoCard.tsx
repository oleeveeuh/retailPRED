/**
 * Model Info Card Component
 * Displays key information about a trained model with VALIDATION metrics
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
      RMSE?: number | { mean: number };
      MAE?: number | { mean: number };
      r2?: number;
    };
    training_date: string;
    is_active: boolean;
    validated_predictions?: number;
    total_predictions?: number;
  };
}

export const ModelInfoCard: FC<ModelInfoCardProps> = ({ model }) => {
  const getMetricColor = (value: number, lowerIsBetter: boolean = true) => {
    if (lowerIsBetter) {
      if (value <= 5) return 'text-green-600 dark:text-green-400';
      if (value <= 10) return 'text-yellow-600 dark:text-yellow-400';
      return 'text-red-600 dark:text-red-400';
    } else {
      if (value >= 0.95) return 'text-green-600 dark:text-green-400';
      if (value >= 0.90) return 'text-yellow-600 dark:text-yellow-400';
      return 'text-red-600 dark:text-red-400';
    }
  };

  const getMetricValue = (metric: number | { mean: number } | undefined): number => {
    if (typeof metric === 'number') return metric;
    if (typeof metric === 'object' && metric?.mean) return metric.mean;
    return 0;
  };

  const mape = getMetricValue(model.metrics.MAPE);
  const rmse = model.metrics.RMSE ? getMetricValue(model.metrics.RMSE) : null;
  const mae = model.metrics.MAE ? getMetricValue(model.metrics.MAE) : null;

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
          <span className="text-sm text-gray-500 dark:text-gray-400">MAPE (Validation)</span>
          <span className={`text-sm font-semibold ${getMetricColor(mape)}`}>
            {mape.toFixed(2)}%
          </span>
        </div>

        {mae !== null && (
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-500 dark:text-gray-400">MAE (Validation)</span>
            <span className="text-sm font-semibold text-gray-700 dark:text-gray-300">
              ${mae.toFixed(2)}
            </span>
          </div>
        )}

        {rmse !== null && (
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-500 dark:text-gray-400">RMSE (Validation)</span>
            <span className="text-sm font-semibold text-gray-700 dark:text-gray-300">
              ${rmse.toFixed(2)}
            </span>
          </div>
        )}

        {model.validated_predictions !== undefined && model.total_predictions !== undefined && (
          <div className="pt-2 mt-2 border-t border-gray-200 dark:border-slate-700">
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Validated: {model.validated_predictions}/{model.total_predictions} predictions
              ({Math.round((model.validated_predictions / model.total_predictions) * 100)}%)
            </p>
          </div>
        )}

        <div className="pt-2 mt-2 border-t border-gray-200 dark:border-slate-700">
          <p className="text-xs text-gray-500 dark:text-gray-400">
            Based on actual test data
          </p>
        </div>
      </div>
    </div>
  );
};

export default ModelInfoCard;

