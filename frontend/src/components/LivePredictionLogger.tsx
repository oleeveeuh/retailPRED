/**
 * Live Prediction Logger Component
 * Shows predictions being processed in real-time
 */

import { FC } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { CheckCircle2, Clock, XCircle, Loader2, Sparkles, TrendingUp } from 'lucide-react';
import { useLivePredictions } from '../../hooks/useRealtimeData';

export const LivePredictionLogger: FC = () => {
  const { predictions, isProcessing, clearCompleted } = useLivePredictions();

  if (predictions.length === 0 && !isProcessing) {
    return null;
  }

  return (
    <div className="fixed bottom-4 left-4 z-40">
      <div className="glass-card p-4 rounded-xl shadow-2xl max-w-md">
        {/* Header */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${isProcessing ? 'bg-primary animate-pulse' : 'bg-slate-400'}`} />
            <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
              Live Predictions
            </h3>
            {isProcessing && (
              <span className="text-xs text-primary-600 dark:text-blue-400 font-medium">
                Processing...
              </span>
            )}
          </div>
          {predictions.some(p => p.status === 'completed') && (
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={clearCompleted}
              className="text-xs text-slate-500 hover:text-slate-700 dark:hover:text-slate-300 transition-colors"
            >
              Clear completed
            </motion.button>
          )}
        </div>

        {/* Predictions List */}
        <div className="space-y-2 max-h-64 overflow-y-auto">
          <AnimatePresence mode="popLayout">
            {predictions.slice(0, 5).map((prediction) => (
              <motion.div
                key={prediction.id}
                initial={{ opacity: 0, x: -20, scale: 0.95 }}
                animate={{ opacity: 1, x: 0, scale: 1 }}
                exit={{ opacity: 0, x: 20, scale: 0.95 }}
                transition={{ duration: 0.3 }}
                className={`p-3 rounded-lg border ${
                  prediction.status === 'completed'
                    ? 'bg-emerald-50 dark:bg-emerald-900/20 border-emerald-200 dark:border-emerald-800'
                    : prediction.status === 'processing'
                    ? 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800'
                    : prediction.status === 'failed'
                    ? 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800'
                    : 'bg-slate-50 dark:bg-slate-800/50 border-slate-200 dark:border-slate-700'
                }`}
              >
                <div className="flex items-start gap-3">
                  {/* Icon */}
                  <div className="flex-shrink-0 mt-0.5">
                    {prediction.status === 'queued' && (
                      <Clock className="w-4 h-4 text-slate-400" />
                    )}
                    {prediction.status === 'processing' && (
                      <Loader2 className="w-4 h-4 text-primary animate-spin" />
                    )}
                    {prediction.status === 'completed' && (
                      <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                    )}
                    {prediction.status === 'failed' && (
                      <XCircle className="w-4 h-4 text-red-500" />
                    )}
                  </div>

                  {/* Content */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between mb-1">
                      <p className="text-sm font-medium text-slate-900 dark:text-slate-100">
                        {prediction.category}
                      </p>
                      <span className="text-xs text-slate-500 dark:text-slate-400">
                        {prediction.model}
                      </span>
                    </div>

                    {/* Progress Bar (for processing) */}
                    {prediction.status === 'processing' && prediction.progress !== undefined && (
                      <div className="mt-2">
                        <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400 mb-1">
                          <span>Generating...</span>
                          <span>{prediction.progress}%</span>
                        </div>
                        <div className="h-1.5 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                          <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${prediction.progress}%` }}
                            transition={{ duration: 0.3 }}
                            className="h-full bg-primary rounded-full"
                          />
                        </div>
                      </div>
                    )}

                    {/* Status Text */}
                    {prediction.status === 'queued' && (
                      <p className="text-xs text-slate-500 dark:text-slate-400">
                        Position in queue: {predictions.findIndex(p => p.id === prediction.id) + 1}
                      </p>
                    )}
                    {prediction.status === 'completed' && (
                      <p className="text-xs text-emerald-600 dark:text-emerald-400 font-medium">
                        Prediction complete!
                      </p>
                    )}
                    {prediction.status === 'failed' && (
                      <p className="text-xs text-red-600 dark:text-red-400 font-medium">
                        Prediction failed
                      </p>
                    )}
                  </div>

                  {/* Time */}
                  <div className="flex-shrink-0 text-xs text-slate-400">
                    {new Date(prediction.timestamp).toLocaleTimeString([], {
                      hour: '2-digit',
                      minute: '2-digit',
                    })}
                  </div>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>

        {/* Queue Info */}
        {predictions.some(p => p.status === 'queued' || p.status === 'processing') && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-3 pt-3 border-t border-slate-200 dark:border-slate-700"
          >
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-500 dark:text-slate-400">
                {predictions.filter(p => p.status === 'queued' || p.status === 'processing').length} in queue
              </span>
              <span className="text-slate-400">
                ~{Math.ceil(predictions.filter(p => p.status === 'queued' || p.status === 'processing').length * 3 / 2)}s remaining
              </span>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
};

/**
 * Training Progress Modal Component
 * Shows real-time training progress
 */

interface TrainingProgressModalProps {
  isOpen: boolean;
  progress: number;
  stage: string;
  model: string;
  onComplete?: () => void;
}

export const TrainingProgressModal: FC<TrainingProgressModalProps> = ({
  isOpen,
  progress,
  stage,
  model,
  onComplete,
}) => {
  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            className="glass-card max-w-md w-full p-6"
          >
            {/* Header */}
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 bg-blue-100 dark:bg-blue-900/20 rounded-lg">
                <Sparkles className="w-6 h-6 text-primary-600" />
              </div>
              <div>
                <h3 className="text-lg font-bold text-slate-900 dark:text-slate-100">
                  Model Training
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                  {model}
                </p>
              </div>
            </div>

            {/* Progress */}
            <div className="space-y-3">
              <div className="flex items-center justify-between text-sm">
                <span className="text-slate-600 dark:text-slate-400">{stage}</span>
                <span className="font-semibold text-slate-900 dark:text-slate-100">{progress}%</span>
              </div>

              <div className="h-3 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${progress}%` }}
                  transition={{ duration: 0.3 }}
                  className="h-full bg-gradient-to-r from-blue-600 to-accent rounded-full relative"
                >
                  <motion.div
                    className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent"
                    animate={{ x: ['-100%', '100%'] }}
                    transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                  />
                </motion.div>
              </div>

              {/* Stages */}
              <div className="space-y-2">
                {[
                  { label: 'Initializing', threshold: 10 },
                  { label: 'Loading data', threshold: 25 },
                  { label: 'Preprocessing', threshold: 40 },
                  { label: 'Training model', threshold: 60 },
                  { label: 'Validating', threshold: 80 },
                  { label: 'Saving model', threshold: 95 },
                ].map((step, index) => (
                  <div
                    key={index}
                    className={`flex items-center gap-2 text-xs transition-colors ${
                      progress >= step.threshold
                        ? 'text-primary-600 dark:text-blue-400'
                        : 'text-slate-400'
                    }`}
                  >
                    <div
                      className={`w-4 h-4 rounded-full flex items-center justify-center border-2 ${
                        progress >= step.threshold
                          ? 'border-blue-600 bg-primary-600'
                          : 'border-slate-300'
                      }`}
                    >
                      {progress >= step.threshold && (
                        <motion.div
                          initial={{ scale: 0 }}
                          animate={{ scale: 1 }}
                          transition={{ delay: 0.1 }}
                        >
                          <CheckCircle2 className="w-3 h-3 text-white" />
                        </motion.div>
                      )}
                    </div>
                    <span>{step.label}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Complete Message */}
            <AnimatePresence>
              {progress === 100 && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mt-4 p-3 bg-emerald-50 dark:bg-emerald-900/20 rounded-lg border border-emerald-200 dark:border-emerald-800"
                >
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="w-5 h-5 text-emerald-600" />
                    <div>
                      <p className="text-sm font-semibold text-emerald-900 dark:text-emerald-200">
                        Training Complete!
                      </p>
                      <p className="text-xs text-emerald-700 dark:text-emerald-400">
                        Model is now ready for predictions
                      </p>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
};
