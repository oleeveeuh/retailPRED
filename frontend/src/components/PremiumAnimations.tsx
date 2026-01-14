import { motion, AnimatePresence } from 'framer-motion';
import { Loader2 } from 'lucide-react';
import type { FC, ReactNode } from 'react';

// ============================================================================
// PREMIUM LOADING SPINNER
// ============================================================================

export const PremiumSpinner: FC<{ size?: number; className?: string }> = ({
  size = 48,
  className = '',
}) => (
  <div className={`flex items-center justify-center ${className}`}>
    <motion.div
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      className="relative"
      style={{ width: size, height: size }}
    >
      {/* Outer ring */}
      <motion.div
        animate={{ rotate: 360 }}
        transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
        className="absolute inset-0 rounded-full border-4 border-transparent border-t-blue-600 border-r-purple-600"
      />

      {/* Middle ring */}
      <motion.div
        animate={{ rotate: -360 }}
        transition={{ duration: 1.5, repeat: Infinity, ease: 'linear' }}
        className="absolute inset-2 rounded-full border-4 border-transparent border-t-emerald-500 border-r-amber-500"
      />

      {/* Inner ring */}
      <motion.div
        animate={{ rotate: 360 }}
        transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
        className="absolute inset-4 rounded-full border-4 border-transparent border-t-blue-500 border-r-purple-500"
      />
    </motion.div>
  </div>
);

// ============================================================================
// SKELETON LOADERS
// ============================================================================

export const SkeletonCard: FC<{ className?: string }> = ({ className = '' }) => (
  <motion.div
    initial={{ opacity: 0 }}
    animate={{ opacity: 1 }}
    className={`bg-white dark:bg-slate-800 rounded-2xl p-6 shadow-lg ${className}`}
  >
    {/* Title skeleton */}
    <div className="h-6 bg-slate-200 dark:bg-slate-700 rounded w-3/4 mb-4 animate-pulse" />

    {/* Content skeleton */}
    <div className="space-y-3">
      <div className="h-4 bg-slate-200 dark:bg-slate-700 rounded w-full animate-pulse" />
      <div className="h-4 bg-slate-200 dark:bg-slate-700 rounded w-5/6 animate-pulse" />
      <div className="h-4 bg-slate-200 dark:bg-slate-700 rounded w-4/6 animate-pulse" />
    </div>

    {/* Button skeleton */}
    <div className="mt-6 h-10 bg-slate-200 dark:bg-slate-700 rounded-lg animate-pulse" />
  </motion.div>
);

export const SkeletonChart: FC<{ className?: string }> = ({ className = '' }) => (
  <motion.div
    initial={{ opacity: 0 }}
    animate={{ opacity: 1 }}
    className={`bg-white dark:bg-slate-800 rounded-2xl p-6 shadow-lg ${className}`}
  >
    {/* Header */}
    <div className="h-6 bg-slate-200 dark:bg-slate-700 rounded w-1/3 mb-6 animate-pulse" />

    {/* Chart area */}
    <div className="h-64 bg-slate-100 dark:bg-slate-700/50 rounded-lg overflow-hidden relative">
      {/* Shimmer effect */}
      <motion.div
        animate={{ x: ['-100%', '200%'] }}
        transition={{ duration: 1.5, repeat: Infinity, ease: 'linear' }}
        className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent"
      />
    </div>
  </motion.div>
);

export const SkeletonTable: FC<{ rows?: number; className?: string }> = ({
  rows = 5,
  className = '',
}) => (
  <motion.div
    initial={{ opacity: 0 }}
    animate={{ opacity: 1 }}
    className={`bg-white dark:bg-slate-800 rounded-2xl shadow-lg overflow-hidden ${className}`}
  >
    {/* Header */}
    <div className="p-6 border-b border-slate-200 dark:border-slate-700">
      <div className="h-6 bg-slate-200 dark:bg-slate-700 rounded w-1/4 animate-pulse" />
    </div>

    {/* Rows */}
    <div className="divide-y divide-slate-200 dark:divide-slate-700">
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="p-6">
          <div className="flex items-center gap-4">
            <div className="h-4 bg-slate-200 dark:bg-slate-700 rounded w-1/4 animate-pulse" />
            <div className="h-4 bg-slate-200 dark:bg-slate-700 rounded w-1/6 animate-pulse" />
            <div className="h-4 bg-slate-200 dark:bg-slate-700 rounded w-1/5 animate-pulse" />
            <div className="h-4 bg-slate-200 dark:bg-slate-700 rounded w-1/6 animate-pulse ml-auto" />
          </div>
        </div>
      ))}
    </div>
  </motion.div>
);

export const SkeletonMetricCard: FC<{ className?: string }> = ({ className = '' }) => (
  <motion.div
    initial={{ opacity: 0 }}
    animate={{ opacity: 1 }}
    className={`bg-white dark:bg-slate-800 rounded-2xl p-6 shadow-lg ${className}`}
  >
    {/* Icon skeleton */}
    <div className="h-10 w-10 bg-slate-200 dark:bg-slate-700 rounded-lg mb-4 animate-pulse" />

    {/* Label skeleton */}
    <div className="h-4 bg-slate-200 dark:bg-slate-700 rounded w-3/4 mb-2 animate-pulse" />

    {/* Value skeleton */}
    <div className="h-8 bg-slate-200 dark:bg-slate-700 rounded w-1/2 mb-3 animate-pulse" />

    {/* Mini chart skeleton */}
    <div className="h-16 bg-slate-100 dark:bg-slate-700/50 rounded-lg animate-pulse" />
  </motion.div>
);

// ============================================================================
// PAGE TRANSITION COMPONENT
// ============================================================================

interface PageTransitionProps {
  children: ReactNode;
  className?: string;
}

export const PageTransition: FC<PageTransitionProps> = ({ children, className = '' }) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    exit={{ opacity: 0, y: -20 }}
    transition={{ duration: 0.3, ease: 'easeInOut' }}
    className={className}
  >
    {children}
  </motion.div>
);

// ============================================================================
// STAGGERED ANIMATIONS FOR LISTS
// ============================================================================

interface StaggeredListProps {
  children: ReactNode;
  className?: string;
  staggerDelay?: number;
}

export const StaggeredList: FC<StaggeredListProps> = ({
  children,
  className = '',
  staggerDelay = 0.1,
}) => (
  <motion.div
    initial="hidden"
    animate="visible"
    variants={{
      visible: {
        transition: {
          staggerChildren: staggerDelay,
        },
      },
    }}
    className={className}
  >
    {children}
  </motion.div>
);

export const StaggeredItem: FC<{ children: ReactNode; className?: string }> = ({
  children,
  className = '',
}) => (
  <motion.div
    variants={{
      hidden: { opacity: 0, y: 20 },
      visible: { opacity: 1, y: 0 },
    }}
    className={className}
  >
    {children}
  </motion.div>
);

// ============================================================================
// PROGRESS BAR
// ============================================================================

interface ProgressBarProps {
  progress: number;
  className?: string;
  color?: 'blue' | 'emerald' | 'amber' | 'purple';
  showLabel?: boolean;
}

export const ProgressBar: FC<ProgressBarProps> = ({
  progress,
  className = '',
  color = 'blue',
  showLabel = false,
}) => {
  const colors = {
    blue: 'bg-primary-600',
    emerald: 'bg-emerald-600',
    amber: 'bg-amber-600',
    purple: 'bg-accent',
  };

  return (
    <div className={`w-full ${className}`}>
      {showLabel && (
        <div className="flex justify-between mb-2">
          <span className="text-sm font-medium text-slate-700 dark:text-slate-300">
            Progress
          </span>
          <span className="text-sm font-medium text-slate-700 dark:text-slate-300">
            {Math.round(progress)}%
          </span>
        </div>
      )}
      <div className="h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${progress}%` }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
          className={`h-full ${colors[color]} relative`}
        >
          {/* Shimmer effect */}
          <motion.div
            animate={{ x: ['-100%', '200%'] }}
            transition={{ duration: 1.5, repeat: Infinity, ease: 'linear' }}
            className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent"
          />
        </motion.div>
      </div>
    </div>
  );
};

// ============================================================================
// CONFETTI ANIMATION
// ============================================================================

export const triggerConfetti = () => {
  import('canvas-confetti').then((confetti) => {
    const duration = 3000;
    const animationEnd = Date.now() + duration;
    const defaults = { startVelocity: 30, spread: 360, ticks: 60, zIndex: 9999 };

    const randomInRange = (min: number, max: number) => Math.random() * (max - min) + min;

    const interval = setInterval(() => {
      const timeLeft = animationEnd - Date.now();

      if (timeLeft <= 0) {
        return clearInterval(interval);
      }

      const particleCount = 50 * (timeLeft / duration);

      // Burst from left side
      confetti.default({
        ...defaults,
        particleCount,
        origin: { x: randomInRange(0.1, 0.3), y: Math.random() - 0.2 },
      });

      // Burst from right side
      confetti.default({
        ...defaults,
        particleCount,
        origin: { x: randomInRange(0.7, 0.9), y: Math.random() - 0.2 },
      });
    }, 250);
  });
};

// ============================================================================
// LOADING OVERLAY
// ============================================================================

interface LoadingOverlayProps {
  isLoading: boolean;
  message?: string;
}

export const LoadingOverlay: FC<LoadingOverlayProps> = ({ isLoading, message = 'Loading...' }) => (
  <AnimatePresence>
    {isLoading && (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-slate-900/50 backdrop-blur-sm z-50 flex items-center justify-center"
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          className="bg-white dark:bg-slate-800 rounded-2xl p-8 shadow-2xl max-w-sm w-full"
        >
          <PremiumSpinner size={64} className="mb-4" />
          <p className="text-center text-slate-700 dark:text-slate-300 font-medium">{message}</p>
        </motion.div>
      </motion.div>
    )}
  </AnimatePresence>
);

// ============================================================================
// ERROR BOUNDARY FALLBACK
// ============================================================================

export const ErrorFallback: FC<{
  error: Error;
  resetErrorBoundary: () => void;
}> = ({ error, resetErrorBoundary }) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    className="min-h-screen flex items-center justify-center p-4 bg-slate-50 dark:bg-slate-900"
  >
    <div className="max-w-md w-full bg-white dark:bg-slate-800 rounded-2xl p-8 shadow-2xl text-center">
      <motion.div
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        transition={{ delay: 0.2, type: 'spring', stiffness: 200 }}
        className="w-16 h-16 bg-red-100 dark:bg-red-900/20 rounded-full flex items-center justify-center mx-auto mb-4"
      >
        <svg
          className="w-8 h-8 text-red-600 dark:text-red-400"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
          />
        </svg>
      </motion.div>

      <h2 className="text-2xl font-bold text-slate-900 dark:text-white mb-2">
        Something went wrong
      </h2>

      <p className="text-slate-600 dark:text-slate-400 mb-6">
        {error.message || 'An unexpected error occurred'}
      </p>

      <motion.button
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        onClick={resetErrorBoundary}
        className="w-full px-6 py-3 bg-gradient-to-r from-blue-600 to-accent text-white rounded-lg font-semibold shadow-lg shadow-blue-500/50 hover:shadow-xl transition-all"
      >
        Try Again
      </motion.button>
    </div>
  </motion.div>
);

// ============================================================================
// EMPTY STATE WITH ANIMATION
// ============================================================================

interface EmptyStateAnimationProps {
  children: ReactNode;
  delay?: number;
}

export const EmptyStateAnimation: FC<EmptyStateAnimationProps> = ({
  children,
  delay = 0,
}) => (
  <motion.div
    initial={{ opacity: 0, scale: 0.9 }}
    animate={{ opacity: 1, scale: 1 }}
    transition={{ delay, duration: 0.4, ease: 'easeOut' }}
  >
    {children}
  </motion.div>
);

export default PremiumSpinner;
