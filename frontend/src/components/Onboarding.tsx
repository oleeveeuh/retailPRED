import { useState, useEffect, FC, ReactNode } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  X,
  Sparkles,
  TrendingUp,
  Brain,
  Calendar,
  Upload,
  ArrowRight,
  ArrowLeft,
  CheckCircle2,
  HelpCircle,
  Info,
} from 'lucide-react';
import toast from 'react-hot-toast';

// ============================================================================
// TYPES
// ============================================================================

export interface TourStep {
  target: string;
  title: string;
  content: string;
  placement?: 'top' | 'bottom' | 'left' | 'right' | 'center';
  disableBeacon?: boolean;
}

export interface OnboardingProps {
  isOpen: boolean;
  onClose: () => void;
  currentStep?: number;
  totalSteps?: number;
  title?: string;
  children?: ReactNode;
}

export interface EmptyStateProps {
  type: 'predictions' | 'models' | 'validations' | 'counterfactuals' | 'shap';
  onAction?: () => void;
  demoMode?: boolean;
}

export interface DemoModeBannerProps {
  isActive: boolean;
  onExit?: () => void;
}

export interface HelpTooltipProps {
  content: string;
  documentationLink?: string;
  term: string;
}

// ============================================================================
// TOUR CONFIGURATION
// ============================================================================

export const TOUR_STEPS: TourStep[] = [
  {
    target: '.predict-nav-item',
    title: '🎯 Generate Predictions',
    content: 'Start by selecting a retail category and ML model to generate accurate sales forecasts. Our ensemble of models (LightGBM, XGBoost, Random Forest) provides predictions with confidence intervals.',
    placement: 'right',
    disableBeacon: true,
  },
  {
    target: '.explain-nav-item',
    title: '🧠 Understand SHAP Values',
    content: 'Dive into model interpretability with SHAP (SHapley Additive exPlanations). See exactly which features (economic indicators, seasonality, trends) influenced each prediction.',
    placement: 'right',
  },
  {
    target: '.counterfactual-nav-item',
    title: '🔮 Explore Counterfactuals',
    content: 'Run "what-if" scenarios to understand how changing economic conditions (CPI, unemployment, interest rates) would affect your forecasts. Perfect for strategic planning.',
    placement: 'right',
  },
  {
    target: '.validation-nav-item',
    title: '📊 Track Validation',
    content: 'Monitor model performance with real-time validation dashboards. Track accuracy metrics, detect drift, and trigger retraining when models need updating.',
    placement: 'right',
  },
];

// ============================================================================
// EMPTY STATE COMPONENTS
// ============================================================================

const NoPredictionsIllustration: FC = () => (
  <svg
    className="w-64 h-64 mx-auto mb-6"
    viewBox="0 0 400 300"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
  >
    {/* Background */}
    <rect width="400" height="300" fill="#f8fafc" rx="12" />

    {/* Chart Area */}
    <rect x="60" y="60" width="280" height="180" fill="white" stroke="#e2e8f0" rx="8" />

    {/* Grid Lines */}
    <line x1="60" y1="105" x2="340" y2="105" stroke="#f1f5f9" strokeWidth="2" />
    <line x1="60" y1="150" x2="340" y2="150" stroke="#f1f5f9" strokeWidth="2" />
    <line x1="60" y1="195" x2="340" y2="195" stroke="#f1f5f9" strokeWidth="2" />

    {/* Dashed Line (Missing Data) */}
    <line x1="80" y1="180" x2="320" y2="180" stroke="#cbd5e1" strokeWidth="3" strokeDasharray="8 8" />

    {/* Question Mark */}
    <circle cx="200" cy="120" r="30" fill="#fef3c7" />
    <text x="200" y="132" textAnchor="middle" fontSize="36" fill="#f59e0b" fontWeight="bold">
      ?
    </text>

    {/* Sparkle Icon */}
    <g transform="translate(320, 80)">
      <path
        d="M15 0L17 10L27 12L17 14L15 24L13 14L3 12L13 10L15 0Z"
        fill="#fbbf24"
        opacity="0.6"
      />
    </g>
  </svg>
);

const NoModelsIllustration: FC = () => (
  <svg
    className="w-64 h-64 mx-auto mb-6"
    viewBox="0 0 400 300"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
  >
    {/* Background */}
    <rect width="400" height="300" fill="#f8fafc" rx="12" />

    {/* Computer Screen */}
    <rect x="100" y="60" width="200" height="140" fill="#1e293b" rx="8" />
    <rect x="110" y="70" width="180" height="120" fill="#0f172a" rx="4" />

    {/* Code Lines */}
    <rect x="120" y="85" width="80" height="8" fill="#334155" rx="4" />
    <rect x="120" y="100" width="120" height="8" fill="#334155" rx="4" />
    <rect x="120" y="115" width="100" height="8" fill="#334155" rx="4" />
    <rect x="120" y="130" width="60" height="8" fill="#334155" rx="4" />

    {/* Upload Arrow */}
    <g transform="translate(180, 160)">
      <circle cx="20" cy="20" r="20" fill="#3b82f6" opacity="0.2" />
      <path
        d="M20 10V30M20 10L14 16M20 10L26 16"
        stroke="#3b82f6"
        strokeWidth="3"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </g>

    {/* Stand */}
    <rect x="180" y="200" width="40" height="20" fill="#64748b" rx="4" />
    <rect x="160" y="220" width="80" height="8" fill="#64748b" rx="4" />
  </svg>
);

const NoValidationsIllustration: FC = () => (
  <svg
    className="w-64 h-64 mx-auto mb-6"
    viewBox="0 0 400 300"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
  >
    {/* Background */}
    <rect width="400" height="300" fill="#f8fafc" rx="12" />

    {/* Calendar */}
    <rect x="100" y="80" width="200" height="160" fill="white" stroke="#e2e8f0" strokeWidth="3" rx="12" />

    {/* Calendar Header */}
    <rect x="100" y="80" width="200" height="50" fill="#3b82f6" rx="12" />
    <rect x="100" y="118" width="200" height="12" fill="#3b82f6" />
    <circle cx="140" cy="105" r="8" fill="white" opacity="0.3" />
    <circle cx="200" cy="105" r="8" fill="white" opacity="0.3" />
    <circle cx="260" cy="105" r="8" fill="white" opacity="0.3" />

    {/* Calendar Grid */}
    <rect x="130" y="150" width="30" height="30" fill="#fef3c7" rx="4" />
    <rect x="170" y="150" width="30" height="30" fill="#f1f5f9" rx="4" />
    <rect x="210" y="150" width="30" height="30" fill="#f1f5f9" rx="4" />
    <rect x="250" y="150" width="30" height="30" fill="#f1f5f9" rx="4" />

    <rect x="130" y="190" width="30" height="30" fill="#f1f5f9" rx="4" />
    <rect x="170" y="190" width="30" height="30" fill="#f1f5f9" rx="4" />
    <rect x="210" y="190" width="30" height="30" fill="#fef3c7" rx="4" />
    <rect x="250" y="190" width="30" height="30" fill="#f1f5f9" rx="4" />

    {/* Checkmark on first date */}
    <path
      d="M140 165L145 170L155 160"
      stroke="#f59e0b"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      fill="none"
    />
  </svg>
);

export const EmptyState: FC<EmptyStateProps> = ({ type, onAction, demoMode = false }) => {
  const content = {
    predictions: {
      illustration: <NoPredictionsIllustration />,
      title: 'No Predictions Yet',
      description:
        'Start by generating your first sales forecast. Select a retail category, choose a model, and get accurate predictions with confidence intervals.',
      actionText: 'Generate Your First Forecast',
      icon: <TrendingUp className="w-12 h-12 text-blue-500" />,
    },
    models: {
      illustration: <NoModelsIllustration />,
      title: 'No Models Trained',
      description:
        'Upload your historical retail sales data to train powerful ML models. Our system supports time series forecasting with LightGBM, XGBoost, and Random Forest.',
      actionText: 'Upload Training Data',
      icon: <Brain className="w-12 h-12 text-purple-500" />,
    },
    validations: {
      illustration: <NoValidationsIllustration />,
      title: 'No Validations Yet',
      description:
        'Compare your predictions against actual sales data to track model accuracy. Set up automatic validation to monitor model performance over time.',
      actionText: 'Validate Predictions',
      icon: <Calendar className="w-12 h-12 text-emerald-500" />,
    },
    counterfactuals: {
      illustration: <NoPredictionsIllustration />,
      title: 'No Counterfactuals Yet',
      description:
        'Explore "what-if" scenarios by adjusting economic indicators. See how changes in CPI, unemployment, or interest rates would affect your forecasts.',
      actionText: 'Create Counterfactual',
      icon: <Sparkles className="w-12 h-12 text-amber-500" />,
    },
    shap: {
      illustration: <NoPredictionsIllustration />,
      title: 'No SHAP Analysis Yet',
      description:
        'Generate a prediction first to view its SHAP (SHapley Additive exPlanations) values. Understand exactly which features influenced each prediction.',
      actionText: 'Generate Prediction',
      icon: <Brain className="w-12 h-12 text-purple-500" />,
    },
  };

  const current = content[type];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="flex flex-col items-center justify-center min-h-[500px] p-8"
    >
      <div className="glass-card p-12 rounded-2xl max-w-2xl w-full text-center">
        {demoMode && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="mb-6 inline-flex items-center gap-2 px-4 py-2 bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 rounded-full text-sm font-medium"
          >
            <Sparkles className="w-4 h-4" />
            Demo Mode Active
          </motion.div>
        )}

        {current.illustration}

        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="flex items-center justify-center gap-3 mb-4"
        >
          {current.icon}
          <h2 className="text-3xl font-bold text-slate-900 dark:text-white">
            {current.title}
          </h2>
        </motion.div>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="text-slate-600 dark:text-slate-300 mb-8 text-lg leading-relaxed"
        >
          {current.description}
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="flex flex-col sm:flex-row gap-4 justify-center"
        >
          {onAction && (
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={onAction}
              className="px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-xl font-semibold shadow-lg shadow-blue-500/50 hover:shadow-xl hover:shadow-blue-500/60 transition-all duration-200 flex items-center justify-center gap-2"
            >
              <Sparkles className="w-5 h-5" />
              {current.actionText}
            </motion.button>
          )}

          {type === 'predictions' && (
            <LoadSampleDataButton onLoad={() => onAction?.()} />
          )}
        </motion.div>

        {type === 'predictions' && (
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
            className="text-sm text-slate-500 dark:text-slate-400 mt-4"
          >
            or load sample data to explore the dashboard instantly
          </motion.p>
        )}
      </div>
    </motion.div>
  );
};

// ============================================================================
// DEMO MODE COMPONENTS
// ============================================================================

const samplePredictionData = {
  category: 'Total_Retail_Sales',
  model: 'LightGBM_Auto',
  date: '2024-12-01',
  predicted_value: 689432,
  confidence_interval: { lower: 675000, upper: 704000 },
  shap_values: [
    { feature: 'CPI', value: 0.23 },
    { feature: 'Unemployment', value: -0.15 },
    { feature: 'Interest Rate', value: 0.12 },
    { feature: 'Seasonality', value: 0.31 },
    { feature: 'Trend', value: 0.18 },
  ],
};

export const LoadSampleDataButton: FC<{ onLoad?: () => void }> = ({ onLoad }) => {
  const [isLoading, setIsLoading] = useState(false);

  const handleLoadSample = async () => {
    setIsLoading(true);

    // Simulate loading
    await new Promise(resolve => setTimeout(resolve, 1500));

    toast.success('Sample data loaded successfully!', {
      icon: <CheckCircle2 className="w-5 h-5 text-emerald-500" />,
      duration: 3000,
    });

    // Store in sessionStorage for demo mode
    sessionStorage.setItem('demoMode', 'true');
    sessionStorage.setItem('samplePrediction', JSON.stringify(samplePredictionData));

    setIsLoading(false);
    onLoad?.();
  };

  return (
    <motion.button
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      onClick={handleLoadSample}
      disabled={isLoading}
      className="px-8 py-4 bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-200 border-2 border-slate-300 dark:border-slate-600 rounded-xl font-semibold hover:border-blue-500 hover:text-blue-600 transition-all duration-200 flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
    >
      {isLoading ? (
        <>
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
            className="w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full"
          />
          Loading Sample Data...
        </>
      ) : (
        <>
          <Sparkles className="w-5 h-5 text-amber-500" />
          Load Sample Data
        </>
      )}
    </motion.button>
  );
};

export const DemoModeBanner: FC<DemoModeBannerProps> = ({ isActive, onExit }) => {
  const [isVisible, setIsVisible] = useState(isActive);

  useEffect(() => {
    setIsVisible(isActive);
  }, [isActive]);

  if (!isVisible) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="fixed top-20 left-1/2 transform -translate-x-1/2 z-50"
    >
      <div className="flex items-center gap-3 px-6 py-3 bg-gradient-to-r from-amber-500 to-orange-500 text-white rounded-full shadow-lg shadow-amber-500/50">
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 3, repeat: Infinity, ease: 'linear' }}
        >
          <Sparkles className="w-5 h-5" />
        </motion.div>
        <span className="font-semibold">Demo Mode - Using Sample Dataset</span>
        {onExit && (
          <button
            onClick={() => {
              onExit();
              setIsVisible(false);
              sessionStorage.removeItem('demoMode');
            }}
            className="ml-2 p-1 hover:bg-white/20 rounded-full transition-colors"
            aria-label="Exit demo mode"
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>
    </motion.div>
  );
};

// ============================================================================
// HELP TOOLTIP COMPONENT
// ============================================================================

export const HelpTooltip: FC<HelpTooltipProps> = ({ content, documentationLink, term }) => {
  const [isVisible, setIsVisible] = useState(false);

  return (
    <div className="relative inline-block ml-1">
      <motion.button
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.9 }}
        onMouseEnter={() => setIsVisible(true)}
        onMouseLeave={() => setIsVisible(false)}
        onFocus={() => setIsVisible(true)}
        onBlur={() => setIsVisible(false)}
        className="inline-flex items-center justify-center w-5 h-5 text-slate-400 hover:text-blue-500 transition-colors"
        aria-label={`Get help with ${term}`}
      >
        <HelpCircle className="w-4 h-4" />
      </motion.button>

      <AnimatePresence>
        {isVisible && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9, y: 5 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: 5 }}
            transition={{ duration: 0.15 }}
            className="absolute left-full ml-2 top-0 z-50 w-72 p-4 bg-slate-900 dark:bg-slate-800 text-white text-sm rounded-lg shadow-xl"
            role="tooltip"
          >
            <div className="flex items-start gap-2 mb-2">
              <Info className="w-4 h-4 text-blue-400 mt-0.5 flex-shrink-0" />
              <p className="font-semibold">{term}</p>
            </div>
            <p className="text-slate-300 mb-2">{content}</p>
            {documentationLink && (
              <a
                href={documentationLink}
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-400 hover:text-blue-300 text-xs font-medium flex items-center gap-1"
              >
                Learn more
                <ArrowRight className="w-3 h-3" />
              </a>
            )}
            {/* Arrow */}
            <div className="absolute left-0 top-3 transform -translate-x-full">
              <div className="w-0 h-0 border-t-8 border-t-transparent border-r-8 border-r-slate-900 border-b-8 border-b-transparent" />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

// ============================================================================
// ONBOARDING MODAL COMPONENT
// ============================================================================

export const OnboardingModal: FC<OnboardingProps> = ({
  isOpen,
  onClose,
  currentStep = 0,
  totalSteps = 4,
  title = 'Welcome to RetailPRED!',
  children,
}) => {
  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          onClick={(e) => e.stopPropagation()}
          className="glass-card max-w-2xl w-full p-8 rounded-2xl shadow-2xl relative"
        >
          {/* Close Button */}
          <button
            onClick={onClose}
            className="absolute top-4 right-4 p-2 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 transition-colors"
            aria-label="Close onboarding"
          >
            <X className="w-5 h-5" />
          </button>

          {/* Progress */}
          <div className="flex items-center gap-2 mb-6">
            {Array.from({ length: totalSteps }).map((_, i) => (
              <div
                key={i}
                className={`h-1 flex-1 rounded-full transition-all duration-300 ${
                  i < currentStep ? 'bg-blue-600' : i === currentStep ? 'bg-blue-400' : 'bg-slate-300 dark:bg-slate-600'
                }`}
              />
            ))}
          </div>

          {/* Content */}
          <div className="mb-8">
            <h2 className="text-3xl font-bold text-slate-900 dark:text-white mb-4">
              {title}
            </h2>
            <div className="text-slate-600 dark:text-slate-300">
              {children}
            </div>
          </div>

          {/* Navigation */}
          <div className="flex items-center justify-between">
            <button
              onClick={onClose}
              className="px-6 py-2 text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white font-medium transition-colors"
            >
              Skip Tour
            </button>

            <div className="flex gap-3">
              {currentStep > 0 && (
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className="px-6 py-2 bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-200 rounded-lg font-medium flex items-center gap-2"
                >
                  <ArrowLeft className="w-4 h-4" />
                  Previous
                </motion.button>
              )}

              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="px-6 py-2 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg font-medium shadow-lg shadow-blue-500/50 flex items-center gap-2"
              >
                {currentStep === totalSteps - 1 ? 'Get Started' : 'Next'}
                {currentStep < totalSteps - 1 && <ArrowRight className="w-4 h-4" />}
              </motion.button>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

// ============================================================================
// FIRST-TIME USER TOUR HOOK
// ============================================================================

export const useOnboardingTour = () => {
  const [isTourOpen, setIsTourOpen] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);

  useEffect(() => {
    // Check if user has seen the tour
    const hasSeenTour = localStorage.getItem('hasSeenOnboardingTour');

    if (!hasSeenTour) {
      // Small delay to let page load
      const timer = setTimeout(() => {
        setIsTourOpen(true);
      }, 1000);

      return () => clearTimeout(timer);
    }
  }, []);

  const closeTour = () => {
    setIsTourOpen(false);
    localStorage.setItem('hasSeenOnboardingTour', 'true');
  };

  const nextStep = () => {
    if (currentStep < TOUR_STEPS.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      closeTour();
    }
  };

  const prevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const restartTour = () => {
    localStorage.removeItem('hasSeenOnboardingTour');
    setCurrentStep(0);
    setIsTourOpen(true);
  };

  return {
    isTourOpen,
    currentStep,
    closeTour,
    nextStep,
    prevStep,
    restartTour,
    TOUR_STEPS,
  };
};

// ============================================================================
// WELCOME MODAL COMPONENT
// ============================================================================

export const WelcomeModal: FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);

  useEffect(() => {
    const hasSeenWelcome = localStorage.getItem('hasSeenWelcome');
    if (!hasSeenWelcome) {
      setIsOpen(true);
    }
  }, []);

  const steps = [
    {
      title: 'Welcome to RetailPRED! 🎯',
      content: (
        <div className="space-y-4">
          <p className="text-lg">
            Your intelligent <span className="font-bold text-blue-600">retail sales forecasting</span>{' '}
            platform powered by advanced machine learning models.
          </p>
          <div className="grid grid-cols-2 gap-4 mt-6">
            <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <TrendingUp className="w-8 h-8 text-blue-600 mb-2" />
              <p className="font-semibold">Accurate Forecasts</p>
              <p className="text-sm text-slate-600 dark:text-slate-400">
                Ensemble ML models
              </p>
            </div>
            <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
              <Brain className="w-8 h-8 text-purple-600 mb-2" />
              <p className="font-semibold">Explainable AI</p>
              <p className="text-sm text-slate-600 dark:text-slate-400">
                SHAP-based insights
              </p>
            </div>
          </div>
        </div>
      ),
    },
    {
      title: 'Generate Predictions 📊',
      content: (
        <div className="space-y-4">
          <p className="text-lg">
            Select a retail category and ML model to generate forecasts with confidence
            intervals.
          </p>
          <ul className="space-y-2 text-left">
            <li className="flex items-start gap-2">
              <CheckCircle2 className="w-5 h-5 text-emerald-500 flex-shrink-0 mt-0.5" />
              <span>Choose from 10+ retail categories</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle2 className="w-5 h-5 text-emerald-500 flex-shrink-0 mt-0.5" />
              <span>Compare LightGBM, XGBoost, Random Forest</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle2 className="w-5 h-5 text-emerald-500 flex-shrink-0 mt-0.5" />
              <span>Get 95% confidence intervals</span>
            </li>
          </ul>
        </div>
      ),
    },
    {
      title: 'Explore Counterfactuals 🔮',
      content: (
        <div className="space-y-4">
          <p className="text-lg">
            Run "what-if" scenarios to understand how economic changes affect your forecasts.
          </p>
          <div className="p-4 bg-amber-50 dark:bg-amber-900/20 rounded-lg">
            <p className="font-semibold mb-2">Example Scenarios:</p>
            <ul className="space-y-1 text-sm">
              <li>• What if unemployment rises by 1%?</li>
              <li>• How would a 0.25% rate cut impact sales?</li>
              <li>• Effect of CPI changes on different categories</li>
            </ul>
          </div>
        </div>
      ),
    },
    {
      title: 'Track Performance 📈',
      content: (
        <div className="space-y-4">
          <p className="text-lg">
            Monitor model accuracy with real-time validation dashboards and automatic drift
            detection.
          </p>
          <div className="flex items-center gap-4">
            <div className="flex-1 p-4 bg-emerald-50 dark:bg-emerald-900/20 rounded-lg text-center">
              <p className="text-3xl font-bold text-emerald-600">98.5%</p>
              <p className="text-sm text-slate-600 dark:text-slate-400">Average Accuracy</p>
            </div>
            <div className="flex-1 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg text-center">
              <p className="text-3xl font-bold text-blue-600">&lt;2%</p>
              <p className="text-sm text-slate-600 dark:text-slate-400">Mean Error Rate</p>
            </div>
          </div>
        </div>
      ),
    },
  ];

  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      setIsOpen(false);
      localStorage.setItem('hasSeenWelcome', 'true');
    }
  };

  const handleSkip = () => {
    setIsOpen(false);
    localStorage.setItem('hasSeenWelcome', 'true');
  };

  return (
    <OnboardingModal
      isOpen={isOpen}
      onClose={handleSkip}
      currentStep={currentStep}
      totalSteps={steps.length}
      title={steps[currentStep].title}
    >
      {steps[currentStep].content}

      <div className="flex justify-between mt-6">
        <button
          onClick={handleSkip}
          className="text-sm text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200"
        >
          Skip tour
        </button>

        <div className="flex gap-2">
          {steps.map((_, i) => (
            <button
              key={i}
              onClick={() => setCurrentStep(i)}
              className={`w-2 h-2 rounded-full transition-all ${
                i === currentStep ? 'bg-blue-600 w-8' : 'bg-slate-300 dark:bg-slate-600'
              }`}
            />
          ))}
        </div>

        <button
          onClick={handleNext}
          className="text-sm font-medium text-blue-600 hover:text-blue-700 dark:text-blue-400"
        >
          {currentStep === steps.length - 1 ? 'Get Started' : 'Next'} →
        </button>
      </div>
    </OnboardingModal>
  );
};

export default Onboarding;
