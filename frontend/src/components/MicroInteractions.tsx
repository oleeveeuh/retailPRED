import { motion, type FC, ReactNode, ButtonHTMLAttributes } from 'react';
import { Loader2 } from 'lucide-react';

// ============================================================================
// PREMIUM BUTTON WITH MICRO-INTERACTIONS
// ============================================================================

interface PremiumButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'success' | 'danger' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  isLoading?: boolean;
  children: ReactNode;
  className?: string;
}

export const PremiumButton: FC<PremiumButtonProps> = ({
  variant = 'primary',
  size = 'md',
  isLoading = false,
  children,
  className = '',
  disabled,
  ...props
}) => {
  const variants = {
    primary: 'bg-gradient-to-r from-blue-600 to-accent text-white shadow-lg shadow-blue-500/50 hover:shadow-xl hover:shadow-blue-500/60',
    secondary: 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 border-2 border-slate-300 dark:border-slate-600 hover:bg-slate-200 dark:hover:bg-slate-700',
    success: 'bg-gradient-to-r from-emerald-500 to-teal-500 text-white shadow-lg shadow-emerald-500/50 hover:shadow-xl hover:shadow-emerald-500/60',
    danger: 'bg-gradient-to-r from-red-500 to-rose-500 text-white shadow-lg shadow-red-500/50 hover:shadow-xl hover:shadow-red-500/60',
    ghost: 'bg-transparent text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800',
  };

  const sizes = {
    sm: 'px-4 py-2 text-sm',
    md: 'px-6 py-3 text-base',
    lg: 'px-8 py-4 text-lg',
  };

  return (
    <motion.button
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      disabled={disabled || isLoading}
      className={`rounded-lg font-semibold transition-all duration-200 flex items-center justify-center gap-2 ${variants[variant]} ${sizes[size]} ${className} ${
        (disabled || isLoading) && 'opacity-50 cursor-not-allowed'
      }`}
      {...props}
    >
      {isLoading ? (
        <>
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
            className="w-5 h-5 border-2 border-current border-t-transparent rounded-full"
          />
          Loading...
        </>
      ) : (
        children
      )}
    </motion.button>
  );
};

// ============================================================================
// PREMIUM CARD WITH LIFT EFFECT
// ============================================================================

interface PremiumCardProps {
  children: ReactNode;
  className?: string;
  hover?: boolean;
  onClick?: () => void;
}

export const PremiumCard: FC<PremiumCardProps> = ({
  children,
  className = '',
  hover = true,
  onClick,
}) => {
  const Component = onClick ? motion.button : motion.div;

  return (
    <Component
      whileHover={hover ? { y: -4, transition: { duration: 0.2 } } : undefined}
      whileTap={onClick ? { scale: 0.98 } : undefined}
      onClick={onClick}
      className={`bg-white dark:bg-slate-800 rounded-2xl shadow-lg hover:shadow-2xl transition-shadow duration-200 ${className}`}
    >
      {children}
    </Component>
  );
};

// ============================================================================
// PREMIUM INPUT WITH GLOW EFFECT
// ============================================================================

interface PremiumInputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  icon?: ReactNode;
}

export const PremiumInput: FC<PremiumInputProps> = ({
  label,
  error,
  icon,
  className = '',
  ...props
}) => (
  <div className={`relative ${className}`}>
    {label && (
      <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
        {label}
      </label>
    )}

    <div className="relative">
      {icon && (
        <div className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400">
          {icon}
        </div>
      )}

      <motion.input
        whileFocus={{
          scale: 1.01,
          boxShadow: '0 0 0 3px rgba(37, 99, 235, 0.1)',
        }}
        transition={{ duration: 0.2 }}
        className={`w-full px-4 py-3 bg-white dark:bg-slate-900 border-2 rounded-lg transition-all duration-200 outline-none ${
          error
            ? 'border-red-500 focus:border-red-600'
            : 'border-slate-300 dark:border-slate-600 focus:border-primary'
        } ${icon ? 'pl-10' : ''} text-slate-900 dark:text-white placeholder:text-slate-400`}
        {...props}
      />
    </div>

    {error && (
      <motion.p
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="mt-1 text-sm text-red-500 dark:text-red-400"
      >
        {error}
      </motion.p>
    )}
  </div>
);

// ============================================================================
// PREMIUM SELECT WITH GLOW EFFECT
// ============================================================================

interface PremiumSelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
  label?: string;
  error?: string;
  options: { value: string; label: string }[];
}

export const PremiumSelect: FC<PremiumSelectProps> = ({
  label,
  error,
  options,
  className = '',
  ...props
}) => (
  <div className={`relative ${className}`}>
    {label && (
      <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
        {label}
      </label>
    )}

    <motion.select
      whileFocus={{
        scale: 1.01,
        boxShadow: '0 0 0 3px rgba(37, 99, 235, 0.1)',
      }}
      transition={{ duration: 0.2 }}
      className={`w-full px-4 py-3 bg-white dark:bg-slate-900 border-2 rounded-lg transition-all duration-200 outline-none appearance-none cursor-pointer ${
        error
          ? 'border-red-500 focus:border-red-600'
          : 'border-slate-300 dark:border-slate-600 focus:border-primary'
      } text-slate-900 dark:text-white`}
      {...props}
    >
      {options.map((option) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </motion.select>

    {/* Custom arrow icon */}
    <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none text-slate-400">
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
      </svg>
    </div>

    {error && (
      <motion.p
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="mt-1 text-sm text-red-500 dark:text-red-400"
      >
        {error}
      </motion.p>
    )}
  </div>
);

// ============================================================================
// RIPPLE EFFECT BUTTON
// ============================================================================

interface RippleButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  children: ReactNode;
  className?: string;
}

export const RippleButton: FC<RippleButtonProps> = ({ children, className = '', ...props }) => {
  return (
    <motion.button
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      className={`relative overflow-hidden rounded-lg ${className}`}
      {...props}
    >
      {children}
    </motion.button>
  );
};

// ============================================================================
// TOGGLE SWITCH WITH ANIMATION
// ============================================================================

interface ToggleSwitchProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label?: string;
  disabled?: boolean;
}

export const ToggleSwitch: FC<ToggleSwitchProps> = ({
  checked,
  onChange,
  label,
  disabled = false,
}) => (
  <button
    type="button"
    onClick={() => !disabled && onChange(!checked)}
    disabled={disabled}
    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors duration-200 ${
      checked ? 'bg-primary-600' : 'bg-slate-300 dark:bg-slate-600'
    } ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
    role="switch"
    aria-checked={checked}
  >
    <motion.span
      animate={{ x: checked ? 24 : 4 }}
      transition={{ type: 'spring', stiffness: 500, damping: 30 }}
      className="inline-block h-4 w-4 transform rounded-full bg-white shadow-lg"
    />
    {label && (
      <span className="sr-only">{label}</span>
    )}
  </button>
);

// ============================================================================
// BADGE WITH PULSE ANIMATION
// ============================================================================

interface BadgeProps {
  children: ReactNode;
  variant?: 'default' | 'success' | 'warning' | 'danger';
  pulse?: boolean;
  className?: string;
}

export const Badge: FC<BadgeProps> = ({
  children,
  variant = 'default',
  pulse = false,
  className = '',
}) => {
  const variants = {
    default: 'bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300',
    success: 'bg-emerald-100 dark:bg-emerald-900/20 text-emerald-700 dark:text-emerald-300',
    warning: 'bg-amber-100 dark:bg-amber-900/20 text-amber-700 dark:text-amber-300',
    danger: 'bg-red-100 dark:bg-red-900/20 text-red-700 dark:text-red-300',
  };

  return (
    <motion.span
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${variants[variant]} ${className}`}
    >
      {pulse && (
        <motion.span
          animate={{ scale: [1, 1.2, 1] }}
          transition={{ duration: 2, repeat: Infinity }}
          className="w-2 h-2 rounded-full bg-current mr-2"
        />
      )}
      {children}
    </motion.span>
  );
};

// ============================================================================
// STATS CARD WITH COUNTER ANIMATION
// ============================================================================

interface StatsCardProps {
  title: string;
  value: number;
  icon?: ReactNode;
  trend?: 'up' | 'down' | 'neutral';
  trendValue?: string;
  className?: string;
}

export const StatsCard: FC<StatsCardProps> = ({
  title,
  value,
  icon,
  trend = 'neutral',
  trendValue,
  className = '',
}) => {
  const TrendIcon = trend === 'up' ? '↑' : trend === 'down' ? '↓' : '→';
  const trendColor = trend === 'up' ? 'text-emerald-500' : trend === 'down' ? 'text-red-500' : 'text-slate-500';

  return (
    <PremiumCard className={`p-6 ${className}`}>
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="text-sm font-medium text-slate-600 dark:text-slate-400 mb-1">
            {title}
          </p>

          <motion.h3
            key={value}
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="text-3xl font-bold text-slate-900 dark:text-white mb-1"
          >
            {value.toLocaleString()}
          </motion.h3>

          {trendValue && (
            <p className={`text-sm font-medium ${trendColor}`}>
              {TrendIcon} {trendValue}
            </p>
          )}
        </div>

        {icon && (
          <div className="p-3 bg-blue-100 dark:bg-blue-900/20 rounded-lg text-primary-600 dark:text-blue-400">
            {icon}
          </div>
        )}
      </div>
    </PremiumCard>
  );
};

// ============================================================================
// PROGRESS DOTS
// ============================================================================

interface ProgressDotsProps {
  current: number;
  total: number;
  className?: string;
}

export const ProgressDots: FC<ProgressDotsProps> = ({ current, total, className = '' }) => (
  <div className={`flex items-center gap-2 ${className}`}>
    {Array.from({ length: total }).map((_, i) => (
      <motion.div
        key={i}
        initial={{ scale: 0.8 }}
        animate={{ scale: 1 }}
        className={`h-2 rounded-full transition-all duration-300 ${
          i < current ? 'bg-primary-600 flex-1' : i === current ? 'bg-blue-400 w-8' : 'bg-slate-300 dark:bg-slate-600 w-2'
        }`}
      />
    ))}
  </div>
);

export default PremiumButton;
