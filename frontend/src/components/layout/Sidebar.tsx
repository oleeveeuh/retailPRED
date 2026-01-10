/**
 * Modern Sidebar Navigation with Frosted Glass Effect
 *
 * Features:
 * - Frosted glass (backdrop-blur) effect
 * - Collapsible with smooth animation
 * - Active state with gradient accent
 * - Dark mode support
 * - Lucide React icons
 */

import { FC, useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Home,
  BarChart3,
  Bot,
  CheckCircle,
  Lightbulb,
  AlertTriangle,
  TrendingUp,
  Sliders,
  ChevronLeft,
  ChevronRight,
  Briefcase,
} from 'lucide-react';

interface NavItem {
  name: string;
  path: string;
  icon: React.ComponentType<{ className?: string }>;
  description: string;
}

const navItems: NavItem[] = [
  {
    name: 'Overview',
    path: '/dashboard/overview',
    icon: Home,
    description: 'Dashboard overview and key metrics',
  },
  {
    name: 'Forecasts',
    path: '/dashboard/predictions',
    icon: BarChart3,
    description: 'Retail sales forecasts',
  },
  {
    name: 'Models',
    path: '/dashboard/models',
    icon: Bot,
    description: 'Model training and management',
  },
  {
    name: 'Validation',
    path: '/dashboard/validation',
    icon: CheckCircle,
    description: 'Model validation and metrics',
  },
  {
    name: 'Model Explainability',
    path: '/dashboard/explain',
    icon: Lightbulb,
    description: 'SHAP values and feature importance',
  },
  {
    name: 'Anomaly Detection',
    path: '/dashboard/anomalies',
    icon: AlertTriangle,
    description: 'Unusual predictions explained',
  },
  {
    name: 'Economic Scenarios',
    path: '/dashboard/scenarios',
    icon: TrendingUp,
    description: 'Recession, recovery, and stress testing',
  },
  {
    name: 'Sensitivity',
    path: '/dashboard/sensitivity',
    icon: Sliders,
    description: 'Economic factor sensitivity analysis',
  },
  {
    name: 'Business View',
    path: '/dashboard/business',
    icon: Briefcase,
    description: 'Executive summary and Tableau',
  },
];

export const Sidebar: FC = () => {
  const location = useLocation();
  const [isCollapsed, setIsCollapsed] = useState(false);
  const toggleCollapse = () => setIsCollapsed(!isCollapsed);

  const isActive = (path: string) => location.pathname === path;

  return (
    <>
      {/* Mobile Overlay */}
      <AnimatePresence>
        {!isCollapsed && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setIsCollapsed(true)}
            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40 lg:hidden"
          />
        )}
      </AnimatePresence>

      {/* Sidebar */}
      <motion.aside
        initial={false}
        animate={{
          width: isCollapsed ? '80px' : '280px',
        }}
        transition={{
          type: 'spring',
          damping: 25,
          stiffness: 200,
        }}
        className={`
          fixed lg:relative left-0 top-0 h-full z-50
          flex flex-col
          bg-gradient-to-b from-slate-900 to-slate-800
          backdrop-blur-xl backdrop-saturate-150
          border-r border-slate-700/50
          shadow-2xl
        `}
      >
        {/* Logo Section */}
        <div className="relative h-20 border-b border-slate-700/50">
          <div className="absolute inset-0 bg-gradient-to-r from-blue-600/20 to-purple-600/20" />

          <div className="relative h-full flex items-center justify-between px-6">
            <AnimatePresence mode="wait">
              {!isCollapsed && (
                <motion.div
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ duration: 0.2 }}
                  className="flex items-center space-x-3"
                >
                  <div className="relative">
                    <div className="absolute inset-0 bg-blue-500 blur-xl opacity-50 rounded-full animate-pulse" />
                    <div className="relative w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center shadow-lg">
                      <BarChart3 className="w-6 h-6 text-white" />
                    </div>
                  </div>
                  <div>
                    <h1 className="text-xl font-bold text-white">RetailPRED</h1>
                    <p className="text-xs text-slate-400">AI Forecasting</p>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Collapse Toggle */}
            <motion.button
              onClick={toggleCollapse}
              className="p-2 rounded-lg hover:bg-slate-700/50 text-slate-400 hover:text-white transition-colors"
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              aria-label="Toggle sidebar"
            >
              {isCollapsed ? (
                <ChevronRight className="w-5 h-5" />
              ) : (
                <ChevronLeft className="w-5 h-5" />
              )}
            </motion.button>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 overflow-y-auto py-6 px-3 space-y-2 custom-scrollbar">
          {navItems.map((item, index) => {
            const Icon = item.icon;
            const active = isActive(item.path);

            return (
              <motion.div
                key={item.path}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.05 }}
              >
                <Link
                  to={item.path}
                  className={`
                    relative group flex items-center px-4 py-3
                    rounded-xl transition-all duration-200
                    ${
                      active
                        ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg shadow-blue-500/50'
                        : 'text-slate-400 hover:bg-slate-800/50 hover:text-white'
                    }
                  `}
                >
                  {/* Active Indicator */}
                  {active && (
                    <motion.div
                      layoutId="activeIndicator"
                      className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-8 bg-white rounded-r-full"
                      transition={{ type: 'spring', stiffness: 300, damping: 30 }}
                    />
                  )}

                  {/* Icon */}
                  <motion.div
                    whileHover={{ scale: 1.1 }}
                    transition={{ type: 'spring', stiffness: 400, damping: 17 }}
                  >
                    <Icon className={`
                      w-5 h-5 flex-shrink-0
                      ${active ? 'text-white' : ''}
                      ${isCollapsed ? 'mx-auto' : 'mr-3'}
                    `} />
                  </motion.div>

                  {/* Text */}
                  <AnimatePresence mode="wait">
                    {!isCollapsed && (
                      <motion.div
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: -10 }}
                        transition={{ duration: 0.2 }}
                        className="flex-1 min-w-0"
                      >
                        <div className="font-semibold truncate">{item.name}</div>
                        {!active && (
                          <div className="text-xs text-slate-500 truncate">{item.description}</div>
                        )}
                      </motion.div>
                    )}
                  </AnimatePresence>

                  {/* Hover Glow */}
                  {!active && (
                    <motion.div
                      className="absolute inset-0 bg-gradient-to-r from-blue-600/10 to-purple-600/10 rounded-xl opacity-0 group-hover:opacity-100 transition-opacity"
                      initial={false}
                      whileHover={{ opacity: 1 }}
                    />
                  )}
                </Link>
              </motion.div>
            );
          })}
        </nav>

        {/* Mobile Close Button */}
        <motion.button
          onClick={() => setIsCollapsed(true)}
          className="lg:hidden absolute top-4 right-4 p-2 rounded-lg hover:bg-slate-700/50 text-slate-400 hover:text-white"
          whileTap={{ scale: 0.9 }}
        >
          <ChevronLeft className="w-6 h-6" />
        </motion.button>
      </motion.aside>
    </>
  );
};
