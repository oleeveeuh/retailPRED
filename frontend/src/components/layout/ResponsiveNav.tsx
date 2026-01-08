import { useState, useEffect, FC } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Menu,
  X,
  TrendingUp,
  Brain,
  Bot,
  Calendar,
  Home,
  ChevronRight,
  Sliders,
  Briefcase,
} from 'lucide-react';

interface NavItem {
  path: string;
  label: string;
  icon: FC<{ className?: string }>;
  className: string;
  badge?: string;
}

const navItems: NavItem[] = [
  { path: '/dashboard/overview', label: 'Overview', icon: Home, className: 'overview-nav-item' },
  { path: '/dashboard/predictions', label: 'Predictions', icon: TrendingUp, className: 'predict-nav-item' },
  { path: '/dashboard/models', label: 'Models', icon: Bot, className: 'models-nav-item' },
  { path: '/dashboard/validation', label: 'Validation', icon: Calendar, className: 'validation-nav-item' },
  { path: '/dashboard/explain', label: 'Model Explainability', icon: Brain, className: 'explain-nav-item' },
  { path: '/dashboard/scenarios', label: 'Economic Scenarios', icon: TrendingUp, className: 'scenarios-nav-item' },
  { path: '/dashboard/sensitivity', label: 'Sensitivity', icon: Sliders, className: 'sensitivity-nav-item' },
  { path: '/dashboard/business', label: 'Business View', icon: Briefcase, className: 'business-nav-item' },
];

export const ResponsiveNav: FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);
  const location = useLocation();

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 10);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Close mobile menu on route change
  useEffect(() => {
    setIsOpen(false);
  }, [location.pathname]);

  // Prevent body scroll when menu is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }

    return () => {
      document.body.style.overflow = '';
    };
  }, [isOpen]);

  return (
    <>
      {/* Desktop Navigation */}
      <nav
        className={`hidden md:block fixed top-0 left-0 right-0 z-40 transition-all duration-200 ${
          isScrolled
            ? 'bg-white/80 dark:bg-slate-800/80 backdrop-blur-xl shadow-lg border-b border-slate-200/50 dark:border-slate-700/50'
            : 'bg-transparent'
        }`}
      >
        <div className="container mx-auto px-4">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <Link
              to="/dashboard/overview"
              className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent hover:opacity-80 transition-opacity"
            >
              RetailPRED
            </Link>

            {/* Navigation Links */}
            <div className="flex items-center gap-1">
              {navItems.map((item) => {
                const Icon = item.icon;
                const isActive = location.pathname === item.path;

                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    className={`
                      relative px-4 py-2 rounded-lg font-medium transition-all duration-200 flex items-center gap-2
                      ${isActive
                        ? 'bg-blue-100 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400'
                        : 'text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700/50'
                      }
                    `}
                  >
                    <Icon className="w-5 h-5" />
                    <span>{item.label}</span>

                    {/* Active indicator */}
                    {isActive && (
                      <motion.div
                        layoutId="activeNav"
                        className="absolute bottom-0 left-0 right-0 h-0.5 bg-blue-600"
                        transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                      />
                    )}

                    {item.badge && (
                      <span className="ml-1 px-2 py-0.5 bg-blue-600 text-white text-xs rounded-full">
                        {item.badge}
                      </span>
                    )}
                  </Link>
                );
              })}
            </div>
          </div>
        </div>
      </nav>

      {/* Mobile Header */}
      <header
        className={`md:hidden fixed top-0 left-0 right-0 z-40 transition-all duration-200 ${
          isScrolled
            ? 'bg-white/80 dark:bg-slate-800/80 backdrop-blur-xl shadow-lg border-b border-slate-200/50 dark:border-slate-700/50'
            : 'bg-white dark:bg-slate-800'
        }`}
      >
        <div className="flex items-center justify-between h-16 px-4">
          {/* Logo */}
          <Link
            to="/dashboard/overview"
            className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent"
          >
            RetailPRED
          </Link>

          {/* Hamburger Menu Button */}
          <motion.button
            whileTap={{ scale: 0.9 }}
            onClick={() => setIsOpen(!isOpen)}
            className="p-2 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors"
            aria-label="Toggle menu"
            aria-expanded={isOpen}
          >
            {isOpen ? (
              <X className="w-6 h-6 text-slate-600 dark:text-slate-300" />
            ) : (
              <Menu className="w-6 h-6 text-slate-600 dark:text-slate-300" />
            )}
          </motion.button>
        </div>
      </header>

      {/* Mobile Menu Overlay */}
      <AnimatePresence>
        {isOpen && (
          <>
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setIsOpen(false)}
              className="md:hidden fixed inset-0 bg-black/50 backdrop-blur-sm z-40"
            />

            {/* Mobile Menu */}
            <motion.div
              initial={{ x: '100%' }}
              animate={{ x: 0 }}
              exit={{ x: '100%' }}
              transition={{ type: 'spring', damping: 30, stiffness: 300 }}
              className="md:hidden fixed top-0 right-0 bottom-0 w-72 bg-white dark:bg-slate-800 shadow-2xl z-50 overflow-y-auto"
            >
              {/* Menu Header */}
              <div className="flex items-center justify-between h-16 px-4 border-b border-slate-200 dark:border-slate-700">
                <span className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  Menu
                </span>
                <motion.button
                  whileTap={{ scale: 0.9 }}
                  onClick={() => setIsOpen(false)}
                  className="p-2 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors"
                  aria-label="Close menu"
                >
                  <X className="w-6 h-6 text-slate-600 dark:text-slate-300" />
                </motion.button>
              </div>

              {/* Navigation Links */}
              <nav className="p-4">
                <div className="space-y-1">
                  {navItems.map((item, index) => {
                    const Icon = item.icon;
                    const isActive = location.pathname === item.path;

                    return (
                      <motion.div
                        key={item.path}
                        initial={{ x: 20, opacity: 0 }}
                        animate={{ x: 0, opacity: 1 }}
                        transition={{ delay: index * 0.05 }}
                      >
                        <Link
                          to={item.path}
                          className={`
                            relative flex items-center gap-3 px-4 py-3 rounded-lg font-medium transition-all duration-200
                            ${isActive
                              ? 'bg-blue-100 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400'
                              : 'text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700/50'
                            }
                          `}
                        >
                          <Icon className="w-5 h-5 flex-shrink-0" />
                          <span className="flex-1">{item.label}</span>

                          {isActive && (
                            <ChevronRight className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                          )}

                          {item.badge && (
                            <span className="px-2 py-0.5 bg-blue-600 text-white text-xs rounded-full">
                              {item.badge}
                            </span>
                          )}
                        </Link>
                      </motion.div>
                    );
                  })}
                </div>

                {/* Menu Footer */}
                <div className="mt-8 pt-8 border-t border-slate-200 dark:border-slate-700">
                  <div className="text-center text-sm text-slate-500 dark:text-slate-400">
                    <p className="mb-2">RetailPRED v1.0</p>
                    <p>© 2024 All rights reserved</p>
                  </div>
                </div>
              </nav>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </>
  );
};

export default ResponsiveNav;
