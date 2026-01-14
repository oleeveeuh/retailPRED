/**
 * Modern Header with Gradient Background
 *
 * Features:
 * - Gradient background (slate-900 to slate-800)
 * - Refresh Data button with loading spinner
 * - Model status indicator
 * - User avatar with dropdown
 * - Breadcrumb navigation
 * - Notification system
 */

import { FC, useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import {
  RefreshCw,
  Bell,
  Search,
  CheckCircle2,
  XCircle,
  AlertCircle,
  ChevronDown,
} from 'lucide-react';
import { dataApi } from '../../api/client';

interface BreadcrumbItem {
  name: string;
  path: string;
}

const getBreadcrumbs = (pathname: string): BreadcrumbItem[] => {
  const segments = pathname.split('/').filter(Boolean);
  const breadcrumbs: BreadcrumbItem[] = [
    { name: 'Home', path: '/dashboard/overview' },
  ];

  let currentPath = '';
  segments.forEach((segment, index) => {
    currentPath += `/${segment}`;
    if (index > 0) {
      // Skip 'dashboard' segment
      const name = segment.charAt(0).toUpperCase() + segment.slice(1);
      breadcrumbs.push({ name, path: currentPath });
    }
  });

  return breadcrumbs;
};

export const Header: FC = () => {
  const location = useLocation();
  const queryClient = useQueryClient();
  const [refreshStatus, setRefreshStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [modelStatus, setModelStatus] = useState<'loading' | 'loaded' | 'error'>('loading');
  const [showNotifications, setShowNotifications] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  // Check model status on mount
  useEffect(() => {
    // Simulate model status check - replace with actual API call
    const timer = setTimeout(() => {
      setModelStatus('loaded');
    }, 1000);
    return () => clearTimeout(timer);
  }, []);

  const refreshMutation = useMutation({
    mutationFn: dataApi.refresh,
    onMutate: () => {
      setRefreshStatus('loading');
    },
    onSuccess: (data) => {
      setRefreshStatus('success');
      queryClient.invalidateQueries();
      setTimeout(() => setRefreshStatus('idle'), 3000);
    },
    onError: () => {
      setRefreshStatus('error');
      setTimeout(() => setRefreshStatus('idle'), 3000);
    },
  });

  const handleRefresh = () => {
    refreshMutation.mutate();
  };

  const breadcrumbs = getBreadcrumbs(location.pathname);

  const notifications = [
    {
      id: 1,
      title: 'Model Training Complete',
      message: 'Total Retail Sales model finished training',
      type: 'success',
      time: '5 minutes ago',
    },
    {
      id: 2,
      title: 'Data Refreshed',
      message: 'FRED and MRTS data updated successfully',
      type: 'success',
      time: '1 hour ago',
    },
    {
      id: 3,
      title: 'Validation Alert',
      message: 'Gasoline Stations model needs attention',
      type: 'warning',
      time: '2 hours ago',
    },
  ];

  const getNotificationIcon = (type: string) => {
    switch (type) {
      case 'success':
        return <CheckCircle2 className="w-5 h-5 text-emerald-500" />;
      case 'warning':
        return <AlertCircle className="w-5 h-5 text-amber-500" />;
      case 'error':
        return <XCircle className="w-5 h-5 text-red-500" />;
      default:
        return <Bell className="w-5 h-5 text-primary" />;
    }
  };

  return (
    <header className="relative bg-gradient-to-r from-slate-900 via-slate-800 to-slate-900 border-b border-slate-700/50">
      {/* Animated Background Pattern */}
      <div className="absolute inset-0 opacity-5">
        <div className="absolute inset-0" style={{
          backgroundImage: `radial-gradient(circle at 2px 2px, white 1px, transparent 0)`,
          backgroundSize: '40px 40px',
        }} />
      </div>

      {/* Gradient Overlay */}
      <div className="absolute inset-0 bg-gradient-to-b from-slate-900/50 to-transparent" />

      <div className="relative">
        {/* Top Bar */}
        <div className="px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between gap-4">
            {/* Left: Breadcrumbs & Search */}
            <div className="flex items-center gap-4 flex-1 min-w-0">
              {/* Mobile Menu Button */}
              <button className="lg:hidden p-2 rounded-lg hover:bg-slate-800/50 text-slate-400 hover:text-white transition-colors">
                <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              </button>

              {/* Breadcrumbs */}
              <nav className="hidden md:flex items-center space-x-2 text-sm overflow-x-auto">
                {breadcrumbs.map((crumb, index) => (
                  <div key={`${crumb.path}-${index}`} className="flex items-center space-x-2">
                    {index > 0 && (
                      <ChevronDown className="w-4 h-4 text-slate-500 rotate-[-90deg]" />
                    )}
                    <Link
                      to={crumb.path}
                      className={`
                        font-medium transition-colors whitespace-nowrap
                        ${
                          index === breadcrumbs.length - 1
                            ? 'text-white'
                            : 'text-slate-400 hover:text-white'
                        }
                      `}
                    >
                      {crumb.name}
                    </Link>
                  </div>
                ))}
              </nav>
            </div>

            {/* Right: Actions */}
            <div className="flex items-center space-x-3">
              {/* Search Bar */}
              <div className="hidden sm:block relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
                <input
                  type="text"
                  placeholder="Search..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-64 pl-10 pr-4 py-2 bg-slate-800/50 border border-slate-700/50 rounded-lg text-sm text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                />
              </div>

              {/* Refresh Button */}
              <motion.button
                onClick={handleRefresh}
                disabled={refreshStatus === 'loading'}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className={`
                  relative px-4 py-2 rounded-lg font-medium text-sm
                  transition-all duration-200 flex items-center space-x-2
                  ${
                    refreshStatus === 'loading'
                      ? 'bg-amber-500 text-white cursor-wait'
                      : refreshStatus === 'success'
                      ? 'bg-emerald-500 text-white'
                      : refreshStatus === 'error'
                      ? 'bg-red-500 text-white hover:bg-red-600'
                      : 'bg-primary-600 text-white hover:bg-primary-700 shadow-lg shadow-blue-500/50'
                  }
                `}
              >
                <AnimatePresence mode="wait">
                  {refreshStatus === 'loading' ? (
                    <motion.div
                      key="loading"
                      initial={{ rotate: 0 }}
                      animate={{ rotate: 360 }}
                      exit={{ rotate: 0 }}
                      transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                    >
                      <RefreshCw className="w-4 h-4" />
                    </motion.div>
                  ) : refreshStatus === 'success' ? (
                    <motion.div
                      key="success"
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      exit={{ scale: 0 }}
                    >
                      <CheckCircle2 className="w-4 h-4" />
                    </motion.div>
                  ) : refreshStatus === 'error' ? (
                    <motion.div
                      key="error"
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      exit={{ scale: 0 }}
                    >
                      <XCircle className="w-4 h-4" />
                    </motion.div>
                  ) : (
                    <motion.div
                      key="idle"
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      exit={{ scale: 0 }}
                    >
                      <RefreshCw className="w-4 h-4" />
                    </motion.div>
                  )}
                </AnimatePresence>
                <span className="hidden sm:inline">
                  {refreshStatus === 'loading'
                    ? 'Refreshing...'
                    : refreshStatus === 'success'
                    ? 'Refreshed!'
                    : refreshStatus === 'error'
                    ? 'Error'
                    : 'Refresh Data'}
                </span>
              </motion.button>

              {/* Model Status Indicator */}
              <div className="hidden md:flex items-center space-x-2 px-3 py-2 bg-slate-800/50 rounded-lg border border-slate-700/50">
                <div className={`w-2 h-2 rounded-full ${
                  modelStatus === 'loaded' ? 'bg-emerald-500 animate-pulse' :
                  modelStatus === 'loading' ? 'bg-amber-500 animate-pulse' :
                  'bg-red-500'
                }`} />
                <span className="text-xs text-slate-300">
                  {modelStatus === 'loaded' ? 'Models Ready' :
                   modelStatus === 'loading' ? 'Loading...' :
                   'Error'}
                </span>
              </div>

              {/* Notifications */}
              <div className="relative">
                <motion.button
                  onClick={() => setShowNotifications(!showNotifications)}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="relative p-2 rounded-lg hover:bg-slate-800/50 text-slate-400 hover:text-white transition-colors"
                >
                  <Bell className="w-5 h-5" />
                  <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full animate-ping" />
                  <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full" />
                </motion.button>

                {/* Notifications Dropdown */}
                <AnimatePresence>
                  {showNotifications && (
                    <>
                      <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        onClick={() => setShowNotifications(false)}
                        className="fixed inset-0 z-40"
                      />
                      <motion.div
                        initial={{ opacity: 0, scale: 0.95, y: -10 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.95, y: -10 }}
                        transition={{ type: 'spring', damping: 25, stiffness: 300 }}
                        className="absolute right-0 mt-2 w-80 bg-slate-800 rounded-xl shadow-2xl border border-slate-700/50 z-50"
                      >
                        <div className="p-4 border-b border-slate-700/50">
                          <h3 className="text-lg font-semibold text-white">Notifications</h3>
                          <p className="text-sm text-slate-400 mt-1">{notifications.length} new notifications</p>
                        </div>
                        <div className="max-h-96 overflow-y-auto">
                          {notifications.map((notif) => (
                            <div
                              key={notif.id}
                              className="p-4 hover:bg-slate-700/50 transition-colors border-b border-slate-700/50 last:border-0"
                            >
                              <div className="flex items-start space-x-3">
                                {getNotificationIcon(notif.type)}
                                <div className="flex-1 min-w-0">
                                  <p className="text-sm font-medium text-white">{notif.title}</p>
                                  <p className="text-xs text-slate-400 mt-1">{notif.message}</p>
                                  <p className="text-xs text-slate-500 mt-1">{notif.time}</p>
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                        <div className="p-3 border-t border-slate-700/50">
                          <button className="w-full text-center text-sm text-blue-400 hover:text-blue-300 font-medium transition-colors">
                            View All Notifications
                          </button>
                        </div>
                      </motion.div>
                    </>
                  )}
                </AnimatePresence>
              </div>
            </div>
          </div>
        </div>

        {/* Refresh Status Banner */}
        <AnimatePresence>
          {refreshStatus === 'success' && refreshMutation.data && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ type: 'spring', damping: 25, stiffness: 300 }}
              className="px-4 sm:px-6 lg:px-8 pb-4"
            >
              <div className="bg-emerald-500/20 border border-emerald-500/50 rounded-xl p-4 backdrop-blur-sm">
                <div className="flex items-center space-x-3">
                  <CheckCircle2 className="w-5 h-5 text-emerald-400 flex-shrink-0" />
                  <div className="flex-1">
                    <p className="text-sm font-medium text-emerald-200">
                      ✓ {refreshMutation.data.message}
                    </p>
                    <p className="text-xs text-emerald-300/70 mt-1">
                      Updated {refreshMutation.data.records_updated} records from{' '}
                      {refreshMutation.data.sources_updated.join(', ')}
                    </p>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </header>
  );
};
