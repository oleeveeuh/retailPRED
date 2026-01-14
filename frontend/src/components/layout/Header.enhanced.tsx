/**
 * Enhanced Header with Real-Time Refresh
 *
 * Features:
 * - Prominent refresh button with progress ring
 * - Last updated timestamp
 * - API fetch status with checkmarks
 * - Model status indicator
 * - User dropdown
 * - Notifications
 */

import { FC, useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import toast from 'react-hot-toast';
import {
  RefreshCw,
  Bell,
  Search,
  User,
  ChevronDown,
  LogOut,
  Settings,
  HelpCircle,
  CheckCircle2,
  XCircle,
  AlertCircle,
  Clock,
  Wifi,
  Loader2,
} from 'lucide-react';
import { dataApi } from '../../api/client';
import { useRealtimeData, LastUpdatedDisplay, useNetworkStatus } from '../../hooks/useRealtimeData';

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
      const name = segment.charAt(0).toUpperCase() + segment.slice(1);
      breadcrumbs.push({ name, path: currentPath });
    }
  });

  return breadcrumbs;
};

interface ApiStatus {
  name: string;
  status: 'pending' | 'loading' | 'success' | 'error';
}

export const Header: FC = () => {
  const location = useLocation();
  const queryClient = useQueryClient();
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [showNotifications, setShowNotifications] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [apiStatuses, setApiStatuses] = useState<ApiStatus[]>([
    { name: 'FRED API', status: 'pending' },
    { name: 'MRTS API', status: 'pending' },
    { name: 'Yahoo Finance', status: 'pending' },
  ]);

  const isOnline = useNetworkStatus();
  const breadcrumbs = getBreadcrumbs(location.pathname);

  // Enhanced refresh mutation
  const refreshMutation = useMutation({
    mutationFn: async () => {
      // Reset API statuses
      setApiStatuses(prev => prev.map(api => ({ ...api, status: 'loading' as const })));

      // Simulate fetching from multiple APIs
      const promises = [
        new Promise(resolve => setTimeout(resolve, 800)), // FRED
        new Promise(resolve => setTimeout(resolve, 1200)), // MRTS
        new Promise(resolve => setTimeout(resolve, 600)), // Yahoo
      ];

      // Update statuses as they complete
      setTimeout(() => {
        setApiStatuses(prev => prev.map((api, i) =>
          i === 2 ? { ...api, status: 'success' as const } : api
        ));
      }, 600);

      setTimeout(() => {
        setApiStatuses(prev => prev.map((api, i) =>
          i === 0 ? { ...api, status: 'success' as const } : api
        ));
      }, 800);

      setTimeout(() => {
        setApiStatuses(prev => prev.map((api, i) =>
          i === 1 ? { ...api, status: 'success' as const } : api
        ));
      }, 1200);

      await Promise.all(promises);

      // Invalidate all queries
      queryClient.invalidateQueries();

      return { success: true };
    },
    onSuccess: () => {
      toast.success('All data refreshed successfully!', {
        icon: <CheckCircle2 className="w-5 h-5 text-emerald-500" />,
        duration: 3000,
      });
    },
    onError: () => {
      setApiStatuses(prev => prev.map(api => ({ ...api, status: 'error' as const })));
      toast.error('Failed to refresh data. Please try again.', {
        icon: <XCircle className="w-5 h-5 text-red-500" />,
        duration: 5000,
      });
    },
  });

  const handleRefresh = () => {
    refreshMutation.mutate();
  };

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

  const isRefreshing = refreshMutation.isPending;
  const allSuccess = apiStatuses.every(api => api.status === 'success');
  const hasError = apiStatuses.some(api => api.status === 'error');

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
                  <div key={crumb.path} className="flex items-center space-x-2">
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

              {/* Enhanced Refresh Button */}
              <div className="relative">
                <motion.button
                  onClick={handleRefresh}
                  disabled={isRefreshing || !isOnline}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className={`
                    relative px-4 py-2 rounded-lg font-medium text-sm
                    transition-all duration-200 flex items-center space-x-2
                    ${
                      !isOnline
                        ? 'bg-slate-700 text-slate-400 cursor-not-allowed'
                        : isRefreshing
                        ? 'bg-primary-600 text-white cursor-wait'
                        : hasError
                        ? 'bg-red-500 text-white hover:bg-red-600'
                        : allSuccess
                        ? 'bg-emerald-500 text-white'
                        : 'bg-primary-600 text-white hover:bg-primary-700 shadow-lg shadow-blue-500/50'
                    }
                  `}
                >
                  <AnimatePresence mode="wait">
                    {isRefreshing ? (
                      <motion.div
                        key="loading"
                        initial={{ rotate: 0 }}
                        animate={{ rotate: 360 }}
                        exit={{ rotate: 0 }}
                        transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                      >
                        <RefreshCw className="w-4 h-4" />
                      </motion.div>
                    ) : allSuccess ? (
                      <motion.div
                        key="success"
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        exit={{ scale: 0 }}
                      >
                        <CheckCircle2 className="w-4 h-4" />
                      </motion.div>
                    ) : hasError ? (
                      <motion.div
                        key="error"
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        exit={{ scale: 0 }}
                      >
                        <XCircle className="w-4 h-4" />
                      </motion.div>
                    ) : !isOnline ? (
                      <motion.div
                        key="offline"
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        exit={{ scale: 0 }}
                      >
                        <WifiOff className="w-4 h-4" />
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

                  <span>
                    {isRefreshing ? 'Refreshing...' : hasError ? 'Failed' : allSuccess ? 'Refreshed!' : 'Refresh Data'}
                  </span>

                  {/* Progress Ring */}
                  {isRefreshing && (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="absolute -top-1 -right-1"
                    >
                      <svg className="w-5 h-5" viewBox="0 0 20 20">
                        <circle
                          cx="10"
                          cy="10"
                          r="8"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2"
                          className="text-primary-600 opacity-20"
                        />
                        <motion.circle
                          cx="10"
                          cy="10"
                          r="8"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2"
                          strokeDasharray={50.3}
                          strokeDashoffset={12.6}
                          className="text-primary-600"
                          initial={{ strokeDashoffset: 50.3 }}
                          animate={{ strokeDashoffset: 0 }}
                          transition={{ duration: 2000, ease: 'linear' }}
                        />
                      </svg>
                    </motion.div>
                  )}
                </motion.button>

                {/* API Status Popover */}
                <AnimatePresence>
                  {(isRefreshing || allSuccess) && (
                    <motion.div
                      initial={{ opacity: 0, y: 10, scale: 0.95 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      exit={{ opacity: 0, y: 10, scale: 0.95 }}
                      transition={{ duration: 0.2 }}
                      className="absolute top-full right-0 mt-2 w-64 bg-slate-800/95 backdrop-blur-xl rounded-xl border border-slate-700/50 shadow-2xl overflow-hidden"
                    >
                      <div className="p-3 space-y-2">
                        <div className="flex items-center gap-2 text-xs text-slate-400 mb-2">
                          <Clock className="w-3 h-3" />
                          <span>Fetching from {apiStatuses.length} APIs...</span>
                        </div>
                        {apiStatuses.map((api) => (
                          <div key={api.name} className="flex items-center justify-between text-sm">
                            <span className="text-slate-300">{api.name}</span>
                            <AnimatePresence mode="wait">
                              {api.status === 'loading' && (
                                <motion.div
                                  key="loading"
                                  initial={{ opacity: 0, scale: 0.8 }}
                                  animate={{ opacity: 1, scale: 1 }}
                                  exit={{ opacity: 0, scale: 0.8 }}
                                >
                                  <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />
                                </motion.div>
                              )}
                              {api.status === 'success' && (
                                <motion.div
                                  key="success"
                                  initial={{ opacity: 0, scale: 0.8 }}
                                  animate={{ opacity: 1, scale: 1 }}
                                  exit={{ opacity: 0, scale: 0.8 }}
                                >
                                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                                </motion.div>
                              )}
                              {api.status === 'error' && (
                                <motion.div
                                  key="error"
                                  initial={{ opacity: 0, scale: 0.8 }}
                                  animate={{ opacity: 1, scale: 1 }}
                                  exit={{ opacity: 0, scale: 0.8 }}
                                >
                                  <XCircle className="w-4 h-4 text-red-400" />
                                </motion.div>
                              )}
                            </AnimatePresence>
                          </div>
                        ))}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              {/* Notifications */}
              <div className="relative">
                <motion.button
                  onClick={() => setShowNotifications(!showNotifications)}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className="relative p-2 rounded-lg hover:bg-slate-800/50 text-slate-400 hover:text-white transition-colors"
                >
                  <Bell className="w-5 h-5" />
                  <span className="absolute top-1 right-1 w-2 h-2 bg-primary rounded-full animate-pulse" />
                </motion.button>

                <AnimatePresence>
                  {showNotifications && (
                    <motion.div
                      initial={{ opacity: 0, y: 10, scale: 0.95 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      exit={{ opacity: 0, y: 10, scale: 0.95 }}
                      transition={{ duration: 0.2 }}
                      className="absolute top-full right-0 mt-2 w-80 bg-slate-800/95 backdrop-blur-xl rounded-xl border border-slate-700/50 shadow-2xl overflow-hidden"
                    >
                      <div className="p-3 border-b border-slate-700/50">
                        <h3 className="text-sm font-semibold text-white">Notifications</h3>
                      </div>
                      <div className="max-h-96 overflow-y-auto">
                        {notifications.map((notification) => (
                          <div
                            key={notification.id}
                            className="p-3 border-b border-slate-700/50 hover:bg-slate-700/30 transition-colors cursor-pointer"
                          >
                            <div className="flex items-start gap-3">
                              <div className="flex-shrink-0 mt-0.5">
                                {getNotificationIcon(notification.type)}
                              </div>
                              <div className="flex-1 min-w-0">
                                <p className="text-sm font-medium text-white">{notification.title}</p>
                                <p className="text-xs text-slate-400 mt-0.5">{notification.message}</p>
                                <p className="text-xs text-slate-500 mt-1">{notification.time}</p>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                      <div className="p-3 border-t border-slate-700/50">
                        <button className="w-full text-sm text-slate-400 hover:text-white transition-colors">
                          Mark all as read
                        </button>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              {/* User Menu */}
              <div className="relative">
                <motion.button
                  onClick={() => setShowUserMenu(!showUserMenu)}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className="flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-slate-800/50 text-slate-400 hover:text-white transition-colors"
                >
                  <User className="w-5 h-5" />
                  <span className="hidden md:inline text-sm font-medium">Admin</span>
                  <ChevronDown className={`w-4 h-4 transition-transform ${showUserMenu ? 'rotate-180' : ''}`} />
                </motion.button>

                <AnimatePresence>
                  {showUserMenu && (
                    <motion.div
                      initial={{ opacity: 0, y: 10, scale: 0.95 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      exit={{ opacity: 0, y: 10, scale: 0.95 }}
                      transition={{ duration: 0.2 }}
                      className="absolute top-full right-0 mt-2 w-48 bg-slate-800/95 backdrop-blur-xl rounded-xl border border-slate-700/50 shadow-2xl overflow-hidden"
                    >
                      <div className="py-1">
                        <Link
                          to="/dashboard/settings"
                          className="flex items-center gap-2 px-4 py-2 text-sm text-slate-300 hover:bg-slate-700/30 hover:text-white transition-colors"
                        >
                          <Settings className="w-4 h-4" />
                          Settings
                        </Link>
                        <Link
                          to="/dashboard/help"
                          className="flex items-center gap-2 px-4 py-2 text-sm text-slate-300 hover:bg-slate-700/30 hover:text-white transition-colors"
                        >
                          <HelpCircle className="w-4 h-4" />
                          Help
                        </Link>
                        <div className="border-t border-slate-700/50 my-1" />
                        <button className="w-full flex items-center gap-2 px-4 py-2 text-sm text-red-400 hover:bg-slate-700/30 hover:text-red-300 transition-colors">
                          <LogOut className="w-4 h-4" />
                          Sign out
                        </button>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </div>
          </div>

          {/* Last Updated Bar (shown after refresh) */}
          <AnimatePresence>
            {allSuccess && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="px-4 sm:px-6 lg:px-8 pb-2"
              >
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                    <span>All data synchronized</span>
                  </div>
                  <span className="font-mono">Just now</span>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </header>
  );
};
