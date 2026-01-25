/**
 * Minimalistic Header
 *
 * Features:
 * - Clean white background with subtle border
 * - Sharp corners
 * - Light typography
 * - Simple refresh button
 * - Minimal notifications
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
    { name: 'Overview', path: '/dashboard/overview' },
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

export const Header: FC = () => {
  const location = useLocation();
  const queryClient = useQueryClient();
  const [refreshStatus, setRefreshStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [modelStatus, setModelStatus] = useState<'loading' | 'loaded' | 'error'>('loading');
  const [showNotifications, setShowNotifications] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
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
        return <CheckCircle2 className="w-4 h-4 text-emerald-500" />;
      case 'warning':
        return <XCircle className="w-4 h-4 text-amber-500" />;
      case 'error':
        return <XCircle className="w-4 h-4 text-red-500" />;
      default:
        return <Bell className="w-4 h-4 text-[#3A3A6C]" />;
    }
  };

  return (
    <header className="bg-white border-b border-gray-200">
      <div className="px-6 py-4">
        <div className="flex items-center justify-between gap-4">
          {/* Left: Breadcrumbs */}
          <div className="flex items-center gap-4 flex-1 min-w-0">
            {/* Mobile Menu Button */}
            <button className="lg:hidden p-2 rounded-sm hover:bg-gray-100 text-gray-500 hover:text-gray-900 transition-colors">
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>

            {/* Breadcrumbs */}
            <nav className="hidden md:flex items-center space-x-2 text-sm overflow-x-auto">
              {breadcrumbs.map((crumb, index) => (
                <div key={`${crumb.path}-${index}`} className="flex items-center space-x-2">
                  {index > 0 && (
                    <ChevronDown className="w-3 h-3 text-gray-400 rotate-[-90deg]" />
                  )}
                  <Link
                    to={crumb.path}
                    className={`
                      font-light transition-colors whitespace-nowrap
                      ${
                        index === breadcrumbs.length - 1
                          ? 'text-gray-900'
                          : 'text-gray-500 hover:text-gray-700'
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
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                type="text"
                placeholder="Search..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-56 pl-9 pr-3 py-1.5 bg-gray-50 border border-gray-200 rounded-sm text-sm text-gray-900 placeholder-gray-400 focus:outline-none focus:border-[#81C1AC] focus:ring-1 focus:ring-[#81C1AC] transition-all font-light"
              />
            </div>

            {/* Refresh Button */}
            <motion.button
              onClick={handleRefresh}
              disabled={refreshStatus === 'loading'}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className={`
                px-3 py-1.5 rounded-sm text-sm font-light
                transition-all duration-150 flex items-center space-x-2
                ${
                  refreshStatus === 'loading'
                    ? 'bg-amber-500 text-white cursor-wait'
                    : refreshStatus === 'success'
                    ? 'bg-emerald-500 text-white'
                    : refreshStatus === 'error'
                    ? 'bg-red-500 text-white hover:bg-red-600'
                    : 'bg-[#3A3A6C] text-white hover:bg-[#2F2F5A]'
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
                    <RefreshCw className="w-3.5 h-3.5" />
                  </motion.div>
                ) : refreshStatus === 'success' ? (
                  <motion.div
                    key="success"
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    exit={{ scale: 0 }}
                  >
                    <CheckCircle2 className="w-3.5 h-3.5" />
                  </motion.div>
                ) : refreshStatus === 'error' ? (
                  <motion.div
                    key="error"
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    exit={{ scale: 0 }}
                  >
                    <XCircle className="w-3.5 h-3.5" />
                  </motion.div>
                ) : (
                  <motion.div
                    key="idle"
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    exit={{ scale: 0 }}
                  >
                    <RefreshCw className="w-3.5 h-3.5" />
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
                  : 'Refresh'}
              </span>
            </motion.button>

            {/* Model Status Indicator */}
            <div className="hidden md:flex items-center space-x-2 px-3 py-1.5 bg-gray-50 rounded-sm border border-gray-200">
              <div className={`w-1.5 h-1.5 rounded-full ${
                modelStatus === 'loaded' ? 'bg-emerald-500' :
                modelStatus === 'loading' ? 'bg-amber-500' :
                'bg-red-500'
              }`} />
              <span className="text-xs font-light text-gray-600">
                {modelStatus === 'loaded' ? 'Ready' :
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
                className="relative p-2 rounded-sm hover:bg-gray-100 text-gray-500 hover:text-gray-900 transition-colors"
              >
                <Bell className="w-4 h-4" />
                <span className="absolute top-1 right-1 w-1.5 h-1.5 bg-red-500 rounded-full" />
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
                      className="absolute right-0 mt-2 w-72 bg-white rounded-sm shadow-lg border border-gray-200 z-50"
                    >
                      <div className="p-3 border-b border-gray-100">
                        <h3 className="text-sm font-normal text-gray-900">Notifications</h3>
                        <p className="text-xs font-light text-gray-500 mt-0.5">{notifications.length} new notifications</p>
                      </div>
                      <div className="max-h-80 overflow-y-auto">
                        {notifications.map((notif) => (
                          <div
                            key={notif.id}
                            className="p-3 hover:bg-gray-50 transition-colors border-b border-gray-100 last:border-0"
                          >
                            <div className="flex items-start space-x-2">
                              {getNotificationIcon(notif.type)}
                              <div className="flex-1 min-w-0">
                                <p className="text-xs font-normal text-gray-900">{notif.title}</p>
                                <p className="text-xs font-light text-gray-500 mt-0.5">{notif.message}</p>
                                <p className="text-xs font-light text-gray-400 mt-1">{notif.time}</p>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                      <div className="p-2 border-t border-gray-100">
                        <button className="w-full text-center text-xs text-[#3A3A6C] hover:text-[#2F2F5A] font-light transition-colors py-1">
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
    </header>
  );
};
