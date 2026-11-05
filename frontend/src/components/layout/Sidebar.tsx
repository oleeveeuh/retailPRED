/**
 * Minimalistic Sidebar Navigation
 *
 * Features:
 * - Clean design with sharp corners
 * - Light typography
 * - Active state with primary background + accent text
 * - RetailPRED brand colors
 */

import { FC, useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  BarChart3,
  ChevronLeft,
  ChevronRight,
  Home,
} from 'lucide-react';

interface NavItem {
  name: string;
  path: string;
  icon: React.ComponentType<{ className?: string }>;
}

const navItems: NavItem[] = [
  { name: 'Overview', path: '/', icon: Home },
  { name: 'Forecast evidence', path: '/dashboard/predictions', icon: BarChart3 },
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
            className="fixed inset-0 bg-black/20 z-40 lg:hidden"
          />
        )}
      </AnimatePresence>

      {/* Sidebar */}
      <motion.aside
        initial={false}
        animate={{
          width: isCollapsed ? '70px' : '240px',
        }}
        transition={{
          type: 'spring',
          damping: 25,
          stiffness: 200,
        }}
        className={`
          fixed lg:relative left-0 top-0 h-full z-50
          flex flex-col
          bg-white
          border-r border-stone-200
        `}
      >
        {/* Logo Section */}
        <div className="h-16 border-b border-stone-200 flex items-center justify-between px-4">
          <AnimatePresence mode="wait">
            {!isCollapsed && (
              <motion.div
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -10 }}
                transition={{ duration: 0.15 }}
                className="flex items-center gap-3"
              >
                <div className="w-9 h-9 bg-[#25251f] rounded-sm flex items-center justify-center">
                  <BarChart3 className="w-5 h-5 text-[#89927a]" />
                </div>
                <div>
                  <h1 className="font-mono text-sm font-semibold lowercase tracking-[-0.06em] text-[#25251f]">RetailPRED</h1>
                  <p className="font-mono text-[10px] lowercase text-stone-500">Evaluation portfolio</p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Collapse Toggle */}
          <motion.button
            onClick={toggleCollapse}
            className="p-1.5 rounded-sm hover:bg-stone-100 text-stone-400 hover:text-stone-700 transition-colors"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            aria-label="Toggle sidebar"
          >
            {isCollapsed ? (
              <ChevronRight className="w-4 h-4" />
            ) : (
              <ChevronLeft className="w-4 h-4" />
            )}
          </motion.button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 overflow-y-auto py-4 px-2 space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const active = isActive(item.path);

            return (
              <Link
                key={item.path}
                to={item.path}
                className={`
                  relative group flex items-center px-3 py-2
                  rounded-sm transition-all duration-150
                  ${
                    active
                      ? 'bg-[#25251f] text-[#89927a]'
                      : 'text-stone-600 hover:bg-stone-100 hover:text-stone-900'
                  }
                `}
              >
                {/* Active Indicator */}
                {active && (
                  <motion.div
                    layoutId="activeIndicator"
                    className="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-5 bg-[#89927a]"
                    transition={{ type: 'spring', stiffness: 300, damping: 30 }}
                  />
                )}

                {/* Icon */}
                <Icon className={`
                  w-4 h-4 flex-shrink-0
                  ${isCollapsed ? 'mx-auto' : 'mr-3'}
                `} />

                {/* Text */}
                <AnimatePresence mode="wait">
                  {!isCollapsed && (
                    <motion.span
                      initial={{ opacity: 0, x: -5 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -5 }}
                      transition={{ duration: 0.15 }}
                      className="text-sm font-light"
                    >
                      {item.name}
                    </motion.span>
                  )}
                </AnimatePresence>
              </Link>
            );
          })}
        </nav>

        {/* Mobile Close Button */}
        <motion.button
          onClick={() => setIsCollapsed(true)}
          className="lg:hidden absolute top-4 right-4 p-1.5 rounded-sm hover:bg-stone-100 text-stone-400"
          whileTap={{ scale: 0.95 }}
        >
          <ChevronLeft className="w-5 h-5" />
        </motion.button>
      </motion.aside>
    </>
  );
};
