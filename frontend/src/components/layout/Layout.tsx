/**
 * Minimalistic Layout Component
 *
 * Features:
 * - Clean white background
 * - Sharp corners (rounded-sm)
 * - Light typography (font-light)
 * - Minimal borders
 * - RetailPRED brand colors (field-notes paper, ink, sage, crimson)
 */

import { FC, ReactNode } from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import { Sidebar } from './Sidebar';
import { Header } from './Header';
import { DemoBanner } from '../DemoBanner';

interface LayoutProps {
  children: ReactNode;
}

const pageVariants = {
  initial: { opacity: 0, y: 10 },
  enter: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -10 },
};

const pageTransition = {
  type: 'spring',
  damping: 25,
  stiffness: 300,
};

export const Layout: FC<LayoutProps> = ({ children }) => {
  return (
    <div className="flex min-h-screen flex-col bg-[#f5f1e8]">
      {/* Demo Banner */}
      <DemoBanner />

      {/* Main Layout */}
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <Sidebar />

        {/* Main Content Area */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Header */}
          <Header />

          {/* Page Content with Transitions */}
          <motion.main
            initial="initial"
            animate="enter"
            exit="exit"
            variants={pageVariants}
            transition={pageTransition}
            className="flex-1 overflow-x-hidden overflow-y-auto bg-stone-50/50"
          >
            <div className="p-6 sm:p-8">
              {/* Content Container */}
              <motion.div
                initial={{ opacity: 0, y: 5 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1, duration: 0.3 }}
                className="max-w-7xl mx-auto"
              >
                {children}
              </motion.div>
            </div>
          </motion.main>
        </div>
      </div>
    </div>
  );
};
