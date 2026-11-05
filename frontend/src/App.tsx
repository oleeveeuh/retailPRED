/**
 * Main App Component
 * Sets up React Router and Query Client
 */

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MotionConfig } from 'framer-motion';
import { HashRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Layout } from './components/layout';
import { PredictionsPage } from './pages/PredictionsPage';
import { ShowcasePage } from './pages/ShowcasePage';
import { StoryPage } from './pages/StoryPage';
import './index.css';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 5 * 60 * 1000, // 5 minutes
    },
  },
});

// HashRouter keeps routing reliable on GitHub Pages, where arbitrary deep
// paths have no server-side rewrite to index.html.
function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <MotionConfig reducedMotion="user">
      <HashRouter>
        <Layout>
          <Routes>
            <Route path="/" element={<ShowcasePage />} />
            <Route path="/story" element={<StoryPage />} />
            <Route path="/dashboard/predictions" element={<PredictionsPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </Layout>
      </HashRouter>
      </MotionConfig>
    </QueryClientProvider>
  );
}

export default App;
