/**
 * Main App Component
 * Sets up React Router and Query Client
 */

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { HashRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Layout } from './components/layout';
import { PredictionsPage } from './pages/PredictionsPage';
import { ShowcasePage } from './pages/ShowcasePage';
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
      <HashRouter>
        <Layout>
          <Routes>
            <Route path="/" element={<ShowcasePage />} />
            <Route path="/dashboard/predictions" element={<PredictionsPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </Layout>
      </HashRouter>
    </QueryClientProvider>
  );
}

export default App;
