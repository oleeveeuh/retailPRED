/**
 * Main App Component
 * Sets up React Router and Query Client
 */

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Layout } from './components/layout';
import { Dashboard } from './components/Dashboard';
import { PredictionsPage, ModelsPage, ValidationPage, ExplainPage } from './pages';
import { EconomicScenarioAnalysis } from './pages/EconomicScenarioAnalysis';
import { SensitivityAnalysis } from './pages/SensitivityAnalysis';
import { BusinessDashboard } from './pages/BusinessDashboard';
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

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Layout>
          <Routes>
            <Route path="/" element={<Navigate to="/dashboard/overview" replace />} />
            <Route path="/dashboard/overview" element={<Dashboard />} />
            <Route path="/dashboard/predictions" element={<PredictionsPage />} />
            <Route path="/dashboard/models" element={<ModelsPage />} />
            <Route path="/dashboard/validation" element={<ValidationPage />} />
            <Route path="/dashboard/explain" element={<ExplainPage />} />
            <Route path="/dashboard/scenarios" element={<EconomicScenarioAnalysis />} />
            <Route path="/dashboard/sensitivity" element={<SensitivityAnalysis />} />
            <Route path="/dashboard/business" element={<BusinessDashboard />} />
          </Routes>
        </Layout>
      </BrowserRouter>
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  );
}

export default App;
