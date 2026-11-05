import type { FC } from 'react';
import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import toast, { Toaster } from 'react-hot-toast';
import {
  AlertCircle,
  BarChart3,
  CheckCircle2,
  Database,
  RefreshCw,
  ShieldAlert,
  Zap,
} from 'lucide-react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import { v2ForecastsApi } from '../api/v2Forecasts';
import { LatestForecastPanel } from '../components/LatestForecastPanel';

const FIXTURE = {
  dataset: 'm5' as const,
  seriesId: 'FOODS_1_001_CA_1_evaluation',
  forecastOrigin: '2016-04-24',
  modelName: 'm5_global_lightgbm',
};

export const PredictionsPage: FC = () => {
  const [horizon, setHorizon] = useState(4);
  const prediction = useMutation({
    mutationFn: () =>
      v2ForecastsApi.forecast({
        dataset: FIXTURE.dataset,
        series_id: FIXTURE.seriesId,
        forecast_origin: FIXTURE.forecastOrigin,
        horizon,
        requested_model: FIXTURE.modelName,
        mode: 'offline_demo',
        freshness_requirement: 'historical_allowed',
      }),
    onSuccess: () => {
      toast.success('Historical evaluation fixture loaded.', {
        icon: <CheckCircle2 className="h-5 w-5 text-emerald-600" />,
      });
    },
    onError: (error: Error) => {
      toast.error(error.message, {
        icon: <AlertCircle className="h-5 w-5 text-red-600" />,
      });
    },
  });

  const chartData =
    prediction.data?.points.map((point) => ({
      date: new Date(`${point.target_time}T00:00:00`).toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
      }),
      forecast: point.prediction,
      actual: point.evaluation_actual,
    })) ?? [];
  const meanForecast = chartData.length
    ? chartData.reduce((total, point) => total + point.forecast, 0) / chartData.length
    : 0;
  const meanActual = chartData.length
    ? chartData.reduce((total, point) => total + (point.actual ?? 0), 0) / chartData.length
    : 0;

  return (
    <div className="space-y-6">
      <Toaster position="top-right" />

      <header>
        <h1 className="font-serif text-2xl lowercase tracking-[-0.02em] text-[#25251f]">Forecast evidence</h1>
        <p className="mt-1 text-sm font-light text-stone-500">
          Inspect a measured historical M5 forecast with complete provenance.
        </p>
      </header>

      <div className="flex items-start gap-3 border border-amber-300 bg-amber-50 p-4 text-amber-950">
        <ShieldAlert className="mt-0.5 h-5 w-5 shrink-0" />
        <div>
          <p className="text-sm font-semibold uppercase tracking-wide">Historical offline demo</p>
          <p className="mt-1 text-sm">
            This page shows a frozen 2016 evaluation fixture. It is not live inference and must not
            be used as a current retail forecast. Production MRTS and M5 inference remain unavailable.
          </p>
        </div>
      </div>

      <LatestForecastPanel />

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-5">
        <motion.section
          initial={{ opacity: 0, x: -16 }}
          animate={{ opacity: 1, x: 0 }}
          className="space-y-5 border border-stone-200 bg-white p-5 lg:col-span-2"
        >
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-full bg-[#dfe1d5]">
              <Database className="h-4 w-4 text-[#89927a]" />
            </div>
            <div>
              <h2 className="text-xs font-medium uppercase tracking-wide text-stone-900">
                Frozen fixture
              </h2>
              <p className="text-xs text-stone-500">Inputs cannot be changed silently.</p>
            </div>
          </div>

          <dl className="space-y-4 text-sm">
            <div>
              <dt className="text-xs font-medium uppercase tracking-wide text-stone-500">Series</dt>
              <dd className="mt-1 break-all text-stone-900">{FIXTURE.seriesId}</dd>
            </div>
            <div>
              <dt className="text-xs font-medium uppercase tracking-wide text-stone-500">Model</dt>
              <dd className="mt-1 flex items-center gap-2 text-stone-900">
                <Zap className="h-4 w-4 text-[#25251f]" /> M5 Global LightGBM v1.0.0
              </dd>
            </div>
            <div>
              <dt className="text-xs font-medium uppercase tracking-wide text-stone-500">
                Forecast origin and data cutoff
              </dt>
              <dd className="mt-1 text-stone-900">April 24, 2016</dd>
            </div>
          </dl>

          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <label className="text-xs font-medium uppercase tracking-wide text-stone-700">
                Demonstration horizon
              </label>
              <span className="bg-stone-100 px-3 py-1 text-sm text-[#25251f]">
                {horizon} {horizon === 1 ? 'day' : 'days'}
              </span>
            </div>
            <input
              aria-label="Demonstration horizon"
              type="range"
              min="1"
              max="4"
              value={horizon}
              onChange={(event) => setHorizon(Number(event.target.value))}
              className="h-2 w-full cursor-pointer appearance-none bg-stone-200 accent-[#25251f]"
            />
          </div>

          <button
            onClick={() => prediction.mutate()}
            disabled={prediction.isPending}
            className="flex w-full items-center justify-center gap-2 bg-[#dfe1d5] py-3 text-white transition hover:bg-[#4a4a7c] disabled:cursor-not-allowed disabled:opacity-50"
          >
            {prediction.isPending ? (
              <RefreshCw className="h-5 w-5 animate-spin" />
            ) : (
              <BarChart3 className="h-5 w-5" />
            )}
            Load historical forecast
          </button>

          <div className="border border-slate-200 bg-slate-50 p-3 text-xs text-slate-700">
            Calibrated uncertainty is unavailable: the evaluated interval policy failed its
            high-volume coverage gate. No confidence band is displayed.
          </div>
        </motion.section>

        <motion.section
          initial={{ opacity: 0, x: 16 }}
          animate={{ opacity: 1, x: 0 }}
          className="space-y-5 lg:col-span-3"
        >
          {!prediction.data ? (
            <div className="flex min-h-[420px] flex-col items-center justify-center border border-stone-200 bg-white p-12 text-center">
              <div className="mb-5 flex h-16 w-16 items-center justify-center rounded-full bg-[#dfe1d5]">
                <BarChart3 className="h-8 w-8 text-[#89927a]" />
              </div>
              <h2 className="text-lg text-stone-900">Evidence ready to inspect</h2>
              <p className="mt-2 max-w-md text-sm text-stone-500">
                Load the deterministic fixture to compare the historical forecast with its realized
                evaluation actuals.
              </p>
            </div>
          ) : (
            <>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
                <Metric label="Mean forecast" value={meanForecast.toFixed(3)} />
                <Metric label="Mean actual" value={meanActual.toFixed(3)} />
                <Metric label="Data status" value="Historical" />
              </div>

              <div className="border border-stone-200 bg-white p-6">
                <div className="mb-5">
                  <h2 className="text-lg text-stone-900">Unit demand: forecast versus actual</h2>
                  <p className="text-sm text-stone-500">
                    Point forecasts only; values are units, not dollars.
                  </p>
                </div>
                <ResponsiveContainer width="100%" height={320}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="date" tick={{ fontSize: 12 }} />
                    <YAxis allowDecimals domain={[0, 'auto']} tick={{ fontSize: 12 }} />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="forecast" name="Forecast" stroke="#a43b36" strokeWidth={3} />
                    <Line type="monotone" dataKey="actual" name="Evaluation actual" stroke="#89927a" strokeWidth={3} />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              <div className="border border-stone-200 bg-white p-5 text-sm text-stone-700">
                <h2 className="mb-3 text-xs font-semibold uppercase tracking-wide text-stone-900">
                  Provenance
                </h2>
                <dl className="grid gap-3 sm:grid-cols-2">
                  <Provenance label="Contract" value={prediction.data.contract_version} />
                  <Provenance label="Model status" value={prediction.data.model_status} />
                  <Provenance label="Data cutoff" value={prediction.data.data_cutoff} />
                  <Provenance label="Code revision" value={prediction.data.provenance.code_revision.slice(0, 12)} />
                  <Provenance label="Evidence" value={prediction.data.provenance.evidence} />
                  <Provenance label="Uncertainty" value={prediction.data.uncertainty_status} />
                </dl>
              </div>
            </>
          )}
        </motion.section>
      </div>
    </div>
  );
};

const Metric: FC<{ label: string; value: string }> = ({ label, value }) => (
  <div className="border border-stone-200 bg-white p-5">
    <p className="text-xs font-medium uppercase tracking-wide text-stone-500">{label}</p>
    <p className="mt-2 text-2xl text-stone-900">{value}</p>
  </div>
);

const Provenance: FC<{ label: string; value: string }> = ({ label, value }) => (
  <div>
    <dt className="text-xs uppercase tracking-wide text-stone-500">{label}</dt>
    <dd className="mt-1 break-all text-stone-900">{value}</dd>
  </div>
);
