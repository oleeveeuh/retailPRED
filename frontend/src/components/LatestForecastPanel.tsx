import type { FC } from 'react';
import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import {
  AlertCircle,
  CalendarClock,
  CheckCircle2,
  RefreshCw,
  Satellite,
  TrendingUp,
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

import {
  fetchLatestSnapshot,
  isSnapshotConfigured,
  type LatestSnapshot,
  type LatestSnapshotPoint,
} from '../api/v2Forecasts';

interface ChartRow {
  label: string;
  horizon: number;
  model: number | null;
  seasonalNaive: number | null;
  actual: number | null;
  sourceMonth: string | null;
  sourceVintage: string | null;
  modelError: number | null;
  naiveError: number | null;
  beat: boolean | null;
  evaluationStatus: string | null;
}

function toChartRows(points: LatestSnapshotPoint[]): ChartRow[] {
  return points.map((point) => ({
    label: point.target_time.slice(0, 7),
    horizon: point.horizon,
    model: point.prediction,
    seasonalNaive: point.seasonal_naive_status === 'available' ? point.seasonal_naive_prediction : null,
    actual: point.evaluation_actual,
    sourceMonth: point.seasonal_naive_source_time,
    sourceVintage: point.seasonal_naive_source_vintage_time,
    modelError: point.model_absolute_error,
    naiveError: point.seasonal_naive_absolute_error,
    beat: point.model_beat_seasonal_naive,
    evaluationStatus: point.evaluation_status,
  }));
}

function realizedSummary(snapshot: LatestSnapshot): { matured: number; wins: number } {
  const all = snapshot.series.flatMap((entry) => entry.points);
  const matured = all.filter((point) => point.evaluation_status !== null);
  const wins = matured.filter((point) => point.model_beat_seasonal_naive === true).length;
  return { matured: matured.length, wins };
}

const formatNumber = (value: number): string =>
  value.toLocaleString(undefined, { maximumFractionDigits: 2 });

/**
 * Displays the latest published scheduled-batch forecast snapshot. The panel
 * renders only when the deployment configures a snapshot URL. Fetching is
 * explicit and fail-closed: an invalid contract or a failed refresh never
 * replaces already-displayed data, and nothing is ever fabricated.
 */
export const LatestForecastPanel: FC = () => {
  const [seriesIndex, setSeriesIndex] = useState(0);
  const snapshot = useQuery({
    queryKey: ['v2-latest-snapshot'],
    queryFn: fetchLatestSnapshot,
    enabled: false,
    retry: false,
  });

  if (!isSnapshotConfigured()) {
    return null;
  }

  const series = snapshot.data?.series[seriesIndex];
  const chartRows = series ? toChartRows(series.points) : [];
  const summary = snapshot.data ? realizedSummary(snapshot.data) : { matured: 0, wins: 0 };

  return (
    <motion.section
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="space-y-4 border border-emerald-300 bg-emerald-50/60 p-5"
      aria-label="Latest published batch forecasts"
    >
      <div className="flex items-center gap-3">
        <div className="flex h-8 w-8 items-center justify-center bg-[#3A3A6C]">
          <Satellite className="h-4 w-4 text-[#81C1AC]" />
        </div>
        <div>
          <h2 className="text-xs font-medium uppercase tracking-wide text-gray-900">
            Latest published batch forecasts
          </h2>
          <p className="text-xs text-gray-500">
            Scheduled pipeline output from the governed store. Not on-demand inference.
          </p>
        </div>
      </div>

      {!snapshot.data && !snapshot.isError && (
        <div>
          <button
            onClick={() => snapshot.refetch()}
            disabled={snapshot.isFetching}
            className="flex w-full items-center justify-center gap-2 bg-[#3A3A6C] py-3 text-white transition hover:bg-[#4a4a7c] disabled:cursor-not-allowed disabled:opacity-50"
          >
            {snapshot.isFetching ? (
              <RefreshCw className="h-5 w-5 animate-spin" />
            ) : (
              <CalendarClock className="h-5 w-5" />
            )}
            Fetch latest published forecasts
          </button>
          <p className="mt-2 text-xs text-gray-500">
            This retrieves the newest forecast already produced by the Databricks pipeline. It does
            not run the model inside your browser.
          </p>
        </div>
      )}

      {snapshot.isError && (
        <div className="flex items-start gap-2 border border-amber-300 bg-amber-50 p-3 text-sm text-amber-950">
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
          <div>
            <p>
              Latest snapshot unavailable: {(snapshot.error as Error).message} Any previously
              displayed snapshot is unchanged and remains labeled as its original fetch.
            </p>
            <button onClick={() => snapshot.refetch()} className="mt-2 underline">
              Try again
            </button>
          </div>
        </div>
      )}

      {snapshot.data && (
        <div className="space-y-4">
          <div className="flex items-start gap-2 border border-slate-200 bg-white p-3 text-sm text-slate-700">
            {snapshot.data.freshness.status === 'current' ? (
              <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-600" />
            ) : (
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />
            )}
            <div>
              <p>
                Snapshot status: <strong>{snapshot.data.freshness.status}</strong>
                {snapshot.data.is_live ? ' (scheduled pipeline enabled)' : ' (manual pipeline runs)'}.
                Batch generated {new Date(snapshot.data.freshness.generated_at).toLocaleString()} from
                data through <strong>{snapshot.data.freshness.data_cutoff}</strong>. Model{' '}
                {snapshot.data.model_name} v{snapshot.data.model_version} — status{' '}
                <strong>{snapshot.data.model_status}</strong>.
              </p>
              {snapshot.data.freshness.next_scheduled_run && (
                <p className="mt-1 text-xs text-slate-500">
                  Next expected Databricks run:{' '}
                  {new Date(snapshot.data.freshness.next_scheduled_run).toLocaleString()}
                </p>
              )}
            </div>
          </div>

          <div className="flex items-start gap-2 border border-slate-200 bg-white p-3 text-sm text-slate-700">
            <TrendingUp className="mt-0.5 h-4 w-4 shrink-0 text-[#3A3A6C]" />
            <p>
              Realized performance so far: the model beat the seasonal-naive benchmark in{' '}
              <strong>
                {summary.wins} of {summary.matured}
              </strong>{' '}
              matured forecasts with published actuals.
              {summary.matured === 0 && ' No forecast target has a published actual yet.'}
            </p>
          </div>

          <div className="space-y-2">
            <label
              htmlFor="latest-snapshot-series"
              className="text-xs font-medium uppercase tracking-wide text-gray-700"
            >
              Series
            </label>
            <select
              id="latest-snapshot-series"
              value={seriesIndex}
              onChange={(event) => setSeriesIndex(Number(event.target.value))}
              className="w-full border border-gray-300 bg-white px-3 py-2 text-sm text-gray-900"
            >
              {snapshot.data.series.map((item, index) => (
                <option key={item.series_id} value={index}>
                  {item.series_id}
                </option>
              ))}
            </select>
          </div>

          {series && (
            <div className="border border-gray-200 bg-white p-4">
              <div className="mb-3">
                <h3 className="text-sm text-gray-900">Forecast versus benchmark and actuals</h3>
                <p className="text-xs text-gray-500">
                  The dashed benchmark is the same month one year earlier, as known at forecast
                  time, aligned to each target month.
                </p>
              </div>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={chartRows}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="label" tick={{ fontSize: 12 }} />
                  <YAxis allowDecimals domain={[0, 'auto']} tick={{ fontSize: 12 }} />
                  <Tooltip
                    content={({ active, payload }) => {
                      if (!active || !payload || payload.length === 0) return null;
                      const row = payload[0].payload as ChartRow;
                      return (
                        <div className="border border-gray-200 bg-white p-2 text-xs text-gray-700">
                          <p className="font-medium text-gray-900">{row.label}</p>
                          <p>Model forecast: {formatNumber(row.model ?? 0)}</p>
                          <p>
                            Seasonal naive — same month last year, as known at forecast time:{' '}
                            {row.seasonalNaive === null ? 'unavailable' : formatNumber(row.seasonalNaive)}
                          </p>
                          {row.sourceMonth && (
                            <p className="text-gray-500">Benchmark source month: {row.sourceMonth}</p>
                          )}
                          {row.sourceVintage && (
                            <p className="text-gray-500">
                              Benchmark source vintage: {new Date(row.sourceVintage).toLocaleString()}
                            </p>
                          )}
                          {row.actual !== null && <p>Published actual: {formatNumber(row.actual)}</p>}
                          {row.modelError !== null && (
                            <p>Model absolute error: {formatNumber(row.modelError)}</p>
                          )}
                          {row.naiveError !== null && (
                            <p>Benchmark absolute error: {formatNumber(row.naiveError)}</p>
                          )}
                          {row.beat !== null && (
                            <p>{row.beat ? 'Model beat the benchmark.' : 'Benchmark won this target.'}</p>
                          )}
                          {row.evaluationStatus && (
                            <p className="text-gray-500">Evaluation: {row.evaluationStatus}</p>
                          )}
                        </div>
                      );
                    }}
                  />
                  <Legend
                    formatter={(value) => (
                      <span className="text-xs text-gray-700">
                        {value === 'model' && 'Model forecast'}
                        {value === 'naive' && 'Seasonal naive (dashed)'}
                        {value === 'actual' && 'Published actual'}
                      </span>
                    )}
                  />
                  <Line
                    type="monotone"
                    dataKey="model"
                    name="model"
                    stroke="#3A3A6C"
                    strokeWidth={3}
                    dot={false}
                    connectNulls={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="seasonalNaive"
                    name="naive"
                    stroke="#81C1AC"
                    strokeWidth={2}
                    strokeDasharray="6 4"
                    dot={false}
                    connectNulls={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="actual"
                    name="actual"
                    stroke="#C2571B"
                    strokeWidth={2}
                    dot={{ r: 3 }}
                    connectNulls={false}
                  />
                </LineChart>
              </ResponsiveContainer>
              <p className="mt-2 text-xs text-gray-500">
                Dashes and labels distinguish the benchmark from the model forecast; missing
                benchmark values are gaps, never zeros.
              </p>
            </div>
          )}

          <div className="border border-slate-200 bg-slate-50 p-3 text-xs text-slate-700">
            Uncertainty: {snapshot.data.uncertainty_status}. No confidence band is shown unless a
            calibrated interval policy passes its promotion gate. Provenance:{' '}
            {snapshot.data.provenance.code_revision?.slice(0, 12) ?? 'unrecorded'}
            {snapshot.data.pipeline_run_id ? ` · pipeline run ${snapshot.data.pipeline_run_id}` : ''}.
          </div>
        </div>
      )}
    </motion.section>
  );
};
