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
} from 'lucide-react';

import { fetchLatestSnapshot, isSnapshotConfigured } from '../api/v2Forecasts';

/**
 * Displays the latest published scheduled-batch forecast snapshot. The panel
 * renders only when the deployment configures a snapshot URL, and every state
 * is explicit: nothing is fetched silently and failed loads never fall back
 * to fabricated values.
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

  return (
    <motion.section
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="space-y-4 border border-emerald-300 bg-emerald-50/60 p-5"
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
          Check for latest forecasts
        </button>
      )}

      {snapshot.isError && (
        <div className="flex items-start gap-2 border border-amber-300 bg-amber-50 p-3 text-sm text-amber-950">
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
          <div>
            <p>
              Latest snapshot unavailable: {(snapshot.error as Error).message} The historical
              evaluation fixture below remains the displayed evidence.
            </p>
            <button
              onClick={() => snapshot.refetch()}
              className="mt-2 underline"
            >
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
                Snapshot status: <strong>{snapshot.data.freshness.status}</strong>. Batch generated{' '}
                {new Date(snapshot.data.freshness.generated_at).toLocaleString()} from data through{' '}
                {snapshot.data.freshness.data_cutoff}. Model status:{' '}
                <strong>{snapshot.data.model_status}</strong>.
              </p>
              {snapshot.data.freshness.next_scheduled_run && (
                <p className="mt-1 text-xs text-slate-500">
                  Next scheduled run:{' '}
                  {new Date(snapshot.data.freshness.next_scheduled_run).toLocaleString()}
                </p>
              )}
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-xs font-medium uppercase tracking-wide text-gray-700">
              Series
            </label>
            <select
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
            <table className="w-full border border-gray-200 bg-white text-sm">
              <thead>
                <tr className="bg-gray-100 text-left text-xs uppercase tracking-wide text-gray-500">
                  <th className="px-3 py-2">Horizon</th>
                  <th className="px-3 py-2">Target month</th>
                  <th className="px-3 py-2 text-right">Prediction</th>
                </tr>
              </thead>
              <tbody>
                {series.points.map((point) => (
                  <tr key={point.horizon} className="border-t border-gray-100">
                    <td className="px-3 py-2 text-gray-500">{point.horizon}</td>
                    <td className="px-3 py-2 text-gray-900">{point.target_time}</td>
                    <td className="px-3 py-2 text-right text-gray-900">
                      {point.prediction.toLocaleString(undefined, {
                        maximumFractionDigits: 2,
                      })}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          <div className="border border-slate-200 bg-slate-50 p-3 text-xs text-slate-700">
            Uncertainty: {snapshot.data.uncertainty_status}. No confidence band is shown unless a
            calibrated interval policy passes its promotion gate. Provenance:{' '}
            {snapshot.data.provenance.code_revision?.slice(0, 12) ?? 'unrecorded'}.
          </div>
        </div>
      )}
    </motion.section>
  );
};
