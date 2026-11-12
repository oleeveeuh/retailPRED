import { useQuery } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import { ExternalLink, RefreshCw } from 'lucide-react';
import type { FC } from 'react';

import { fetchLatestSnapshot, isSnapshotConfigured } from '../api/v2Forecasts';
import { entrance } from '../lib/entrance';

const RUN_LOG_URL =
  'https://github.com/oleeveeuh/retailPRED/blob/main/docs/v2/DATABRICKS_RUN.md';

const TASKS: Array<{ step: string; label: string; artifact: string }> = [
  { step: '1', label: 'pull census release', artifact: 'silver · immutable merge' },
  { step: '2', label: 'score promoted model', artifact: 'gold.forecasts · fail-closed gate' },
  { step: '3', label: 'evaluate realized errors', artifact: 'gold.realized_errors' },
  { step: '4', label: 'publish snapshot', artifact: 'live-data branch · one json' },
];

const monoLabel = 'font-mono text-[10px] uppercase tracking-[0.13em]';

/**
 * What the scheduled Databricks cycle does each run, and its live state.
 * Reads the published snapshot's own provenance fields when a snapshot URL
 * is configured; otherwise it shows the declared state only — cron cadence,
 * paused schedule, and the verified fail-closed run. Nothing here upgrades a
 * claim level: the vocabulary belongs to docs/v2/DATABRICKS_RUN.md.
 */
export const PipelineCyclePanel: FC = () => {
  const configured = isSnapshotConfigured();
  const snapshot = useQuery({
    queryKey: ['v2-latest-snapshot'],
    queryFn: fetchLatestSnapshot,
    enabled: configured,
    staleTime: 5 * 60 * 1000,
    retry: 1,
  });
  const data = snapshot.data;

  return (
    <motion.section
      {...entrance(0.05)}
      className="border border-stone-200 bg-white p-5"
      aria-label="scheduled databricks cycle status"
    >
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="text-sm font-semibold uppercase tracking-wide">
          Scheduled cycle · Databricks
        </h3>
        <span className={`${monoLabel} text-stone-400`}>
          cron 0 30 14 * * ? · daily 14:30 utc · paused
        </span>
      </div>

      {/* task graph */}
      <ol className="mt-4 grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-4">
        {TASKS.map((task) => (
          <li
            key={task.step}
            className="border border-stone-200 bg-[#fbf9f4] px-3 py-2.5"
          >
            <div className="flex items-center gap-2">
              <span className="flex h-5 w-5 items-center justify-center rounded-full bg-[#89927a] font-mono text-[10px] text-white">
                {task.step}
              </span>
              <span className="text-xs font-medium text-stone-700">
                {task.label}
              </span>
            </div>
            <p className={`${monoLabel} mt-1.5 text-stone-400`}>{task.artifact}</p>
          </li>
        ))}
      </ol>

      {/* state */}
      {data ? (
        <dl className="mt-4 grid grid-cols-1 gap-x-6 gap-y-1.5 text-xs text-stone-600 sm:grid-cols-2">
          <div className="flex gap-2">
            <dt className={`${monoLabel} shrink-0 text-stone-400`}>last publish</dt>
            <dd>{data.freshness.generated_at}</dd>
          </div>
          <div className="flex gap-2">
            <dt className={`${monoLabel} shrink-0 text-stone-400`}>data cutoff</dt>
            <dd>{data.freshness.data_cutoff}</dd>
          </div>
          <div className="flex gap-2">
            <dt className={`${monoLabel} shrink-0 text-stone-400`}>decision</dt>
            <dd>
              {data.model_status}
              {data.is_live
                ? ''
                : ' — serving stays fail-closed (is_live = false)'}
            </dd>
          </div>
          <div className="flex gap-2">
            <dt className={`${monoLabel} shrink-0 text-stone-400`}>pipeline run</dt>
            <dd>{data.pipeline_run_id ?? 'not recorded in this snapshot'}</dd>
          </div>
        </dl>
      ) : (
        <div className="mt-4 border-t border-stone-200 pt-3 text-xs leading-relaxed text-stone-500">
          <p>
            No snapshot is published yet — the scheduled job is unpaused
            (2026-11-11) and scoring stays refused until the vintage gate
            promotes a champion. The full four-task graph and every
            fail-closed gate are verified on a real workspace run
            (2026-11-04, run 452582046469351): ingest merged 2,200 silver
            rows from live Census, scoring refused the unauthorized
            challenger, realized errors skipped with no forecasts,
            publication declined without gold forecasts.
          </p>
          <p className="mt-2 flex items-center gap-3">
            <a
              href={RUN_LOG_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1 text-[#25251f] underline hover:no-underline"
            >
              run log — claim levels never silently upgrade{' '}
              <ExternalLink className="h-3 w-3" />
            </a>
            {configured && (
              <button
                type="button"
                onClick={() => snapshot.refetch()}
                className="flex items-center gap-1 text-stone-400 hover:text-[#a43b36]"
              >
                <RefreshCw className="h-3 w-3" aria-hidden="true" /> check again
              </button>
            )}
          </p>
        </div>
      )}
    </motion.section>
  );
};
