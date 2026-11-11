import type { FC } from 'react';
import { useQuery } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import {
  CheckCircle2,
  Clock,
  Satellite,
  ShieldAlert,
  XCircle,
} from 'lucide-react';

import {
  fetchIngestStatus,
  fetchRunHistory,
  isIngestStatusConfigured,
  type IngestStatus,
} from '../api/ingestStatus';
import {
  gateChips,
  liveState,
  missingRunCopy,
  statusSummary,
  unavailableRunCopy,
  type ChipTone,
} from '../content/pipelineStatus';

const toneClasses: Record<ChipTone, string> = {
  ok: 'border-emerald-300 bg-emerald-50 text-emerald-800',
  gate: 'border-amber-300 bg-amber-50 text-amber-800',
  awaiting: 'border-sky-300 bg-sky-50 text-sky-800',
  dormant: 'border-stone-300 bg-stone-50 text-stone-600',
};

const toneIcon = (tone: ChipTone) => {
  switch (tone) {
    case 'ok':
      return <CheckCircle2 className="h-3.5 w-3.5" />;
    case 'gate':
      return <ShieldAlert className="h-3.5 w-3.5" />;
    case 'awaiting':
      return <Clock className="h-3.5 w-3.5" />;
    default:
      return <XCircle className="h-3.5 w-3.5" />;
  }
};

const isMissingRecord = (error: unknown): boolean =>
  error instanceof Error && error.message.includes('no hosted run recorded yet');

/** The fail-closed framing shared by a 404 and an invalid artifact: the
 * absence of a record is a governed state, never a broken look. */
const FailClosedNotice: FC<{ title: string; body: string }> = ({ title, body }) => (
  <div className="border border-[#d8d1c4] bg-[#f5f1e8] p-4">
    <div className="flex items-center gap-2">
      <ShieldAlert className="h-4 w-4 shrink-0 text-[#a43b36]" />
      <h4 className="font-mono text-xs uppercase tracking-wide text-stone-700">{title}</h4>
    </div>
    <p className="mt-2 text-xs leading-relaxed text-stone-600">{body}</p>
  </div>
);

const RunHistoryList: FC<{ runs: IngestStatus[] }> = ({ runs }) => {
  const recent = runs.slice(-5).reverse();
  if (recent.length === 0) return null;
  return (
    <div className="border-t border-[#d8d1c4] pt-3">
      <p className="font-mono text-[10px] uppercase tracking-[0.13em] text-stone-500">
        run history · last {recent.length} of {runs.length} recorded
      </p>
      <ul className="mt-2 space-y-1">
        {recent.map((run) => {
          const summary = statusSummary(run);
          return (
            <li
              key={`${summary.runId}-${run.finished_at}`}
              className="flex flex-wrap items-center gap-x-3 gap-y-0.5 font-mono text-[11px] text-stone-600"
            >
              <span className="text-stone-800">{summary.lastRun}</span>
              <span>run {summary.runId}</span>
              <span className={run.gates.publish === 'declined' ? 'text-[#a43b36]' : 'text-emerald-800'}>
                publish {run.gates.publish}
              </span>
            </li>
          );
        })}
      </ul>
      <p className="mt-2 text-[11px] text-stone-500">
        The log is append-only, machine-written by the pipeline — refusals included.
      </p>
    </div>
  );
};

/**
 * The last Databricks run's honest verdicts, from the machine-written
 * 2.2-status record. Renders the fail-closed framing when no record
 * exists: the pipeline refusing to fabricate status is the feature.
 */
export const PipelineStatusCard: FC = () => {
  const status = useQuery({
    queryKey: ['v2-ingest-status'],
    queryFn: fetchIngestStatus,
    retry: false,
    staleTime: 60_000,
  });
  const runs = useQuery({
    queryKey: ['v2-run-history'],
    queryFn: fetchRunHistory,
    retry: false,
    staleTime: 60_000,
  });

  if (!isIngestStatusConfigured()) {
    return null;
  }

  if (status.isError) {
    const missing = isMissingRecord(status.error);
    const copy = missing ? missingRunCopy : unavailableRunCopy;
    return (
      <motion.section
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        aria-label="Pipeline status"
        className="border border-[#d8d1c4] bg-[#fbf9f4] p-5"
      >
        <CardHeader />
        <div className="mt-3">
          <FailClosedNotice title={copy.title} body={copy.body} />
        </div>
      </motion.section>
    );
  }

  const record = status.data;
  if (!record) {
    return (
      <motion.section aria-label="Pipeline status" className="border border-[#d8d1c4] bg-[#fbf9f4] p-5">
        <CardHeader />
        <p className="mt-3 text-xs text-stone-500">Fetching the last run's record…</p>
      </motion.section>
    );
  }

  const summary = statusSummary(record);
  const chips = gateChips(record);

  return (
    <motion.section
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      aria-label="Pipeline status"
      className="border border-[#d8d1c4] bg-[#fbf9f4] p-5"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <CardHeader />
        <span className="flex items-center gap-1 border border-[#d8d1c4] bg-white px-2 py-0.5 font-mono text-[11px] text-stone-700">
          <Satellite className="h-3 w-3 text-[#89927a]" />
          {liveState(record)}
        </span>
      </div>

      <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="last run (UTC)" value={summary.lastRun} />
        <Stat label="data cutoff" value={summary.cutoff} />
        <Stat label="silver rows in Delta" value={summary.rows} />
        <Stat label="pipeline run" value={summary.runId} />
      </div>

      <div className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-4">
        {chips.map((chip) => (
          <div key={chip.key} className={`flex items-start gap-2 border p-2.5 ${toneClasses[chip.tone]}`}>
            <span className="mt-0.5 shrink-0">{toneIcon(chip.tone)}</span>
            <div>
              <p className="text-xs font-medium">{chip.label}</p>
              <p className="mt-0.5 text-[11px] leading-snug opacity-80">{chip.detail}</p>
            </div>
          </div>
        ))}
      </div>

      <p className="mt-3 border-t border-[#d8d1c4] pt-3 text-[11px] leading-relaxed text-stone-500">
        This card is a status record, not a forecast claim: it reports what the scheduled pipeline
        actually did — including the gates that refused — and its presence authorizes nothing.
        {runs.isError && ' Run history is unavailable right now; the current record above is unchanged.'}
      </p>

      {runs.data && <RunHistoryList runs={runs.data} />}
    </motion.section>
  );
};

const CardHeader: FC = () => (
  <div>
    <p className="font-mono text-[10px] uppercase tracking-[0.13em] text-stone-500">
      live loop · contract 2.2-status
    </p>
    <h3 className="mt-1 text-lg text-stone-900">what the pipeline did last run</h3>
  </div>
);

const Stat: FC<{ label: string; value: string }> = ({ label, value }) => (
  <div className="border border-[#d8d1c4] bg-white p-2.5">
    <p className="text-[10px] uppercase tracking-wide text-stone-400">{label}</p>
    <p className="mt-1 font-mono text-sm text-[#25251f]">{value}</p>
  </div>
);
