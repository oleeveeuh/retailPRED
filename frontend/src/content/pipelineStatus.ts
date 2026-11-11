/**
 * Framing content for the pipeline-status card. Every verdict and number
 * rendered from a fetched record is copied from that record — this module
 * only shapes the honest phrasing. The refusals are the story: the card
 * presents a gate that said no exactly like it presents a gate that
 * passed, and a missing record reads as the fail-closed state it is,
 * never as a malfunction.
 */

import type { IngestStatus } from '../api/ingestStatus';

export type ChipTone = 'ok' | 'gate' | 'awaiting' | 'dormant';

export interface GateChip {
  key: string;
  label: string;
  detail: string;
  tone: ChipTone;
}

const formatRows = (rows: number): string => rows.toLocaleString('en-US');

/** One chip per gate, with the record's own verdict — never a gloss. */
export function gateChips(status: IngestStatus): GateChip[] {
  const { gates, ingest } = status;
  const chips: GateChip[] = [];

  if (gates.ingest === 'success') {
    chips.push({
      key: 'ingest',
      label: 'ingest ✓',
      detail: ingest.rows_in_delta !== null
        ? `${formatRows(ingest.rows_in_delta)} rows${ingest.data_cutoff ? ` · cutoff ${ingest.data_cutoff}` : ''}`
        : 'merge completed',
      tone: 'ok',
    });
  } else {
    chips.push({
      key: 'ingest',
      label: 'ingest failed',
      detail: 'no merge recorded for this run',
      tone: 'gate',
    });
  }

  if (gates.score.status === 'refused') {
    chips.push({
      key: 'score',
      label: 'score refused',
      detail: `challenger unauthorized · ${gates.score.model_status}`,
      tone: 'gate',
    });
  } else {
    chips.push({
      key: 'score',
      label: 'score authorized',
      detail: `${gates.score.model} refit at the scoring moment`,
      tone: 'ok',
    });
  }

  if (gates.realized.status === 'evaluated') {
    chips.push({
      key: 'realized',
      label: 'realized evaluated',
      detail: `${formatRows(gates.realized.realized_rows)} immutable rows`,
      tone: 'ok',
    });
  } else {
    chips.push({
      key: 'realized',
      label: 'realized skipped',
      detail:
        gates.realized.realized_rows > 0
          ? 'nothing new matured'
          : 'nothing matured yet',
      tone: 'awaiting',
    });
  }

  if (gates.publish === 'published') {
    chips.push({ key: 'publish', label: 'publish published', detail: 'snapshot committed', tone: 'ok' });
  } else if (gates.publish === 'skipped-identical') {
    chips.push({
      key: 'publish',
      label: 'publish skipped-identical',
      detail: 'unchanged content — no commit',
      tone: 'ok',
    });
  } else {
    chips.push({
      key: 'publish',
      label: 'publish declined',
      detail: 'no gold forecasts — nothing published',
      tone: 'dormant',
    });
  }

  return chips;
}

/** The schedule-state line, mirroring the snapshot contract's is_live rule. */
export function liveState(status: IngestStatus): string {
  return status.is_live ? 'scheduled runs: live' : 'scheduled runs: paused';
}

export function statusSummary(status: IngestStatus): {
  lastRun: string;
  cutoff: string;
  rows: string;
  runId: string;
} {
  return {
    lastRun: status.finished_at.slice(0, 10),
    cutoff: status.ingest.data_cutoff ?? 'unrecorded',
    rows:
      status.ingest.rows_in_delta !== null ? formatRows(status.ingest.rows_in_delta) : 'unrecorded',
    runId: status.pipeline_run_id ?? status.job_id ?? 'unrecorded',
  };
}

/** Rendered when the status artifact is absent (404 / never published). */
export const missingRunCopy = {
  title: 'no hosted run recorded yet',
  body:
    'The Databricks pipeline has not completed a publish task, so there is no status to show — and the pipeline refuses to fabricate one. An absent record means the run has not happened, not that something failed silently. That refusal is the governance story: every surface here reports exactly what did, or did not, run.',
} as const;

/** Rendered when a fetch fails or an artifact fails contract validation. */
export const unavailableRunCopy = {
  title: 'status artifact unavailable',
  body:
    'The status record could not be fetched, or it failed its contract validation. Nothing is shown rather than a stale or unverifiable state; the data branch is machine-written, and only records that validate end to end are displayed.',
} as const;
