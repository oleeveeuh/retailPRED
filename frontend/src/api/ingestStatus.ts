/**
 * Client for the pipeline's per-run status record (contract `2.2-status`)
 * and its append-only run log (`live/runs.jsonl`).
 *
 * A status record is run mechanics and gate verdicts — never a forecast
 * claim. Fetching is fail-closed like the snapshot fetch: anything that
 * does not match the contract, or that smuggles forecast-claim keys, is
 * refused rather than displayed. A missing record is the honest
 * fail-closed state (no hosted run has completed yet), never an error to
 * paper over.
 */

import { config } from '../config/environment';

export interface IngestObservation {
  rows_in_delta: number | null;
  data_cutoff: string | null;
  series_count: number | null;
}

export interface ScoreGate {
  status: 'authorized' | 'refused';
  model: string;
  model_status: string;
  required_action: string;
}

export interface RealizedGate {
  status: 'evaluated' | 'skipped';
  realized_rows: number;
}

export interface StatusGates {
  ingest: 'success' | 'failed';
  score: ScoreGate;
  realized: RealizedGate;
  publish: 'published' | 'declined' | 'skipped-identical';
}

export interface IngestStatus {
  schema_version: '2.2-status';
  pipeline_run_id: string | null;
  job_id: string | null;
  pipeline_started_at: string | null;
  finished_at: string;
  ingest: IngestObservation;
  gates: StatusGates;
  is_live: boolean;
  generated_at: string;
}

/** Keys that would turn the mechanical record into a forecast claim. The
 * frontend refuses any status artifact carrying one, at any depth. */
const FORBIDDEN_CLAIM_KEYS: readonly string[] = [
  'prediction',
  'predictions',
  'forecast',
  'forecasts',
  'series',
  'points',
  'lower',
  'upper',
  'seasonal_naive_prediction',
  'evaluation_actual',
  'model_absolute_error',
  'seasonal_naive_absolute_error',
  'model_beat_seasonal_naive',
  'wape',
  'wrmsse',
  'mape',
  'accuracy',
];

function findClaimKeys(payload: unknown, prefix = ''): string[] {
  const found: string[] = [];
  if (Array.isArray(payload)) {
    payload.forEach((item, index) => found.push(...findClaimKeys(item, `${prefix}[${index}]`)));
  } else if (typeof payload === 'object' && payload !== null) {
    for (const [key, value] of Object.entries(payload)) {
      const path = prefix ? `${prefix}.${key}` : key;
      if (FORBIDDEN_CLAIM_KEYS.includes(key)) found.push(path);
      found.push(...findClaimKeys(value, path));
    }
  }
  return found;
}

function isNonNegativeInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value >= 0;
}

/**
 * Fail-closed validation: anything that does not match the 2.2-status
 * contract — or that carries a forecast-claim key anywhere — is refused.
 */
export function validateIngestStatusPayload(payload: unknown): IngestStatus {
  if (typeof payload !== 'object' || payload === null) {
    throw new Error('Status record is not a JSON object.');
  }
  const candidate = payload as Partial<IngestStatus>;
  if (candidate.schema_version !== '2.2-status') {
    throw new Error(
      `Status schema version must be 2.2-status, got ${String(candidate.schema_version)}; refusing to display it.`,
    );
  }
  if (!candidate.gates || typeof candidate.gates !== 'object') {
    throw new Error('Status record carries no gates; refusing to display it.');
  }
  const gates = candidate.gates as Partial<StatusGates>;
  if (gates.ingest !== 'success' && gates.ingest !== 'failed') {
    throw new Error('Status gate ingest must be success or failed; refusing to display it.');
  }
  const score = gates.score as Partial<ScoreGate> | undefined;
  if (!score || (score.status !== 'authorized' && score.status !== 'refused')) {
    throw new Error('Status gate score must be authorized or refused; refusing to display it.');
  }
  const realized = gates.realized as Partial<RealizedGate> | undefined;
  if (
    !realized ||
    (realized.status !== 'evaluated' && realized.status !== 'skipped') ||
    !isNonNegativeInteger(realized.realized_rows)
  ) {
    throw new Error('Status gate realized must carry a verdict and a row count; refusing to display it.');
  }
  if (
    gates.publish !== 'published' &&
    gates.publish !== 'declined' &&
    gates.publish !== 'skipped-identical'
  ) {
    throw new Error('Status gate publish has an unknown verdict; refusing to display it.');
  }
  if (!candidate.ingest || typeof candidate.ingest !== 'object') {
    throw new Error('Status record carries no ingest observation; refusing to display it.');
  }
  const ingest = candidate.ingest as Partial<IngestObservation>;
  for (const field of ['rows_in_delta', 'series_count'] as const) {
    const value = ingest[field];
    if (value !== null && value !== undefined && !isNonNegativeInteger(value)) {
      throw new Error(`Status ingest.${field} must be a non-negative integer or null.`);
    }
  }
  if (typeof candidate.is_live !== 'boolean') {
    throw new Error('Status is_live flag is missing; refusing to display it.');
  }
  if (candidate.is_live && (gates.ingest !== 'success' || gates.publish === 'declined')) {
    throw new Error(
      'Status claims is_live without a successful ingest and a completed publish; refusing to display it.',
    );
  }
  if (typeof candidate.finished_at !== 'string' || !candidate.finished_at) {
    throw new Error('Status record carries no finished_at timestamp.');
  }
  const claims = findClaimKeys(payload);
  if (claims.length > 0) {
    throw new Error(
      `Status record carries forbidden forecast-claim keys (${claims.join(', ')}); refusing to display it.`,
    );
  }
  return candidate as IngestStatus;
}

/** Fetch and validate a status record from an explicit URL. */
export async function fetchStatusFromUrl(url: string): Promise<IngestStatus> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 8000);
  try {
    const response = await fetch(url, { signal: controller.signal, cache: 'no-store' });
    if (response.status === 404) {
      // The honest empty state: no hosted run has completed a publish task.
      throw new Error('no hosted run recorded yet (404)');
    }
    if (!response.ok) {
      throw new Error(`Status request failed (${response.status}).`);
    }
    const payload: unknown = await response.json();
    return validateIngestStatusPayload(payload);
  } finally {
    clearTimeout(timer);
  }
}

/** Fetch the current run's status record relative to this deployment. */
export async function fetchIngestStatus(): Promise<IngestStatus> {
  const url = config.ingestStatusUrl.trim();
  if (!url) {
    throw new Error('No status URL is configured for this deployment.');
  }
  return fetchStatusFromUrl(url);
}

/**
 * Parse the append-only run log, newest line last. A malformed or
 * claim-carrying line fails the whole fetch — never a silently thinned
 * history.
 */
export function parseRunHistory(text: string): IngestStatus[] {
  const runs: IngestStatus[] = [];
  for (const line of text.split('\n')) {
    if (!line.trim()) continue;
    runs.push(validateIngestStatusPayload(JSON.parse(line) as unknown));
  }
  return runs;
}

/** Fetch the run log relative to this deployment. */
export async function fetchRunHistory(): Promise<IngestStatus[]> {
  const url = config.runHistoryUrl.trim();
  if (!url) {
    throw new Error('No run-history URL is configured for this deployment.');
  }
  return fetchRunHistoryFromUrl(url);
}

/** Fetch and validate the run log from an explicit URL. */
export async function fetchRunHistoryFromUrl(url: string): Promise<IngestStatus[]> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 8000);
  try {
    const response = await fetch(url, { signal: controller.signal, cache: 'no-store' });
    if (response.status === 404) {
      throw new Error('no hosted run recorded yet (404)');
    }
    if (!response.ok) {
      throw new Error(`Run-history request failed (${response.status}).`);
    }
    return parseRunHistory(await response.text());
  } finally {
    clearTimeout(timer);
  }
}

export function isIngestStatusConfigured(): boolean {
  return config.ingestStatusUrl.trim().length > 0;
}
