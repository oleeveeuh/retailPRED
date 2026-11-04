/** Minimal client for the only API surface published by the v2 portfolio. */

import { config } from '../config/environment';

export interface V2ForecastRequest {
  dataset: 'mrts' | 'm5';
  series_id: string;
  forecast_origin: string;
  horizon: number;
  requested_model?: string;
  mode: 'production' | 'offline_demo';
  freshness_requirement: 'current' | 'historical_allowed';
  quantiles?: number[];
}

export interface V2ForecastPoint {
  horizon: number;
  target_time: string;
  prediction: number;
  lower: number | null;
  upper: number | null;
  evaluation_actual: number | null;
}

export interface V2ForecastResponse {
  contract_version: '2.0';
  dataset: 'mrts' | 'm5';
  use_case: string;
  series_id: string;
  forecast_origin: string;
  data_cutoff: string;
  model_name: string;
  model_version: string;
  model_status: string;
  mode: 'offline_demo';
  data_status: 'historical_evaluation_fixture';
  uncertainty_status: string;
  is_demo: true;
  points: V2ForecastPoint[];
  provenance: Record<string, string>;
}

async function demoForecast(request: V2ForecastRequest): Promise<V2ForecastResponse> {
  if (
    request.dataset !== 'm5' ||
    request.series_id !== 'FOODS_1_001_CA_1_evaluation' ||
    request.forecast_origin !== '2016-04-24' ||
    request.mode !== 'offline_demo' ||
    request.freshness_requirement !== 'historical_allowed'
  ) {
    throw new Error('Only the labeled historical M5 fixture is available in offline demo mode.');
  }
  if (request.quantiles?.length) {
    throw new Error('Calibrated uncertainty is unavailable for this model.');
  }
  // BASE_URL keeps the fixture fetch correct on any deployment root
  // ('/' locally, '/retailPRED/' on GitHub Pages).
  const response = await fetch(`${import.meta.env.BASE_URL}demo-data/v2-forecast.json`);
  if (!response.ok) {
    throw new Error(`Historical fixture failed to load (${response.status}).`);
  }
  const fixture = (await response.json()) as V2ForecastResponse;
  if (request.horizon < 1 || request.horizon > fixture.points.length) {
    throw new Error(`Historical fixture supports horizons 1-${fixture.points.length}.`);
  }
  return { ...fixture, points: fixture.points.slice(0, request.horizon) };
}

async function apiForecast(request: V2ForecastRequest): Promise<V2ForecastResponse> {
  const apiRoot = config.apiUrl.replace(/\/$/, '');
  const response = await fetch(`${apiRoot}/api/v2/forecasts`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as
      | { detail?: { message?: string } }
      | null;
    throw new Error(payload?.detail?.message ?? `Forecast request failed (${response.status}).`);
  }
  return (await response.json()) as V2ForecastResponse;
}

export interface LatestSnapshotPoint {
  horizon: number;
  target_time: string;
  prediction: number;
  lower: number | null;
  upper: number | null;
  seasonal_naive_prediction: number | null;
  seasonal_naive_source_time: string | null;
  seasonal_naive_source_vintage_time: string | null;
  seasonal_naive_status: 'available' | 'unavailable';
  seasonal_naive_unavailable_reason: string | null;
  evaluation_actual: number | null;
  evaluation_actual_vintage_time: string | null;
  model_absolute_error: number | null;
  seasonal_naive_absolute_error: number | null;
  model_beat_seasonal_naive: boolean | null;
  evaluation_status: string | null;
}

export interface LatestSnapshotSeries {
  series_id: string;
  points: LatestSnapshotPoint[];
}

export interface LatestSnapshotFreshness {
  status: 'current' | 'stale';
  generated_at: string;
  forecast_vintage_time: string;
  data_cutoff: string;
  next_scheduled_run: string | null;
  max_age_hours: number;
}

export interface LatestSnapshot {
  contract_version: '2.1';
  snapshot_kind: 'scheduled_batch_forecasts';
  dataset: 'mrts';
  use_case: string;
  forecast_origin: string;
  model_name: string;
  model_version: string;
  model_status: string;
  uncertainty_status: string;
  is_live: boolean;
  pipeline_run_id: string | null;
  freshness: LatestSnapshotFreshness;
  series: LatestSnapshotSeries[];
  provenance: Record<string, string>;
}

export function isSnapshotConfigured(): boolean {
  return config.latestSnapshotUrl.trim().length > 0;
}

/**
 * Fail-closed validation: anything that does not match the v2 snapshot
 * contract is refused rather than displayed.
 */
export function validateSnapshotPayload(payload: unknown): LatestSnapshot {
  if (typeof payload !== 'object' || payload === null) {
    throw new Error('Snapshot is not a JSON object.');
  }
  const candidate = payload as Partial<LatestSnapshot>;
  if (candidate.contract_version !== '2.1') {
    throw new Error(
      `Snapshot contract version must be 2.1, got ${String(candidate.contract_version)}; refusing to display it.`,
    );
  }
  if (candidate.snapshot_kind !== 'scheduled_batch_forecasts') {
    throw new Error('Snapshot kind is not scheduled_batch_forecasts; refusing to display it.');
  }
  if (typeof candidate.is_live !== 'boolean') {
    throw new Error('Snapshot is_live flag is missing; refusing to display it.');
  }
  if (
    typeof candidate.model_status !== 'string' ||
    typeof candidate.uncertainty_status !== 'string'
  ) {
    throw new Error('Snapshot is missing model or uncertainty status.');
  }
  if (!candidate.freshness || typeof candidate.freshness.status !== 'string') {
    throw new Error('Snapshot is missing freshness metadata.');
  }
  if (!Array.isArray(candidate.series) || candidate.series.length === 0) {
    throw new Error('Snapshot contains no series.');
  }
  for (const series of candidate.series) {
    if (!Array.isArray(series.points) || series.points.length === 0) {
      throw new Error(`Snapshot series ${series.series_id} contains no points.`);
    }
    for (const point of series.points) {
      if (
        point.seasonal_naive_status !== 'available' &&
        point.seasonal_naive_status !== 'unavailable'
      ) {
        throw new Error(
          `Snapshot point for ${series.series_id} has an invalid seasonal-naive status.`,
        );
      }
      if (point.seasonal_naive_status === 'unavailable' && point.seasonal_naive_prediction !== null) {
        throw new Error(
          `Snapshot point for ${series.series_id} marks the benchmark unavailable but carries a value.`,
        );
      }
    }
  }
  return candidate as LatestSnapshot;
}

/**
 * Fetch the published scheduled-batch snapshot. cache: 'no-store' keeps a
 * refresh from serving a stale cached copy as newly fetched.
 */
export async function fetchLatestSnapshot(): Promise<LatestSnapshot> {
  const url = config.latestSnapshotUrl.trim();
  if (!url) {
    throw new Error('No snapshot URL is configured for this deployment.');
  }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 8000);
  try {
    const response = await fetch(url, { signal: controller.signal, cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`Snapshot request failed (${response.status}).`);
    }
    const payload: unknown = await response.json();
    return validateSnapshotPayload(payload);
  } finally {
    clearTimeout(timer);
  }
}

export const v2ForecastsApi = {
  forecast: config.isDemoMode ? demoForecast : apiForecast,
};
