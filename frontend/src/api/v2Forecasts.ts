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
  const response = await fetch('/demo-data/v2-forecast.json');
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

export const v2ForecastsApi = {
  forecast: config.isDemoMode ? demoForecast : apiForecast,
};
