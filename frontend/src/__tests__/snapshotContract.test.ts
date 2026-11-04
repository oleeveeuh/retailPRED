import { describe, expect, it } from 'vitest';

import { validateSnapshotPayload, type LatestSnapshot } from '../api/v2Forecasts';

const VALID: LatestSnapshot = {
  contract_version: '2.1',
  snapshot_kind: 'scheduled_batch_forecasts',
  dataset: 'mrts',
  use_case: 'macro_planning',
  forecast_origin: '2025-12-01',
  model_name: 'global_lightgbm',
  model_version: '2.0.0',
  model_status: 'promoted_point_champion',
  uncertainty_status: 'research_only_not_calibrated',
  is_live: false,
  pipeline_run_id: null,
  freshness: {
    status: 'current',
    generated_at: '2026-01-14T16:00:00+00:00',
    forecast_vintage_time: '2026-01-14T15:30:00+00:00',
    data_cutoff: '2025-12-01',
    next_scheduled_run: null,
    max_age_hours: 336,
  },
  series: [
    {
      series_id: 'retail_total',
      points: [
        {
          horizon: 1,
          target_time: '2026-01-01',
          prediction: 1000,
          lower: null,
          upper: null,
          seasonal_naive_prediction: 990,
          seasonal_naive_source_time: '2025-01-01',
          seasonal_naive_source_vintage_time: '2025-02-14T14:00:00+00:00',
          seasonal_naive_status: 'available',
          seasonal_naive_unavailable_reason: null,
          evaluation_actual: null,
          evaluation_actual_vintage_time: null,
          model_absolute_error: null,
          seasonal_naive_absolute_error: null,
          model_beat_seasonal_naive: null,
          evaluation_status: null,
        },
      ],
    },
  ],
  provenance: { code_revision: 'a'.repeat(40) },
};

describe('validateSnapshotPayload', () => {
  it('accepts a contract-valid snapshot', () => {
    expect(validateSnapshotPayload(VALID)).toEqual(VALID);
  });

  it('refuses an older contract version', () => {
    const payload = { ...VALID, contract_version: '2.0' };
    expect(() => validateSnapshotPayload(payload)).toThrow(/contract version must be 2.1/i);
  });

  it('refuses a non-boolean is_live flag', () => {
    const payload = { ...VALID, is_live: 'true' };
    expect(() => validateSnapshotPayload(payload)).toThrow(/is_live/i);
  });

  it('refuses an empty series list', () => {
    const payload = { ...VALID, series: [] };
    expect(() => validateSnapshotPayload(payload)).toThrow(/no series/i);
  });

  it('refuses a series with no points', () => {
    const payload = {
      ...VALID,
      series: [{ series_id: 'retail_total', points: [] }],
    };
    expect(() => validateSnapshotPayload(payload)).toThrow(/no points/i);
  });

  it('refuses an invalid seasonal-naive status', () => {
    const payload = JSON.parse(JSON.stringify(VALID)) as LatestSnapshot;
    payload.series[0].points[0].seasonal_naive_status = 'maybe';
    expect(() => validateSnapshotPayload(payload)).toThrow(/seasonal-naive status/i);
  });

  it('refuses unavailable benchmark values', () => {
    const payload = JSON.parse(JSON.stringify(VALID)) as LatestSnapshot;
    payload.series[0].points[0].seasonal_naive_status = 'unavailable';
    expect(() => validateSnapshotPayload(payload)).toThrow(/unavailable but carries a value/i);
  });

  it('refuses non-object payloads', () => {
    expect(() => validateSnapshotPayload('nope')).toThrow(/not a JSON object/i);
  });
});
