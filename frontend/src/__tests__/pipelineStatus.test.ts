import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  fetchRunHistoryFromUrl,
  fetchStatusFromUrl,
  parseRunHistory,
  validateIngestStatusPayload,
  type IngestStatus,
} from '../api/ingestStatus';
import {
  gateChips,
  liveState,
  missingRunCopy,
  statusSummary,
  unavailableRunCopy,
} from '../content/pipelineStatus';

/** Mirrors the verified hosted-run receipt: the only pipeline run ever
 * executed (run 452582046469351, 2026-11-04) — a full fail-closed pass. */
const REFUSAL_RECORD: IngestStatus = {
  schema_version: '2.2-status',
  pipeline_run_id: '452582046469351',
  job_id: '326179734702237',
  pipeline_started_at: '2026-11-04T22:29:51+00:00',
  finished_at: '2026-11-04T22:33:12+00:00',
  ingest: { rows_in_delta: 2200, data_cutoff: '2026-08-01', series_count: 11 },
  gates: {
    ingest: 'success',
    score: {
      status: 'refused',
      model: 'global_lightgbm',
      model_status: 'development_challenger',
      required_action: 'unavailable_until_point_in_time_validation',
    },
    realized: { status: 'skipped', realized_rows: 0 },
    publish: 'declined',
  },
  is_live: false,
  generated_at: '2026-11-04T22:33:12+00:00',
};

describe('validateIngestStatusPayload', () => {
  it('accepts the fail-closed refusal record', () => {
    expect(validateIngestStatusPayload(REFUSAL_RECORD)).toEqual(REFUSAL_RECORD);
  });

  it('refuses an unknown schema version', () => {
    const payload = { ...REFUSAL_RECORD, schema_version: '2.1' };
    expect(() => validateIngestStatusPayload(payload)).toThrow(/schema version must be 2.2-status/i);
  });

  it('refuses a record carrying forecast-claim keys at any depth', () => {
    const smuggled = JSON.parse(JSON.stringify(REFUSAL_RECORD)) as Record<string, unknown>;
    smuggled.prediction = 1000;
    expect(() => validateIngestStatusPayload(smuggled)).toThrow(/forbidden forecast-claim keys/);
    const nested = JSON.parse(JSON.stringify(REFUSAL_RECORD)) as IngestStatus;
    (nested.ingest as unknown as Record<string, unknown>).series = ['retail_total'];
    expect(() => validateIngestStatusPayload(nested)).toThrow(/forbidden forecast-claim keys/);
  });

  it('refuses a premature is_live claim', () => {
    const premature = { ...REFUSAL_RECORD, is_live: true };
    expect(() => validateIngestStatusPayload(premature)).toThrow(
      /is_live without a successful ingest/i,
    );
    const completed = {
      ...REFUSAL_RECORD,
      is_live: true,
      gates: { ...REFUSAL_RECORD.gates, publish: 'published' as const },
    };
    expect(validateIngestStatusPayload(completed).is_live).toBe(true);
  });

  it('refuses a record with no gates', () => {
    const payload = { ...REFUSAL_RECORD, gates: undefined };
    expect(() => validateIngestStatusPayload(payload)).toThrow(/no gates/i);
  });
});

describe('gateChips', () => {
  it('renders the honest refusal verdicts from the record', () => {
    const chips = gateChips(REFUSAL_RECORD);
    expect(chips.map((chip) => chip.key)).toEqual(['ingest', 'score', 'realized', 'publish']);
    const byKey = Object.fromEntries(chips.map((chip) => [chip.key, chip]));
    expect(byKey.ingest.label).toBe('ingest ✓');
    expect(byKey.ingest.detail).toBe('2,200 rows · cutoff 2026-08-01');
    expect(byKey.score.label).toBe('score refused');
    expect(byKey.score.detail).toContain('challenger unauthorized');
    expect(byKey.score.detail).toContain('development_challenger');
    expect(byKey.realized.label).toBe('realized skipped');
    expect(byKey.realized.detail).toBe('nothing matured yet');
    expect(byKey.publish.label).toBe('publish declined');
    expect(byKey.publish.detail).toContain('no gold forecasts');
  });

  it('renders the success-side verdicts without inventing detail', () => {
    const published: IngestStatus = {
      ...REFUSAL_RECORD,
      gates: {
        ingest: 'success',
        score: {
          status: 'authorized',
          model: 'theta',
          model_status: 'promoted_point_champion',
          required_action: 'scheduled_batch_on_databricks',
        },
        realized: { status: 'evaluated', realized_rows: 132 },
        publish: 'skipped-identical',
      },
    };
    const chips = gateChips(published);
    const byKey = Object.fromEntries(chips.map((chip) => [chip.key, chip]));
    expect(byKey.score.label).toBe('score authorized');
    expect(byKey.score.detail).toContain('theta');
    expect(byKey.realized.label).toBe('realized evaluated');
    expect(byKey.realized.detail).toBe('132 immutable rows');
    expect(byKey.publish.label).toBe('publish skipped-identical');
  });

  it('renders a failed ingest without stale numbers', () => {
    const failed: IngestStatus = {
      ...REFUSAL_RECORD,
      ingest: { rows_in_delta: null, data_cutoff: null, series_count: null },
      gates: { ...REFUSAL_RECORD.gates, ingest: 'failed' },
    };
    const chips = gateChips(failed);
    expect(chips[0].label).toBe('ingest failed');
    expect(chips[0].detail).toBe('no merge recorded for this run');
  });
});

describe('the fail-closed framings', () => {
  it('presents a missing record as governance, not breakage', () => {
    expect(missingRunCopy.title).toMatch(/no hosted run recorded yet/i);
    expect(missingRunCopy.body).toMatch(/refuses to fabricate/i);
    expect(missingRunCopy.body).toMatch(/governance/i);
  });

  it('presents an unavailable artifact as refused display, not broken UI', () => {
    expect(unavailableRunCopy.title).toMatch(/unavailable/i);
    expect(unavailableRunCopy.body).toMatch(/contract validation/i);
    expect(unavailableRunCopy.body).toMatch(/machine-written/i);
  });

  it('mirrors the schedule flag in the live-state line', () => {
    expect(liveState(REFUSAL_RECORD)).toBe('scheduled runs: paused');
    expect(liveState({ ...REFUSAL_RECORD, is_live: true })).toBe('scheduled runs: live');
  });

  it('summarizes the record with its own numbers', () => {
    const summary = statusSummary(REFUSAL_RECORD);
    expect(summary.lastRun).toBe('2026-11-04');
    expect(summary.cutoff).toBe('2026-08-01');
    expect(summary.rows).toBe('2,200');
    expect(summary.runId).toBe('452582046469351');
  });
});

describe('run history', () => {
  it('parses every jsonl line as a validated record', () => {
    const second = { ...REFUSAL_RECORD, pipeline_run_id: '999', finished_at: '2026-11-05T22:00:00+00:00' };
    const runs = parseRunHistory(`${JSON.stringify(REFUSAL_RECORD)}\n${JSON.stringify(second)}\n`);
    expect(runs).toHaveLength(2);
    expect(runs[1].pipeline_run_id).toBe('999');
  });

  it('fails closed on a malformed line instead of thinning history', () => {
    const log = `${JSON.stringify(REFUSAL_RECORD)}\nnot-json\n`;
    expect(() => parseRunHistory(log)).toThrow();
    const claim = JSON.parse(JSON.stringify(REFUSAL_RECORD)) as Record<string, unknown>;
    claim.forecasts = [];
    expect(() => parseRunHistory(JSON.stringify(claim))).toThrow(/forbidden forecast-claim keys/);
  });
});

describe('status fetch', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('returns a validated record for a 200 fixture', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(JSON.stringify(REFUSAL_RECORD), { status: 200 })),
    );
    const record = await fetchStatusFromUrl('live/v2-ingest-status.json');
    expect(record.pipeline_run_id).toBe('452582046469351');
  });

  it('maps a 404 to the honest missing-record state', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(null, { status: 404 })));
    await expect(fetchStatusFromUrl('live/v2-ingest-status.json')).rejects.toThrow(
      /no hosted run recorded yet/,
    );
  });

  it('refuses to display an invalid artifact', async () => {
    const broken = JSON.parse(JSON.stringify(REFUSAL_RECORD)) as Record<string, unknown>;
    broken.schema_version = '9.9';
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(JSON.stringify(broken), { status: 200 })),
    );
    await expect(fetchStatusFromUrl('live/v2-ingest-status.json')).rejects.toThrow(
      /refusing to display/i,
    );
  });

  it('maps a run-history 404 to the same honest state', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(null, { status: 404 })));
    await expect(fetchRunHistoryFromUrl('live/runs.jsonl')).rejects.toThrow(
      /no hosted run recorded yet/,
    );
  });
});
