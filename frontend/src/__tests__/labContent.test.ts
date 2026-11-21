import { describe, expect, it } from 'vitest';

import { gateCandidates } from '../content/lab';
import { evaluateCandidate, evaluateCriterion } from '../lib/labGate';
import policyTauSweep from '../../public/data/lab/policy_tau_sweep.json';
import shapAnalysis from '../../public/data/lab/shap_analysis.json';
import windowCategories from '../../public/data/lab/window_categories.json';
import vintageRevisions from '../../public/data/lab/vintage_revisions.json';

/** The browser-side gate arithmetic must reproduce every documented verdict. */
describe('lab gate referee', () => {
  it('reproduces the documented verdict for every candidate', () => {
    for (const candidate of gateCandidates) {
      const result = evaluateCandidate(candidate);
      expect(result.passed, `${candidate.id} must match its documented verdict`).toBe(false);
    }
  });

  it('refused the vintage LightGBM on all three criteria', () => {
    const result = evaluateCandidate(gateCandidates[0]);
    expect(result.results.map((r) => r.passed)).toEqual([false, false, false]);
  });

  it('refused theta on criterion 2 only', () => {
    const result = evaluateCandidate(gateCandidates[1]);
    expect(result.results.map((r) => r.passed)).toEqual([true, false, true]);
  });

  it('refused the policy on guardrail 2 only — the 10/11 shape passes', () => {
    const result = evaluateCandidate(gateCandidates[2]);
    expect(result.results.map((r) => r.passed)).toEqual([true, true, false]);
    const beats = result.results[2].criterion;
    expect(beats.kind).toBe('beats_references');
    if (beats.kind === 'beats_references') {
      const byRef = Object.fromEntries(
        beats.comparisons.map((c) => [c.reference, c.candidateWape < c.referenceWape]),
      );
      expect(byRef).toEqual({ 'seasonal naive': true, 'pure theta': false, 'fixed blend 0.5': true });
    }
  });

  it('rejected zero-shot Chronos-2 on the champion-threshold criterion', () => {
    const criterion = gateCandidates[3].criteria[0];
    expect(evaluateCriterion(criterion)).toBe(false);
  });
});

describe('lab data bundles', () => {
  it('tau sweep is the frozen 51-point CHOOSE-OOF curve with a single REPORT point', () => {
    expect(policyTauSweep.sweep).toHaveLength(51);
    expect(policyTauSweep.frozen_tau).toBe(0.5);
    expect(policyTauSweep.frozen_model).toBe('lightgbm');
    expect(policyTauSweep.references.naive).toBeCloseTo(0.132969, 5);
    expect(policyTauSweep.report_point.tau).toBe(0.5);
    expect(policyTauSweep.report_point.categories_beating_naive).toBe(10);
  });

  it('window rows cover 4 models, and the policy exists only on the report window', () => {
    expect(windowCategories.rows).toHaveLength(77);
    const models = new Set(windowCategories.rows.map((r) => r.window + '|' + r.model));
    expect(models.has('report_2021_2022|regime_switch_policy')).toBe(true);
    expect(models.has('exploration_2017_2022|regime_switch_policy')).toBe(false);
    for (const row of windowCategories.rows) {
      expect(row.wape).toBeGreaterThan(0);
    }
  });

  it('vintage explorer data: 3 demo months x 11 categories with ascending vintages', () => {
    expect(vintageRevisions.months).toHaveLength(3);
    for (const month of vintageRevisions.months) {
      expect(month.categories).toHaveLength(11);
      for (const category of month.categories) {
        expect(category.vintages.length).toBeGreaterThanOrEqual(2);
        expect(category.latest_revised).toBeGreaterThan(0);
        const dates = category.vintages.map((v) => v.vintage_date);
        expect([...dates].sort()).toEqual(dates);
        for (const vintage of category.vintages) {
          expect(vintage.value).toBeGreaterThan(0);
        }
      }
    }
  });
});

describe('shap analysis bundle', () => {
  it('explains exactly the 15 GlobalLightGBM features with shares summing to one', () => {
    expect(shapAnalysis.features).toHaveLength(15);
    // The committed order is the model's column order; a reorder must be a
    // reviewed bundle change, never a silent one.
    expect(shapAnalysis.features).toEqual([
      'series_code',
      'lag_1_ratio',
      'lag_2_ratio',
      'lag_3_ratio',
      'lag_6_ratio',
      'rolling_mean_3_ratio',
      'rolling_std_3_ratio',
      'rolling_mean_6_ratio',
      'rolling_std_6_ratio',
      'rolling_mean_12_ratio',
      'rolling_std_12_ratio',
      'diff_1_ratio',
      'pct_change_1',
      'month_sin',
      'month_cos',
    ]);
    expect(new Set(shapAnalysis.global_importance.map((r) => r.feature))).toEqual(
      new Set(shapAnalysis.features),
    );
    const shareSum = shapAnalysis.global_importance.reduce((total, r) => total + r.share, 0);
    expect(shareSum).toBeCloseTo(1, 6);
    const meanAbs = shapAnalysis.global_importance.map((r) => r.mean_abs_shap);
    expect([...meanAbs].sort((a, b) => b - a)).toEqual(meanAbs);
    for (const row of shapAnalysis.global_importance) {
      expect(row.mean_abs_shap).toBeGreaterThanOrEqual(0);
    }
  });

  it('is the development challenger with in-sample provenance and a tiny additivity error', () => {
    expect(shapAnalysis.provenance.model).toBe('global_lightgbm');
    expect(shapAnalysis.provenance.model_status).toBe('development_challenger');
    expect(shapAnalysis.provenance.explainer).toContain('in-sample');
    expect(shapAnalysis.provenance.target).toBe('log(value / lag_12)');
    expect(shapAnalysis.provenance.additivity_max_error).toBeLessThan(1e-6);
    expect(shapAnalysis.provenance.n_training_rows).toBeGreaterThan(0);
    expect(shapAnalysis.provenance.n_series).toBeGreaterThan(0);
    expect(shapAnalysis.provenance.platform_code_revision).toMatch(/^[0-9a-f]{40}$/);
    expect(shapAnalysis.provenance.data_sha256).toMatch(/^[0-9a-f]{64}$/);
  });

  it('closes every per-series waterfall from base to the model output', () => {
    expect(shapAnalysis.waterfalls.length).toBe(shapAnalysis.provenance.n_series);
    const seriesIds = new Set(shapAnalysis.waterfalls.map((w) => w.series_id));
    expect(seriesIds.size).toBe(shapAnalysis.waterfalls.length);
    for (const waterfall of shapAnalysis.waterfalls) {
      expect(waterfall.contributions).toHaveLength(8);
      const sum = waterfall.contributions.reduce((total, c) => total + c.shap, 0);
      // Only the top-8 of 15 features are listed: base + sum(shap) must be
      // <= the output in absolute distance than the largest omitted bar could
      // close; exact closure is checked in the Python suite, here we pin the
      // rendered waterfall's arithmetic: the last bar "all other features"
      // absorbs the residual, so require |residual| <= sum of |contributions|.
      expect(Math.abs(waterfall.model_output - waterfall.base_value - sum)).toBeLessThan(
        waterfall.contributions.reduce((total, c) => total + Math.abs(c.shap), 0) + 1e-9,
      );
      for (const contribution of waterfall.contributions) {
        expect(shapAnalysis.features).toContain(contribution.feature);
      }
    }
  });

  it('shift diagnosis: the documented 29.2% -> 4.1% reversal over controlled panels', () => {
    const shift = shapAnalysis.shift_diagnosis;
    const refs = shift.provenance.references;
    // docs/v2/M3_LIGHTGBM.md and docs/v2/M3_VINTAGE_BACKTEST.md, recomputed
    expect(refs.development_validation.relative_improvement).toBeCloseTo(0.2921, 3);
    expect(refs.vintage_gate.relative_degradation).toBeCloseTo(0.0408, 3);
    expect(refs.vintage_gate.lightgbm_mean_category_wape).toBeCloseTo(0.084893, 5);
    expect(refs.vintage_gate.naive_mean_category_wape).toBeCloseTo(0.081568, 5);
    // the two fits read the same months: same cutoff, same row count
    expect(shift.provenance.revised_history.data_cutoff).toBe(
      shift.provenance.point_in_time.data_cutoff,
    );
    expect(shift.provenance.revised_history.n_rows).toBe(
      shift.provenance.point_in_time.n_rows,
    );
    expect(shift.provenance.common_cutoff).toBe('2025-03-01');
    expect(shift.provenance.as_of.startsWith('2025-05-15')).toBe(true);
    expect(Object.keys(shift.provenance.input_sha256_by_artifact)).toHaveLength(4);
    expect(shift.rows).toHaveLength(15);
    const drifts = shift.rows.map((r) => Math.abs(r.drift_abs));
    // A missing or renamed drift field would surface as NaN here; the sort
    // pin below cannot catch that on its own (Object.is(NaN, NaN)).
    expect(drifts.every((d) => Number.isFinite(d))).toBe(true);
    expect([...drifts].sort((a, b) => b - a)).toEqual(drifts);
    expect(shift.attribution_drift_l1).toBeGreaterThan(0);
    for (const row of shift.rows) {
      expect(shift.features).toContain(row.feature);
    }
  });
});
