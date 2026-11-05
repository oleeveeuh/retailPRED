import { describe, expect, it } from 'vitest';

import { gateCandidates } from '../content/lab';
import { evaluateCandidate, evaluateCriterion } from '../lib/labGate';
import policyTauSweep from '../../public/data/lab/policy_tau_sweep.json';
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
