import type { Criterion, GateCandidate } from '../content/lab';

export interface CriterionResult {
  criterion: Criterion;
  passed: boolean;
}

export interface GateResult {
  candidate: GateCandidate;
  results: CriterionResult[];
  passed: boolean;
}

/** Evaluate one criterion with the same arithmetic the predeclared gates ran. */
export function evaluateCriterion(criterion: Criterion): boolean {
  switch (criterion.kind) {
    case 'improves_mean':
      return criterion.candidateWape < criterion.baselineWape;
    case 'min_categories':
      return criterion.improved >= criterion.required;
    case 'max_regression_pp':
      return Math.abs(criterion.worstRegressionPp) <= criterion.thresholdPp;
    case 'beats_references':
      return criterion.comparisons.every((c) => c.candidateWape < c.referenceWape);
    case 'below_threshold':
      return criterion.lowerIsBetter
        ? criterion.value < criterion.limit
        : criterion.value > criterion.limit;
  }
}

/** Run every criterion for a candidate; the gate passes only if all do. */
export function evaluateCandidate(candidate: GateCandidate): GateResult {
  const results = candidate.criteria.map((criterion) => ({
    criterion,
    passed: evaluateCriterion(criterion),
  }));
  return { candidate, results, passed: results.every((r) => r.passed) };
}

/** The documented human verdict, as a string: only 'refused' | 'rejected'. */
export function verdictLabel(candidate: GateCandidate): string {
  return candidate.documentedVerdict === 'refused' ? 'GATE REFUSED' : 'REJECTED';
}
