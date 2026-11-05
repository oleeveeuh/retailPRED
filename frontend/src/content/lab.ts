/** Governed content for the #/lab interactive page.
 *
 * Every number here is copied from a named frozen artifact (documented per
 * candidate in `source`). The browser recomputes only the verdicts — the
 * same arithmetic the predeclared gates ran — never a forecast. Nothing on
 * this page is served, hosted, or live.
 */

export type Criterion =
  | {
      kind: 'improves_mean';
      name: string;
      detail: string;
      candidateWape: number;
      baselineWape: number;
    }
  | {
      kind: 'min_categories';
      name: string;
      detail: string;
      improved: number;
      total: number;
      required: number;
    }
  | {
      kind: 'max_regression_pp';
      name: string;
      detail: string;
      worstRegressionPp: number;
      thresholdPp: number;
    }
  | {
      kind: 'beats_references';
      name: string;
      detail: string;
      comparisons: { reference: string; candidateWape: number; referenceWape: number }[];
    }
  | {
      kind: 'below_threshold';
      name: string;
      detail: string;
      value: number;
      limit: number;
      lowerIsBetter: boolean;
    };

export interface GateCandidate {
  id: string;
  model: string;
  dataset: string;
  window: string;
  documentedVerdict: 'refused' | 'rejected';
  documentedSummary: string;
  source: { label: string; href: string };
  criteria: Criterion[];
}

const VINTAGE_GATE_DOC =
  'https://github.com/oleeveeuh/retailPRED/blob/main/docs/v2/M3_VINTAGE_BACKTEST.md';
const SATELLITE_POLICY =
  'https://github.com/oleeveeuh/retailPRED-satellite/blob/main/POLICY_FINDINGS.md';
const SATELLITE_FINDINGS =
  'https://github.com/oleeveeuh/retailPRED-satellite/blob/main/FINDINGS.md';
const CHRONOS_DOC =
  'https://github.com/oleeveeuh/retailPRED/blob/main/docs/v2/M4_M5_CHRONOS2.md';

export const gateCandidates: GateCandidate[] = [
  {
    id: 'lightgbm_vintage',
    model: 'Global LightGBM',
    dataset: 'MRTS · 11 categories',
    window: 'full vintage grid 2015-01 → 2025-12 · 1,408 folds',
    documentedVerdict: 'refused',
    documentedSummary:
      'Refused on all three criteria. The serving decision stays fail-closed and the scheduled pipeline continues to refuse scoring.',
    source: { label: 'M3_VINTAGE_BACKTEST.md', href: VINTAGE_GATE_DOC },
    criteria: [
      {
        kind: 'improves_mean',
        name: 'c1 · mean category WAPE improves over seasonal naive',
        detail: '8.49% vs 8.16% mean-category WAPE on identical origins.',
        candidateWape: 0.0849,
        baselineWape: 0.0816,
      },
      {
        kind: 'min_categories',
        name: 'c2 · WAPE improves in at least 10 of 11 categories',
        detail: 'Improved 4 of 11 (building materials, food & beverage, gasoline, general merchandise).',
        improved: 4,
        total: 11,
        required: 10,
      },
      {
        kind: 'max_regression_pp',
        name: 'c3 · any regressing category stays within 2pp of naive',
        detail: 'Worst regression: clothing −4.02pp (also furniture −2.32pp, electronics −2.10pp).',
        worstRegressionPp: 4.02,
        thresholdPp: 2.0,
      },
    ],
  },
  {
    id: 'theta_holdout',
    model: 'Theta method',
    dataset: 'MRTS · 11 categories',
    window: 'holdout 2023-01 → 2025-03 · 8 origins · single frozen run',
    documentedVerdict: 'refused',
    documentedSummary:
      'Criterion 1 passes (+0.17pp mean-category), criterion 3 passes (worst regression clothing −1.73pp), criterion 2 fails at 7/11 — the gate does not pass.',
    source: { label: 'satellite FINDINGS.md §3', href: SATELLITE_FINDINGS },
    criteria: [
      {
        kind: 'improves_mean',
        name: 'c1 · mean category WAPE improves over seasonal naive',
        detail: '4.98% vs 5.15% on the calm-regime holdout.',
        candidateWape: 0.0498,
        baselineWape: 0.0515,
      },
      {
        kind: 'min_categories',
        name: 'c2 · WAPE improves in at least 10 of 11 categories',
        detail: 'Improved 7 of 11 — loses retail total (−1.79pp) and clothing (−1.73pp).',
        improved: 7,
        total: 11,
        required: 10,
      },
      {
        kind: 'max_regression_pp',
        name: 'c3 · any regressing category stays within 2pp of naive',
        detail: 'Worst regression: clothing −1.73pp, inside the 2pp band.',
        worstRegressionPp: 1.73,
        thresholdPp: 2.0,
      },
    ],
  },
  {
    id: 'regime_switch_policy',
    model: 'Regime-switch policy',
    dataset: 'MRTS · 11 categories · n=99 pairs',
    window: 'report 2021-01 → 2022-10 · single frozen evaluation',
    documentedVerdict: 'refused',
    documentedSummary:
      'Passes the 10/11 count shape and guardrail 1, and beats naive and the blend — but pure theta (7.02%) beats the policy (7.43%) with its whole delta CI below zero. Guardrail 2 (beat everything) fails, so no gate document is issued.',
    source: { label: 'satellite POLICY_FINDINGS.md', href: SATELLITE_POLICY },
    criteria: [
      {
        kind: 'improves_mean',
        name: 'guardrail 1 · bounded downside on CHOOSE (out-of-fold)',
        detail: 'Policy 11.71% vs naive 13.30% — it does not lose to naive; it wins by 1.59pp.',
        candidateWape: 0.1171,
        baselineWape: 0.133,
      },
      {
        kind: 'min_categories',
        name: 'primary · WAPE beats naive in at least 10 of 11 categories',
        detail: '10 of 11 on the report window (bootstrap CI 6–11); loses sporting goods.',
        improved: 10,
        total: 11,
        required: 10,
      },
      {
        kind: 'beats_references',
        name: 'guardrail 2 · mean-category WAPE beats naive AND theta AND the blend',
        detail:
          'Policy 7.43%: beats naive (10.51%) ✓ and the 0.5 blend (7.77%) ✓, but pure theta is 7.02% — the entire theta-vs-policy delta CI sits below zero.',
        comparisons: [
          { reference: 'seasonal naive', candidateWape: 0.0743, referenceWape: 0.1051 },
          { reference: 'pure theta', candidateWape: 0.0743, referenceWape: 0.0702 },
          { reference: 'fixed blend 0.5', candidateWape: 0.0743, referenceWape: 0.0777 },
        ],
      },
    ],
  },
  {
    id: 'chronos_2_zero_shot',
    model: 'Chronos-2 (zero-shot)',
    dataset: 'M5 · full 12-level hierarchy',
    window: 'official evaluation window 2016-04-25 → 2016-05-22',
    documentedVerdict: 'rejected',
    documentedSummary:
      'Zero-shot Chronos-2 scored 1.937 WRMSSE on the full hierarchy — more than double the promoted champion. Rejected by its own predeclared bar.',
    source: { label: 'M4_M5_CHRONOS2.md', href: CHRONOS_DOC },
    criteria: [
      {
        kind: 'below_threshold',
        name: 'c1 · full-hierarchy WRMSSE beats the promoted champion',
        detail: '1.937 vs the champion’s 0.8133 — not competitive zero-shot.',
        value: 1.937,
        limit: 0.8133,
        lowerIsBetter: true,
      },
    ],
  },
];

export const MODEL_COLORS: Record<string, string> = {
  seasonal_naive: '#343330',
  global_lightgbm: '#a38a1c',
  theta_method: '#a43b36',
  regime_switch_policy: '#0d5a8a',
};

export const MODEL_LABELS: Record<string, string> = {
  seasonal_naive: 'naive',
  global_lightgbm: 'LightGBM',
  theta_method: 'theta',
  regime_switch_policy: 'policy',
};

export const WINDOW_LABELS: Record<string, string> = {
  exploration_2017_2022: 'exploration 2017-10 → 2022-10',
  report_2021_2022: 'report 2021-01 → 2022-10',
};

export const SERIES_LABELS: Record<string, string> = {
  building_material_and_garden: 'building materials',
  clothing_and_accessories: 'clothing',
  electronics_and_appliances: 'electronics',
  food_and_beverage: 'food & beverage',
  furniture_and_home_furnishings: 'furniture',
  gasoline_stations: 'gasoline stations',
  general_merchandise: 'general merchandise',
  health_and_personal_care: 'health & personal care',
  motor_vehicle_and_parts: 'motor vehicles & parts',
  retail_total: 'retail total',
  sporting_goods_and_hobby: 'sporting goods',
};

export const LAB_NOTE =
  'Every component on this page recomputes display state in your browser from frozen, ' +
  'provenance-carrying artifacts. Nothing here is a served model, a live forecast, or an ' +
  'API call — the serving decision remains fail-closed.';
