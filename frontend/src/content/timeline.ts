/**
 * The execution record: the real research story from the v1 scoreboard
 * through the vintage-gate refusal to the satellite investigation and the
 * regime-classifier horizon. Every number mirrors the governed evidence
 * ledger (docs/v2/*) and the satellite findings — this module adds no
 * claims of its own.
 *
 * The satellite repository (retailPRED-satellite) is public; satellite-phase
 * entries link their evidence doc via `satelliteDoc` on the satellite repo.
 */

export const REPO_URL = 'https://github.com/oleeveeuh/retailPRED';

export const SATELLITE_REPO_URL: string | null =
  'https://github.com/oleeveeuh/retailPRED-satellite';

/** Human label per satellite evidence doc, used by the timeline links. */
export const SATELLITE_DOC_LABELS: Record<string, string> = {
  'FINDINGS.md': 'satellite findings',
  'ERROR_ANALYSIS.md': 'failure analysis',
  'H5_ALTERNATIVES.md': 'H5 alternatives',
  'PROMPT_REGIME_CLASSIFIER.md': 'regime classifier prompt',
};

const repoDoc = (docPath: string) => `${REPO_URL}/blob/main/${docPath}`;

export interface TimelineLink {
  label: string;
  href: string;
}

export interface TimelineEntry {
  id: string;
  phase: string;
  headline: string;
  detail: string;
  /** Links to public evidence. Satellite phases stay empty until the repo is pushed. */
  links: TimelineLink[];
  /** Satellite-repo evidence doc; rendered only while SATELLITE_REPO_URL is set. */
  satelliteDoc?: string;
}

/** All links for an entry: public links always; satellite links once the repo is public. */
export const effectiveLinks = (entry: TimelineEntry): TimelineLink[] =>
  entry.satelliteDoc && SATELLITE_REPO_URL
    ? [
        ...entry.links,
        {
          label: SATELLITE_DOC_LABELS[entry.satelliteDoc] ?? 'satellite findings',
          href: `${SATELLITE_REPO_URL}/blob/main/${entry.satelliteDoc}`,
        },
      ]
    : entry.links;

export const timeline: TimelineEntry[] = [
  {
    id: 'v1-gcp-airflow',
    phase: 'phase 01 · v1',
    headline: 'GCP + Airflow, and a scoreboard that flattered',
    detail:
      'Recurring forecasts on the undergrad stack. The scoreboard said feature engineering worked: LightGBM 9.76% vs naive 14.11%.',
    links: [
      {
        label: 'v1 baseline document',
        href: repoDoc('docs/v2/V1_BASELINE.md'),
      },
    ],
  },
  {
    id: 'v2-rebuild',
    phase: 'phase 02 · v2 rebuild',
    headline: 'The protocol, corrected',
    detail:
      'Under point-in-time evaluation, seasonal naive’s honest value is 6.35%. The entire v1 gap was protocol artifact — weekly pseudo-replication plus inclusive target features, i.e. backtest/serve skew.',
    links: [
      {
        label: 'leakage report',
        href: repoDoc('docs/v2/LEAKAGE_REPORT.md'),
      },
    ],
  },
  {
    id: 'revised-history',
    phase: 'phase 03 · revised history',
    headline: 'Engineered features genuinely win',
    detail:
      'On revised history the gap is real: global LightGBM 4.49% vs naive 6.35% mean-category WAPE, better in all 11 categories.',
    links: [
      {
        label: 'MRTS benchmark document',
        href: repoDoc('docs/v2/M3_LIGHTGBM.md'),
      },
    ],
  },
  {
    id: 'vintage-gate',
    phase: 'phase 04 · the vintage gate',
    headline: 'Real-time conditions erase the edge',
    detail:
      'Across 106 ALFRED vintages the predeclared gate measured 8.49% vs 8.16% — and refused the champion. The pipeline kept serving nothing.',
    links: [
      {
        label: 'vintage gate document',
        href: repoDoc('docs/v2/M3_VINTAGE_BACKTEST.md'),
      },
      {
        label: 'execution ledger',
        href: repoDoc('docs/v2/DATABRICKS_RUN.md'),
      },
    ],
  },
  {
    id: 'satellite-phase-0',
    phase: 'phase 05 · satellite, phase 0',
    headline: 'The gate, reproduced fold-for-fold',
    detail:
      'A dedicated investigation re-ran the failed gate with pinned dependency versions: naive diff 0.0, fold for fold. Nothing was blamed on a broken pipeline.',
    links: [],
    satelliteDoc: 'FINDINGS.md',
  },
  {
    id: 'satellite-phase-1',
    phase: 'phase 06 · satellite, phase 1',
    headline: 'Where the failure lives',
    detail:
      'The failure is equal-weighted, concentrated at the h=12 recursion and high-revision categories (Spearman −0.755). Staleness ruled out.',
    links: [],
    satelliteDoc: 'ERROR_ANALYSIS.md',
  },
  {
    id: 'h1-h4',
    phase: 'phase 07 · H1–H4',
    headline: 'Four candidates, four refutations',
    detail:
      'The preliminary-noise drop and revision-aware features failed; three transformations refuted — the log-ratio target is load-bearing. The naive combination helped (+1.18pp) but wasn’t gate-shaped.',
    links: [],
    satelliteDoc: 'FINDINGS.md',
  },
  {
    id: 'h5-theta',
    phase: 'phase 08 · H5, theta',
    headline: 'The breakthrough',
    detail:
      'Theta — a per-series method from 2000 — beat naive in 10/11 categories, +3.48pp on the report window.',
    links: [],
    satelliteDoc: 'H5_ALTERNATIVES.md',
  },
  {
    id: 'holdout',
    phase: 'phase 09 · holdout, run once',
    headline: 'Dominance was regime-specific',
    detail:
      'The single predeclared holdout run: +0.17pp, 7/11 categories. Criterion 2 fails — theta’s dominance was regime-specific. Naive stands.',
    links: [],
    satelliteDoc: 'FINDINGS.md',
  },
  {
    id: 'challenge-audits',
    phase: 'phase 10 · challenge audits',
    headline: 'The verdict under other metrics',
    detail:
      'MAPE, relative MAE, and win-rate agree with the gate. The win matrix: naive wins 0/11 outright; theta+naive beats naive 11/11.',
    links: [],
    satelliteDoc: 'FINDINGS.md',
  },
  {
    id: 'next-regime-classifier',
    phase: 'phase 11 · next',
    headline: 'A regime classifier',
    detail:
      'Trend strength, volatility, revision churn, and lagged performance route each forecast origin to naive or theta — worst case ties naive.',
    links: [],
    satelliteDoc: 'PROMPT_REGIME_CLASSIFIER.md',
  },
];
