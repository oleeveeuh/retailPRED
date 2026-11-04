/**
 * Curated showcase content. Every number mirrors the governed evidence
 * ledger (docs/v2/DATABRICKS_RUN.md, M3_VINTAGE_BACKTEST.md, M4_* docs)
 * and the frozen model portfolio — this module adds no claims of its own.
 */

export type TaskStatus = 'executed' | 'refused-by-gate' | 'awaiting' | 'dormant';

export interface PipelineTask {
  task: string;
  headline: string;
  description: string;
  status: TaskStatus;
  evidence: string;
}

export interface ModelCard {
  dataset: string;
  useCase: string;
  model: string;
  headline: string;
  metricName: string;
  metricValue: string;
  baselineName: string;
  baselineValue: string;
  improvement: string;
  decision: 'promoted-demo' | 'challenger-gate-failed';
  decisionLine: string;
  evidencePath: string;
}

export interface RejectionCard {
  name: string;
  oneLine: string;
  outcome: string;
}

export interface TableauView {
  name: string;
  marks: string;
  oneLine: string;
}

export const hero = {
  eyebrow: 'RetailPRED · v2.0.0',
  title: 'Retail forecasting, with an evidence gate',
  subtitle:
    'A graduate-level rebuild of an undergraduate project: leakage-safe evaluation, governed model promotion on Databricks, and a serving contract that stays dark until a model earns the right to forecast.',
  chips: [
    { label: 'Databricks pipeline', value: 'executed', tone: 'mint' as const },
    { label: 'Vintage gate', value: 'refused the challenger', tone: 'amber' as const },
    { label: 'Tableau', value: '5 governed views', tone: 'mint' as const },
    { label: 'Serving', value: 'fail-closed', tone: 'slate' as const },
  ],
};

export const pipelineTasks: PipelineTask[] = [
  {
    task: '1 · Ingest',
    headline: 'The Databricks pull',
    description:
      'A scheduled task pulls the latest Census MRTS release into immutable bronze rows and a versioned silver table — every value stamped with its retrieval vintage, replays are no-ops.',
    status: 'executed',
    evidence: '2,200 monthly observations merged · cutoff Aug 2026',
  },
  {
    task: '2 · Score',
    headline: 'Governed model scoring',
    description:
      'Refits the promoted champion on point-in-time gold and freezes a seasonal-naive benchmark beside every forecast — but only when the frozen portfolio decision authorizes it.',
    status: 'refused-by-gate',
    evidence: 'challenger ≠ champion · scoring stays dark',
  },
  {
    task: '3 · Evaluate',
    headline: 'Realized-error tracking',
    description:
      'When published actuals arrive, every stored forecast vintage is scored against them — first-release and revised evaluations kept as separate immutable rows.',
    status: 'awaiting',
    evidence: 'awaits the first matured forecast',
  },
  {
    task: '4 · Publish',
    headline: 'Snapshot to the live site',
    description:
      'Commits a contract-validated JSON snapshot to a machine-written branch; GitHub Pages validates and serves it to the evidence page on this site.',
    status: 'dormant',
    evidence: 'activates automatically after a gate pass',
  },
];

export const gateStory = {
  title: 'The gate refused its own model',
  body: 'On revised history, global LightGBM beat seasonal naive by 29%. So we asked the harder question — what could each model know at forecast time? — on 106 real ALFRED vintages. The edge disappeared: 8.49% vs 8.16% mean category WAPE, only 4 of 11 categories improved. The predeclared gate failed it, and the pipeline kept serving nothing.',
  metrics: [
    { label: 'Real-time naive', value: '8.16%' },
    { label: 'Real-time LightGBM', value: '8.49%' },
    { label: 'Revised-history naive', value: '6.35%' },
    { label: 'Folds evaluated', value: '1,408' },
  ],
  links: [
    { label: 'Vintage gate document', href: 'https://github.com/oleeveeuh/retailPRED/blob/main/docs/v2/M3_VINTAGE_BACKTEST.md' },
    { label: 'Execution ledger', href: 'https://github.com/oleeveeuh/retailPRED/blob/main/docs/v2/DATABRICKS_RUN.md' },
  ],
};

export const modelCards: ModelCard[] = [
  {
    dataset: 'M5 · operational demand',
    useCase: '30,490 item/store daily series · 42,840-node hierarchy',
    model: 'Global LightGBM',
    headline: 'One pooled-horizon model across the full hierarchy, reconciled bottom-up.',
    metricName: 'WRMSSE',
    metricValue: '0.8133',
    baselineName: 'Seasonal naive',
    baselineValue: '0.8697',
    improvement: '6.48% better · won all 5 rolling windows',
    decision: 'promoted-demo',
    decisionLine: 'promoted — offline demo evidence served on this site',
    evidencePath:
      'https://github.com/oleeveeuh/retailPRED/blob/main/docs/v2/M4_M5_LIGHTGBM.md',
  },
  {
    dataset: 'MRTS · macro planning',
    useCase: '11 U.S. retail categories, monthly, release-lagged and revised',
    model: 'Global LightGBM',
    headline: 'Scale-free seasonal log-ratio training across all categories at once.',
    metricName: 'Mean category WAPE (revised)',
    metricValue: '4.49%',
    baselineName: 'Seasonal naive (revised)',
    baselineValue: '6.35%',
    improvement: '29% better on revised history · all 11 categories',
    decision: 'challenger-gate-failed',
    decisionLine: 'not promoted — failed the predeclared point-in-time gate 8.49% vs 8.16%',
    evidencePath:
      'https://github.com/oleeveeuh/retailPRED/blob/main/docs/v2/M3_VINTAGE_BACKTEST.md',
  },
];

export const rejections: RejectionCard[] = [
  {
    name: 'Chronos-2 (zero-shot)',
    oneLine: 'Foundation model advanced from a 600-series subset, then met the full hierarchy.',
    outcome: 'rejected — 1.937 WRMSSE, 32% aggregate underforecast',
  },
  {
    name: 'NHITS',
    oneLine: 'Bounded neural feasibility on a predeclared 600-series revenue-stratified subset.',
    outcome: 'rejected — trailed subset-trained LightGBM by 10.58%',
  },
  {
    name: 'MinT-style reconciliation',
    oneLine: 'Nonnegative variance-weighted reconciliation restored exact coherence.',
    outcome: 'rejected — failed the all-window stability gate',
  },
  {
    name: 'Conformal intervals',
    oneLine: 'Calibrated on a validation window against an 80% coverage target.',
    outcome: 'rejected — 69.38% coverage on high-volume bottom series',
  },
];

export const tableauViews: TableauView[] = [
  { name: 'Model scorecard', marks: '6 marks', oneLine: 'Every frozen decision with its measured value and baseline.' },
  { name: 'M5 hierarchy', marks: '12 marks', oneLine: 'Accuracy across the official 12-level item/store hierarchy.' },
  { name: 'M5 uncertainty', marks: '47 marks', oneLine: 'Coverage and interval width by level, horizon, and segment.' },
  { name: 'MRTS regimes', marks: '3 marks', oneLine: 'Pre-COVID, disruption, and post-disruption skill split.' },
  { name: 'Data freshness', marks: '2 marks', oneLine: 'Evaluation cutoffs and serving states per dataset.' },
];

export const claimLadder = [
  'implemented',
  'locally verified',
  'executed on Databricks',
  'scheduled',
  'publicly deployed',
];
