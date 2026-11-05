/**
 * The data-exploration story, chapter by chapter — the long-form companion
 * to the execution timeline on the showcase page. Every number is copied
 * from a named frozen artifact:
 *
 *  - Full-grid gate: docs/v2/M3_VINTAGE_BACKTEST.md (naive 8.16% /
 *    LightGBM 8.49%, 4 of 11 categories) — a different measurement from
 *    the revised-history benchmark in docs/v2/M3_LIGHTGBM.md (4.49% vs
 *    6.35%). The two are never mixed.
 *  - Window research: data/v2/evidence/mrts_research_windows/
 *    mrts_window_summary.json (9 rows, frozen copy of the satellite's
 *    consolidated window summary).
 *  - Policy detail: the satellite's results/policy_report.json
 *    (n=99 REPORT pairs, origin-block bootstrap CIs).
 *
 * This module adds no claims of its own; the satellite is public at
 * github.com/oleeveeuh/retailPRED-satellite and its files are linked
 * directly.
 */

export const SATELLITE_URL = 'https://github.com/oleeveeuh/retailPRED-satellite';

const platformDoc = (docPath: string) =>
  `https://github.com/oleeveeuh/retailPRED/blob/main/${docPath}`;
const satelliteDoc = (docPath: string) => `${SATELLITE_URL}/blob/main/${docPath}`;

export interface StoryEvidence {
  label: string;
  href: string;
}

export interface StoryFigure {
  /** Path as served, including the deployed base (import.meta.env.BASE_URL). */
  src: string;
  caption: string;
  alt: string;
}

export interface StoryChapter {
  id: string;
  chapter: number;
  kicker: string;
  title: string;
  lede: string;
  body: string[];
  evidence: StoryEvidence;
  /** Themed research figures, inserted after the body (see Part 2 of the build). */
  figures?: StoryFigure[];
}

export const storyChapters: StoryChapter[] = [
  {
    id: 'the-gate-that-said-no',
    chapter: 1,
    kicker: 'chapter 01 · 2026-11-04',
    title: 'the gate that said no',
    lede:
      'A model that beat seasonal naive by 29% on revised history met the one test that matters — and lost.',
    body: [
      'On revised history, the global LightGBM looked genuinely good: 4.49% mean-category WAPE against the seasonal naive’s 6.35%, better in all 11 Census retail categories. That result earned it a shot at the real thing. The vintage gate had been written before the measurement, in the style I want to keep: fix the criteria first, then let the data answer.',
      'The gate rebuilt every training set from 106 real ALFRED vintages — only what was publishable at each forecast origin — and scored 1,408 folds. Real time flipped the result: 8.49% vs the naive’s 8.16%, with LightGBM improving just 4 of 11 categories. It failed on all three predeclared criteria, and the serving decision stayed fail-closed. No champion, no hand-waving — the pipeline kept serving nothing, which is what an honest gate is for.',
    ],
    evidence: {
      label: 'vintage gate document (M3_VINTAGE_BACKTEST.md)',
      href: platformDoc('docs/v2/M3_VINTAGE_BACKTEST.md'),
    },
  },
  {
    id: 'where-the-failure-lives',
    chapter: 2,
    kicker: 'chapter 02 · satellite, phase 1',
    title: 'where the failure lives',
    lede:
      'Before rebuilding anything, I wanted to know exactly where the edge died — so a satellite repo re-ran the gate, fold for fold, and took it apart.',
    body: [
      'First, reproduction: the satellite re-ran the failed gate with pinned versions and got the naive back to the digit — max diff 0.0. Nothing was wrong with the platform. The failure belonged to the model.',
      'The decomposition that followed changed how I read every metric in this project. Pooled across folds (volume-weighted), the LightGBM actually beat naive even in real time — 5.87% vs 6.54% on exploration origins. The gate metric weights categories equally, and the model lost exactly there: it won big, quiet categories and lost small, volatile ones. By horizon, it won h=1 and lost h=12 badly — the recursive rollout feeds its own drift back eleven times, and clothing degrades monotonically from +0.6pp to −8.5pp. By regime, every point of advantage came from the pandemic disruption. And the category regressions ranked by revision magnitude (Spearman −0.755): loudly-revised categories were where it bled. Training staleness, the first suspect, measured constant and was ruled out.',
    ],
    evidence: {
      label: 'satellite forensics (FINDINGS.md, ERROR_ANALYSIS.md)',
      href: satelliteDoc('FINDINGS.md'),
    },
  },
  {
    id: 'the-hypothesis-graveyard',
    chapter: 3,
    kicker: 'chapter 03 · satellite, phase 2',
    title: 'the hypothesis graveyard',
    lede:
      'Four predeclared hypotheses about why the model failed in real time. Four refutations — and one load-bearing accident.',
    body: [
      'H1: drop the preliminary training rows, the values most likely to be revised later. Refuted — trimming never helped (−0.41pp on the report window). The poison, if any, was not in the fit-row targets. H2: give the model revision metadata — counts, magnitudes, value ages. Refuted (−0.68pp); an L1 tree ensemble could not turn reliability notes into sharper targets.',
      'H4: change the target transformation. Every alternative lost: year-over-year differencing −0.88pp, STL-detrended inputs −1.06pp, and a plain log-level target collapsing outright at −19.07pp. That last one was the most useful negative result in the project: the scale-free log(value/lag₁₂) target is load-bearing, and nothing replaces it. H3, shrinking the model toward naive, did help (+1.18pp, 7/11 categories) — real, but not gate-shaped, and it pointed away from patching LightGBM toward looking for a better method entirely.',
    ],
    evidence: {
      label: 'hypothesis table (satellite FINDINGS.md §2)',
      href: satelliteDoc('FINDINGS.md'),
    },
  },
  {
    id: 'theta',
    chapter: 4,
    kicker: 'chapter 04 · satellite, H5',
    title: 'theta',
    lede:
      'The method that finally beat naive in real time was twenty-five years old, per-series, and had almost no parameters.',
    body: [
      'H5 was the hypothesis I almost skipped: the theta method (Assimakopoulos & Nikolopoulos 2000) — decompose a series into short- and long-term components, extrapolate the short one with a single in-sample smoothing parameter. No learned features, no training panel, nothing to leak.',
      'It won. On exploration origins (2017-10..2022-10) it beat naive in 9 of 11 categories — 7.51% vs 9.30% mean-category WAPE. On the report window (2021-01..2022-10) it won 10 of 11 — 7.02% vs 10.51%. The mechanism is legible: theta tracks trend at short horizons, and at h=1 through the 2021–22 inflationary rebound it scored 3.1% where naive carried 11.5% — naive was still anchoring on year-ago crash values. For the first time since the gate, a challenger had a case.',
    ],
    evidence: {
      label: 'window summary (frozen, 9 rows)',
      href: platformDoc('data/v2/evidence/mrts_research_windows/mrts_window_summary.json'),
    },
  },
  {
    id: 'the-holdout',
    chapter: 5,
    kicker: 'chapter 05 · one run',
    title: 'the holdout, run once',
    lede:
      'A holdout gets spent the moment you look at it. This one was spent exactly once, and it said no.',
    body: [
      'The 2023–2025 origins had been reserved the whole investigation. One run, predeclared: theta needs ≥10 of 11 categories to beat naive for the case to continue. The result — 4.98% vs 5.15% mean-category WAPE, +0.17pp, with only 7 of 11 categories improved. Criterion 2 failed. Theta’s dominance was regime-specific: real on volatile targets, gone on the settled ones.',
      'I find this the most satisfying negative result in the project. The holdout was computed once, written down, and never re-read — no second look, no criterion adjustment. For the record: theta’s 4.98% was still better than the frozen gate’s LightGBM slice (5.64%) on the same origins, and the serving decision stayed fail-closed. Naive stands.',
    ],
    evidence: {
      label: 'holdout record (satellite FINDINGS.md §3)',
      href: satelliteDoc('FINDINGS.md'),
    },
  },
  {
    id: 'two-questions-about-time',
    chapter: 6,
    kicker: 'chapter 06 · satellite, FINDINGS §8–§9',
    title: 'two questions about time',
    lede:
      'How much history does a forecaster need? Fewer questions have cleaner negative answers.',
    body: [
      'Question one: can the past N years predict 2020? Measured on a designated pre-shock/in-shock split. From 2019 origins: no — nothing beat naive on mean-category WAPE (all methods 18.4–19.2% vs naive 18.5%); the collapse was not in any window of pre-2020 history. From 2020 origins, after the shock started: yes — LightGBM beat naive in 8 of 11 categories. Window length itself was second-order: the spread across 5/10/20/full-year training panels was ~1–2pp with inconsistent direction. The five-year protocol floor stays defensible.',
      'Question two: can a decade or two of history predict the 2020s? Theta scored 8.09/8.02/8.03% mean-category WAPE with 10y/20y/full windows — ten years is enough, two decades add nothing. What moved predictability was phase, not window: theta halved naive’s error on 2021 targets (7.9% vs 16.8%), and naive re-took the lead on 2023-target folds (6.86% vs ~7.3%). Adaptive methods own the volatile phase; naive owns the settled tail. The 2020s are also a noisier data regime independent of volatility — preliminary→final revision magnitudes remain ~60% larger than the 2010s (median 1.96% → 3.17%).',
    ],
    evidence: {
      label: 'window studies (satellite FINDINGS.md §8–§9)',
      href: satelliteDoc('FINDINGS.md'),
    },
  },
  {
    id: 'the-switch',
    chapter: 7,
    kicker: 'chapter 07 · satellite, policy layer',
    title: 'the switch',
    lede:
      'If different regimes favor different models, train something to tell them apart. I predeclared what a win would mean — and it wasn’t this.',
    body: [
      'The policy: a capped LightGBM classifier on causal, point-in-time features (trend strength, volatility, revision churn, lagged performance), isotonic-calibrated on leave-one-origin-out CHOOSE scores, calling naive or theta per category and origin when P(theta wins) clears a predeclared τ = 0.50. Trained on 77 CHOOSE pairs, then evaluated once on the report window — n = 99 pairs (9 origins × 11 categories), every uncertainty an origin-block bootstrap CI over 2,000 resamples.',
      'It beat naive in 10 of 11 categories (7.43% vs 10.51%, +3.08pp, CI +0.70..+5.13) and beat the fixed 0.5/0.5 blend (7.77%). It did not beat pure theta (7.02%) — the entire delta CI sits below zero (−0.59..−0.21), so guardrail 2 failed and the predeclared rule earned a findings document, not a promotion. The diagnosis is structural: the report window is single-regime (theta wins 72.7% of pairs there), and a switch has no surface to pay off when one model is already near-ceiling. The measured defect is calibration — Brier 0.155 on CHOOSE OOF degraded to 0.276 on REPORT, worse than always calling theta at the 72.7% base rate. Theta-call quality on REPORT: precision 0.716, recall 0.806.',
    ],
    evidence: {
      label: 'policy findings (satellite POLICY_FINDINGS.md, policy_report.json)',
      href: satelliteDoc('POLICY_FINDINGS.md'),
    },
  },
  {
    id: 'what-remains-open',
    chapter: 8,
    kicker: 'chapter 08 · open questions',
    title: 'what remains open',
    lede:
      'The gate is still closed, on purpose. Here is exactly what would reopen it.',
    body: [
      'The serving decision stays fail-closed until a challenger passes a predeclared gate — that has not changed since the first refusal, and nothing in the satellite challenged it. The documented re-open path: accumulate roughly 24 more calm forecast origins (≈ end of 2027), then re-gate with theta as a third benchmark beside naive and LightGBM.',
      'The research agenda the policy layer left behind is specific: a switch only pays on regime-mixed windows, so any future attempt needs evaluation windows that actually mix regimes — and a calibrator that survives base-rate movement between windows. Isotonic on 77 out-of-fold rows collapsed to a single step and transferred badly across the regime boundary; base-rate-robust calibration is the spec for whoever tries next. Everything above is recorded, versioned, and linked — including the parts that failed. Especially those.',
    ],
    evidence: {
      label: 're-open conditions (satellite FINDINGS.md §5)',
      href: satelliteDoc('FINDINGS.md'),
    },
  },
];
