/**
 * The toolkit: every entry is a technology, a hand-rolled inline SVG mark
 * in the field-notes palette (no icon-font dependency), and a proof link
 * into this repository — the artifact that demonstrates the skill.
 *
 * Icons are simple original marks (stroke-based, 24×24, currentColor);
 * they are first-party static strings rendered by SkillsProofGrid.
 */

const REPO_URL = 'https://github.com/oleeveeuh/retailPRED';

const repoBlob = (docPath: string) => `${REPO_URL}/blob/main/${docPath}`;
const repoTree = (dirPath: string) => `${REPO_URL}/tree/main/${dirPath}`;

export interface Skill {
  name: string;
  /** Inline SVG mark (full <svg> element, stroke-based, 24×24 viewBox). */
  icon: string;
  proofLabel: string;
  proofHref: string;
}

export const skills: Skill[] = [
  {
    name: 'Python',
    icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M11.5 3h4A2.5 2.5 0 0 1 18 5.5v4a2.5 2.5 0 0 1-2.5 2.5h-7A3.5 3.5 0 0 0 5 15.5v3A2.5 2.5 0 0 0 7.5 21h4" fill="none"/><path d="M12.5 21h-4A2.5 2.5 0 0 1 6 18.5v-4A2.5 2.5 0 0 1 8.5 12h7a3.5 3.5 0 0 0 3.5-3.5v-3A2.5 2.5 0 0 0 16.5 3h-4" fill="none"/><circle cx="8.8" cy="6.6" r="0.9" fill="currentColor" stroke="none"/><circle cx="15.2" cy="17.4" r="0.9" fill="currentColor" stroke="none"/></svg>`,
    proofLabel: 'forecasting package',
    proofHref: repoTree('forecasting'),
  },
  {
    name: 'LightGBM',
    icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 4c-9.5 0-14.5 5-15.5 15.5C15 18.5 20 13.5 20 4Z"/><path d="M4.5 19.5C7.5 14 13 8.5 20 4"/></svg>`,
    proofLabel: 'global model source',
    proofHref: repoBlob('forecasting/models/global_lightgbm.py'),
  },
  {
    name: 'Databricks',
    icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m12 3 7 3.5L12 10 5 6.5Z"/><path d="m5 10.5 7 3.5 7-3.5"/><path d="m5 14.5 7 3.5 7-3.5"/><path d="m5 18.5 7 3.5 7-3.5"/></svg>`,
    proofLabel: 'governed pipeline',
    proofHref: repoTree('databricks'),
  },
  {
    name: 'Delta Lake',
    icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3.5 21 19H3Z"/><path d="M8 14.5h8"/><path d="M9.5 17h5"/></svg>`,
    proofLabel: 'spark/delta contract',
    proofHref: repoBlob('retailpred/platform/spark_delta.py'),
  },
  {
    name: 'dbt',
    icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 19 12 4l7 15"/><path d="M8.5 13h7"/><path d="M12 19v2"/></svg>`,
    proofLabel: 'warehouse project',
    proofHref: repoTree('warehouse'),
  },
  {
    name: 'FastAPI',
    icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M13.5 3 5.5 13H11l-1.5 8 8-10H12Z"/></svg>`,
    proofLabel: 'serving api',
    proofHref: repoBlob('backend/api/v2_forecasts.py'),
  },
  {
    name: 'Tableau',
    icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3v5m0 8v5m9-9h-5M8 12H3"/><rect x="10" y="10" width="4" height="4"/><path d="M18.5 6.5v3m0 5v3m0-8h.01M5.5 8.5v3m0 5v3m0-8h.01" opacity="0.55"/></svg>`,
    proofLabel: 'governed bi views',
    proofHref: repoTree('bi'),
  },
  {
    name: 'pandas',
    icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 20h16"/><path d="M7 20v-8m5 8V6m5 14v-11"/><path d="M7 9V7" opacity="0.55"/><path d="M17 6V4" opacity="0.55"/></svg>`,
    proofLabel: 'vintage harness',
    proofHref: repoBlob('forecasting/data/monthly_dataset.py'),
  },
  {
    name: 'GitHub Actions',
    icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 12a8 8 0 1 1-8-8"/><circle cx="20" cy="12" r="2.4"/><path d="m9.5 8.5 5 3.5-5 3.5Z"/></svg>`,
    proofLabel: 'ci/cd workflow',
    proofHref: repoBlob('.github/workflows/ci-cd.yml'),
  },
  {
    name: 'pytest',
    icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M19.5 13.5A7.5 7.5 0 1 1 17 6"/><path d="m9 12 2.4 2.4L16 9.5"/><path d="M17 6V3.5M17 6h-2.5" transform="translate(0.5 0.5)"/></svg>`,
    proofLabel: 'test suite',
    proofHref: repoTree('tests'),
  },
];
