import { existsSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

import {
  gateStory,
  hero,
  modelCards,
  pipelineTasks,
  rejections,
  tableauViews,
} from '../content/showcase';
import {
  SATELLITE_DOC_LABELS,
  SATELLITE_REPO_URL,
  REPO_URL,
  effectiveLinks,
  timeline,
} from '../content/timeline';
import { skills } from '../content/skills';

const BANNED_CLAIMS = ['90.2', '0.26%', 'production database', 'currently deployed'];

const serialized = JSON.stringify({ hero, gateStory, modelCards, pipelineTasks, rejections, tableauViews, timeline, skills });

/** Map a GitHub blob/tree URL to its path on this checkout, or null if foreign. */
const repoPathForHref = (href: string): string | null => {
  const match = href.match(
    /^https:\/\/github\.com\/oleeveeuh\/retailPRED\/(blob|tree)\/main\/(.+)$/,
  );
  if (!match) return null;
  return path.resolve(
    path.dirname(fileURLToPath(import.meta.url)),
    '../../..',
    decodeURIComponent(match[2]),
  );
};

const expectHrefResolves = (href: string) => {
  expect(href, 'every public link must point into this repository').toMatch(
    /^https:\/\/github\.com\/oleeveeuh\/retailPRED\/(blob|tree)\/main\//,
  );
  const localPath = repoPathForHref(href);
  expect(localPath, `link target must exist in the checkout: ${href}`).not.toBeNull();
  expect(existsSync(localPath as string), `missing on disk: ${href}`).toBe(true);
  if (href.includes('/tree/')) {
    expect(statSync(localPath as string).isDirectory(), `tree link must be a directory: ${href}`).toBe(true);
  } else {
    expect(statSync(localPath as string).isFile(), `blob link must be a file: ${href}`).toBe(true);
  }
};

describe('showcase content', () => {
  it('carries no legacy fabricated claims', () => {
    for (const claim of BANNED_CLAIMS) {
      expect(serialized.toLowerCase()).not.toContain(claim.toLowerCase());
    }
  });

  it('keeps the four-task pipeline with one executed and one gated task', () => {
    expect(pipelineTasks).toHaveLength(4);
    expect(pipelineTasks.some((task) => task.status === 'executed')).toBe(true);
    expect(pipelineTasks.some((task) => task.status === 'refused-by-gate')).toBe(true);
    for (const task of pipelineTasks) {
      expect(task.headline.length).toBeGreaterThan(0);
      expect(task.evidence.length).toBeGreaterThan(0);
    }
  });

  it('presents one promoted and one gate-failed model', () => {
    expect(modelCards).toHaveLength(2);
    const decisions = modelCards.map((card) => card.decision).sort();
    expect(decisions).toEqual(['challenger-gate-failed', 'promoted-demo']);
    for (const card of modelCards) {
      expect(card.metricValue).not.toBe('');
      expect(card.baselineValue).not.toBe('');
      expect(card.evidencePath).toContain('github.com/oleeveeuh/retailPRED');
    }
  });

  it('documents every rejected challenger with an outcome', () => {
    expect(rejections.length).toBeGreaterThanOrEqual(4);
    for (const rejection of rejections) {
      expect(rejection.outcome.toLowerCase()).toContain('reject');
    }
  });

  it('describes all nine governed tableau views (the local workbook)', () => {
    expect(tableauViews).toHaveLength(9);
    expect(new Set(tableauViews.map((view) => view.name)).size).toBe(9);
    // The three added MRTS diagnostics are present with their frozen mark counts.
    const names = tableauViews.map((view) => view.name);
    expect(names).toContain('MRTS feature ladder');
    expect(names).toContain('MRTS horizon accuracy');
    expect(names).toContain('MRTS candidate comparisons');
    const research = tableauViews.find((view) => view.name === 'MRTS research windows')!;
    expect(research.oneLine).toMatch(/Chronos/);
    expect(research.oneLine).toMatch(/NHITS/);
    expect(research.oneLine).toMatch(/post-hoc/);
  });

  it('keeps the gate story honest about the failed verdict', () => {
    expect(gateStory.title).toMatch(/refused/i);
    expect(gateStory.metrics.some((metric) => metric.value === '8.49%')).toBe(true);
    expect(gateStory.metrics.some((metric) => metric.value === '8.16%')).toBe(true);
  });
});

describe('execution timeline content', () => {
  it('tells the eleven-phase story in order with unique ids', () => {
    expect(timeline).toHaveLength(11);
    expect(new Set(timeline.map((entry) => entry.id)).size).toBe(11);
    for (const entry of timeline) {
      expect(entry.phase).toMatch(/^phase /);
      expect(entry.phase.length).toBeGreaterThan(0);
      expect(entry.headline.length).toBeGreaterThan(0);
      expect(entry.detail.length).toBeGreaterThan(0);
    }
    expect(timeline[0].id).toBe('v1-gcp-airflow');
    expect(timeline[10].id).toBe('next-regime-classifier');
  });

  it('keeps the honest measured numbers at each turn', () => {
    const detail = (id: string) => {
      const entry = timeline.find((candidate) => candidate.id === id);
      expect(entry, `missing timeline entry ${id}`).toBeTruthy();
      return entry!.detail;
    };
    expect(detail('v1-gcp-airflow')).toContain('9.76%');
    expect(detail('v1-gcp-airflow')).toContain('14.11%');
    expect(detail('v2-rebuild')).toContain('6.35%');
    expect(detail('revised-history')).toContain('4.49%');
    expect(detail('vintage-gate')).toContain('8.49%');
    expect(detail('vintage-gate')).toContain('8.16%');
    expect(detail('satellite-phase-0')).toContain('0.0');
    expect(detail('satellite-phase-1')).toContain('−0.755');
    expect(detail('h5-theta')).toContain('10/11');
    expect(detail('h5-theta')).toContain('+3.48pp');
    expect(detail('holdout')).toContain('+0.17pp');
    expect(detail('holdout')).toContain('7/11');
    expect(detail('holdout')).toContain('Naive stands');
    expect(detail('challenge-audits')).toContain('0/11');
    expect(detail('challenge-audits')).toContain('11/11');
    // The selector is a measured negative, not "the fix" — no unmeasured
    // downside-bound claims survive in the timeline.
    expect(detail('next-regime-classifier')).toContain('not justified');
    expect(detail('next-regime-classifier')).not.toContain('worst case ties naive');
  });

  it('links public phases to evidence that exists in the checkout', () => {
    const publicIds = ['v1-gcp-airflow', 'v2-rebuild', 'revised-history', 'vintage-gate'];
    for (const id of publicIds) {
      const entry = timeline.find((candidate) => candidate.id === id)!;
      expect(entry.links.length, `phase ${id} must carry evidence links`).toBeGreaterThan(0);
      for (const link of entry.links) {
        expectHrefResolves(link.href);
      }
    }
  });

  it('links satellite phases to the satellite repository once it is public', () => {
    const satelliteEntries = timeline.filter((entry) => entry.satelliteDoc);
    expect(satelliteEntries).toHaveLength(7);
    if (SATELLITE_REPO_URL === null) {
      for (const entry of satelliteEntries) {
        expect(entry.links, `phase ${entry.id} must not link a private repo`).toHaveLength(0);
      }
      return;
    }
    expect(SATELLITE_REPO_URL).toMatch(
      /^https:\/\/github\.com\/oleeveeuh\/retailPRED-satellite$/,
    );
    for (const entry of satelliteEntries) {
      expect(entry.satelliteDoc).toMatch(/\.md$/);
      expect(SATELLITE_DOC_LABELS[entry.satelliteDoc!], `doc ${entry.satelliteDoc} needs a label`).toBeTruthy();
      const satelliteLink = effectiveLinks(entry).find((link) =>
        link.href.startsWith(SATELLITE_REPO_URL!),
      );
      expect(satelliteLink, `phase ${entry.id} must carry its satellite link`).toBeTruthy();
      expect(satelliteLink!.href).toBe(`${SATELLITE_REPO_URL}/blob/main/${entry.satelliteDoc}`);
    }
    expect(REPO_URL).toBe('https://github.com/oleeveeuh/retailPRED');
  });
});

describe('skills proof grid content', () => {
  it('covers every required technology with a proof link', () => {
    const required = [
      'Python',
      'LightGBM',
      'Databricks',
      'Delta Lake',
      'dbt',
      'FastAPI',
      'Tableau',
      'pandas',
      'GitHub Actions',
      'pytest',
    ];
    const names = skills.map((skill) => skill.name);
    expect(new Set(names).size).toBe(skills.length);
    for (const name of required) {
      expect(names, `missing skill: ${name}`).toContain(name);
    }
  });

  it('gives every skill a hand-rolled icon and an on-disk proof', () => {
    for (const skill of skills) {
      expect(skill.icon.trim().startsWith('<svg')).toBe(true);
      expect(skill.icon.trim().endsWith('</svg>')).toBe(true);
      expect(skill.icon).toContain('viewBox="0 0 24 24"');
      // No icon-font or remote dependency: the mark must be self-contained.
      expect(skill.icon).not.toMatch(/https?:|<script|onerror|onload|javascript:|url\(/i);
      expect(skill.proofLabel.length).toBeGreaterThan(0);
      expectHrefResolves(skill.proofHref);
    }
  });
});
