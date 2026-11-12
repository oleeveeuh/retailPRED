import { existsSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

import { dataContext, SATELLITE_URL, storyChapters, v1Chapters } from '../content/story';

const BANNED_CLAIMS = ['90.2', '0.26%', 'production database', 'currently deployed'];

const serialized = JSON.stringify({ storyChapters, v1Chapters, dataContext });

const repoRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  '../../..',
);

/** Map a platform GitHub blob URL to its path on this checkout, or null if foreign. */
const platformPathForHref = (href: string): string | null => {
  const match = href.match(
    /^https:\/\/github\.com\/oleeveeuh\/retailPRED\/blob\/main\/(.+)$/,
  );
  if (!match) return null;
  return path.resolve(repoRoot, decodeURIComponent(match[1]));
};

const expectPlatformHrefResolves = (href: string) => {
  const localPath = platformPathForHref(href);
  expect(localPath, `expected a platform blob URL: ${href}`).not.toBeNull();
  expect(
    existsSync(localPath as string),
    `missing on disk: ${href}`,
  ).toBe(true);
  expect(statSync(localPath as string).isFile(), `not a file: ${href}`).toBe(true);
};

const expectSatelliteHref = (href: string) => {
  expect(
    href.startsWith(`${SATELLITE_URL}/blob/main/`),
    `expected a satellite blob URL: ${href}`,
  ).toBe(true);
};

const chapter = (id: string) => {
  const found = storyChapters.find((candidate) => candidate.id === id);
  expect(found, `missing story chapter ${id}`).toBeTruthy();
  return found!;
};

describe('story content', () => {
  it('tells exactly eleven chapters with unique ordered ids', () => {
    expect(storyChapters).toHaveLength(11);
    expect(new Set(storyChapters.map((chapter) => chapter.id)).size).toBe(11);
    storyChapters.forEach((chapter, index) => {
      expect(chapter.chapter).toBe(index + 1);
      expect(chapter.kicker.length).toBeGreaterThan(0);
      expect(chapter.title.length).toBeGreaterThan(0);
      expect(chapter.lede.length).toBeGreaterThan(0);
      const paragraphs = chapter.blocks.filter((block) => block.kind === 'paragraph');
      expect(paragraphs.length).toBeGreaterThanOrEqual(2);
      for (const block of chapter.blocks) {
        if (block.kind === 'paragraph') {
          expect(block.text.length).toBeGreaterThan(0);
        }
      }
    });
    expect(storyChapters[0].id).toBe('the-gate-that-said-no');
    expect(storyChapters[7].id).toBe('the-wider-search');
    expect(storyChapters[8].id).toBe('feature-engineering-ceiling');
    expect(storyChapters[9].id).toBe('what-remains-open');
    expect(storyChapters[10].id).toBe('the-live-loop');
  });

  it('carries no legacy fabricated claims', () => {
    for (const claim of BANNED_CLAIMS) {
      expect(serialized.toLowerCase()).not.toContain(claim.toLowerCase());
    }
  });

  it('links every chapter to evidence that resolves', () => {
    for (const chapter of storyChapters) {
      expect(chapter.evidence.label.length).toBeGreaterThan(0);
      if (chapter.evidence.href.startsWith('https://github.com/oleeveeuh/retailPRED/')) {
        expectPlatformHrefResolves(chapter.evidence.href);
      } else {
        expectSatelliteHref(chapter.evidence.href);
      }
    }
  });

  it('freezes the honest measured numbers in the prose', () => {
    const text = (id: string) => {
      const found = chapter(id);
      return [
        found.lede,
        ...found.blocks.map((block) => (block.kind === 'paragraph' ? block.text : '')),
      ].join(' ');
    };
    // Full-grid gate (docs/v2/M3_VINTAGE_BACKTEST.md) — real-time measurement.
    expect(text('the-gate-that-said-no')).toContain('8.49%');
    expect(text('the-gate-that-said-no')).toContain('8.16%');
    expect(text('the-gate-that-said-no')).toContain('4 of 11');
    // Window research (mrts_window_summary.json).
    expect(text('where-the-failure-lives')).toContain('−0.755');
    expect(text('theta')).toContain('7.51%');
    expect(text('theta')).toContain('7.02%');
    expect(text('the-holdout')).toContain('+0.17pp');
    expect(text('the-holdout')).toContain('7 of 11');
    // Holdout scorecard (reporting-only audit; satellite results/holdout_scorecard.md).
    expect(text('the-holdout')).toContain('−0.17..+0.54');
    expect(text('the-holdout')).toContain('−1.24..−0.36');
    expect(text('the-holdout')).toContain('+1.19pp');
    expect(text('the-holdout')).toContain('computed once and frozen');
    // Policy (satellite results/policy_report.json) with its sample-size constraint.
    expect(text('the-switch')).toContain('n = 99');
    expect(text('the-switch')).toContain('77 pairs');
    expect(text('the-switch')).toContain('7.43%');
    expect(text('the-switch')).toContain('+0.70..+5.13');
    expect(text('the-switch')).toContain('7.02%');
    expect(text('the-switch')).toContain('0.155');
    expect(text('the-switch')).toContain('0.276');
    // Family benchmarks (satellite CHRONOS_MRTS.md / NHITS_MRTS.md; holdout post-hoc).
    expect(text('the-wider-search')).toContain('7.75%');
    expect(text('the-wider-search')).toContain('8.91%');
    expect(text('the-wider-search')).toContain('5.00%');
    expect(text('the-wider-search')).toContain('6.90%');
    expect(text('the-wider-search')).toMatch(/post-hoc/);
    expect(text('the-wider-search')).toMatch(/not conclusions/);
    // Feature ladder (satellite H7_FEATURE_LADDER.md / results/h7_scorecard.json).
    expect(text('feature-engineering-ceiling')).toContain('+0.16pp');
    expect(text('feature-engineering-ceiling')).toContain('13 of 15');
    expect(text('feature-engineering-ceiling')).toContain('−1.50pp');
    expect(text('feature-engineering-ceiling')).toContain('11.47%');
    expect(text('feature-engineering-ceiling')).toContain('2.23pp');
    // The verdict stays fail-closed.
    expect(text('what-remains-open')).toMatch(/fail-closed/);
    expect(text('what-remains-open')).toMatch(/closed for this design/);
    expect(text('the-holdout')).toMatch(/Naive stands/);
    // The H9 feature revisit (satellite FINDINGS §13; results/h9_scorecard.md,
    // 2026-11-11): the v1 families rebuilt vintage-safe — the §5 open route is
    // measured and closed, and the second challenger is pre-qualified.
    expect(text('what-remains-open')).toContain('56,364');
    expect(text('what-remains-open')).toContain('+0.21pp');
    expect(text('what-remains-open')).toContain('8.85%');
    expect(text('what-remains-open')).toContain('−1.34pp');
    expect(text('what-remains-open')).toContain('−2.39..−0.50');
    expect(text('what-remains-open')).toMatch(/second pre-qualified challenger/);
    expect(text('what-remains-open')).toMatch(/June 2027/);
    // The live loop (DATABRICKS_RUN.md run log; the verified hosted-run receipt).
    expect(text('the-live-loop')).toContain('452582046469351');
    expect(text('the-live-loop')).toContain('2,200');
    expect(text('the-live-loop')).toContain('2026-08-01');
    expect(text('the-live-loop')).toContain('14:30');
    expect(text('the-live-loop')).toMatch(/fail-closed/);
    expect(text('the-live-loop')).toMatch(/2\.2-status/);
  });

  it('resolves any declared chapter figure inside frontend/public', () => {
    const base = (import.meta.env.BASE_URL || '/').replace(/\/$/, '');
    for (const chapter of storyChapters) {
      for (const block of chapter.blocks) {
        if (block.kind !== 'figure') continue;
        const figure = block.figure;
        expect(figure.caption.length).toBeGreaterThan(0);
        expect(figure.alt.length).toBeGreaterThan(0);
        expect(figure.src.startsWith(base + '/images/story/')).toBe(true);
        const localPath = path.resolve(
          repoRoot,
          'frontend/public',
          figure.src.slice(base.length + 1),
        );
        expect(existsSync(localPath), `missing figure: ${figure.src}`).toBe(true);
      }
    }
  });

  it('grounds the data context in the platform contract', () => {
    expect(dataContext.paragraphs).toHaveLength(2);
    const text = dataContext.paragraphs.join(' ');
    // docs/architecture/data_platform.md + DATABRICKS_RUN.md run log.
    expect(text).toContain('110,921');
    expect(text).toContain('(series, observation month, vintage)');
    expect(text).toContain('1.96%');
    expect(text).toContain('3.17%');
    expect(text).toContain('45-day publication allowance');
    expectPlatformHrefResolves(dataContext.evidence.href);
  });

  it('keeps the satellite constant pointed at the public repository', () => {
    expect(SATELLITE_URL).toBe('https://github.com/oleeveeuh/retailPRED-satellite');
  });

  it('tells exactly three v1 chapters with unique ids', () => {
    expect(v1Chapters).toHaveLength(3);
    expect(new Set(v1Chapters.map((chapter) => chapter.id)).size).toBe(3);
    for (const chapter of v1Chapters) {
      expect(chapter.kicker.length).toBeGreaterThan(0);
      expect(chapter.title.length).toBeGreaterThan(0);
      expect(chapter.lede.length).toBeGreaterThan(0);
      const paragraphs = chapter.blocks.filter((block) => block.kind === 'paragraph');
      expect(paragraphs.length).toBeGreaterThanOrEqual(2);
    }
  });

  it('links every v1 chapter to platform evidence that resolves', () => {
    for (const chapter of v1Chapters) {
      expectPlatformHrefResolves(chapter.evidence.href);
    }
  });

  it('freezes the audited v1 numbers without the fabricated headline', () => {
    const text = (id: string) => {
      const found = v1Chapters.find((entry) => entry.id === id)!;
      return [
        found.lede,
        ...found.blocks.map((block) => (block.kind === 'paragraph' ? block.text : '')),
      ].join(' ');
    };
    // README §"The v1 → v2 story" / docs/v2/AUDIT.md.
    expect(text('v1-numbers-without-meaning')).toContain('9.76%');
    expect(text('v1-numbers-without-meaning')).toContain('14.11%');
    expect(text('v1-numbers-without-meaning')).toContain('429');
    expect(text('v1-numbers-without-meaning')).toMatch(/pseudo-replication/i);
    // The fabricated accuracy figure is never restated — the evidence link
    // carries it (BANNED_CLAIMS already enforces the '90.2' literal above).
    expect(text('v1-epistemic-not-operational')).toContain('29%');
    expect(text('v1-epistemic-not-operational')).toMatch(/fail-closed/);
  });
});
