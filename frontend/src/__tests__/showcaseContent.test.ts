import { describe, expect, it } from 'vitest';

import {
  gateStory,
  hero,
  modelCards,
  pipelineTasks,
  rejections,
  tableauViews,
} from '../content/showcase';

const BANNED_CLAIMS = ['90.2', '0.26%', 'production database', 'currently deployed'];

const serialized = JSON.stringify({ hero, gateStory, modelCards, pipelineTasks, rejections, tableauViews });

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

  it('describes all five governed tableau views', () => {
    expect(tableauViews).toHaveLength(5);
    expect(new Set(tableauViews.map((view) => view.name)).size).toBe(5);
  });

  it('keeps the gate story honest about the failed verdict', () => {
    expect(gateStory.title).toMatch(/refused/i);
    expect(gateStory.metrics.some((metric) => metric.value === '8.49%')).toBe(true);
    expect(gateStory.metrics.some((metric) => metric.value === '8.16%')).toBe(true);
  });
});
