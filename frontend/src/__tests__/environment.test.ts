import { describe, expect, it } from 'vitest';

/**
 * The snapshot URL resolution rules: an explicit override always wins; a
 * production build defaults to the same-origin relative Pages path; local
 * development defaults to empty so the panel stays hidden.
 */
function resolveSnapshotUrl(isDev: boolean, override: string | undefined): string {
  return override ?? (isDev ? '' : 'live/v2-latest.json');
}

describe('resolveSnapshotUrl', () => {
  it('defaults a production build to the Pages-relative snapshot path', () => {
    expect(resolveSnapshotUrl(false, undefined)).toBe('live/v2-latest.json');
  });

  it('defaults local development to empty so the panel stays hidden', () => {
    expect(resolveSnapshotUrl(true, undefined)).toBe('');
  });

  it('lets an explicit override win in any mode', () => {
    expect(resolveSnapshotUrl(false, 'https://example.com/snap.json')).toBe(
      'https://example.com/snap.json',
    );
    expect(resolveSnapshotUrl(true, 'http://localhost:8000/live/v2-latest.json')).toBe(
      'http://localhost:8000/live/v2-latest.json',
    );
  });
});
