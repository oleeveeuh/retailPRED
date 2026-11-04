/**
 * Environment Configuration
 *
 * Centralized configuration for demo mode and API settings
 * Updated: 2026-01-08
 */

export const config = {
  /**
   * Demo mode: when true, loads data from static JSON files
   * when false, makes API calls to backend
   */
  isDemoMode: import.meta.env.VITE_DEMO_MODE === 'true',

  /**
   * Backend API URL (only used when isDemoMode is false)
   * Note: Empty string is valid (no backend), only fallback to localhost if undefined
   */
  apiUrl: import.meta.env.VITE_API_URL !== undefined ? import.meta.env.VITE_API_URL : 'http://localhost:8000',

  /**
   * Public URL of the v2 latest-forecast snapshot (scheduled batch output).
   * GitHub Pages deployments default to the same-origin relative path the
   * Pages workflow places the snapshot at; local development defaults to
   * empty (the snapshot panel stays hidden). Override with
   * VITE_LATEST_SNAPSHOT_URL for other hosts.
   */
  latestSnapshotUrl:
    import.meta.env.VITE_LATEST_SNAPSHOT_URL ?? (import.meta.env.DEV ? '' : 'live/v2-latest.json'),

  /**
   * Enable debug logging
   */
  isDebug: import.meta.env.VITE_DEBUG === 'true',
};

// Log configuration in development
if (config.isDebug || import.meta.env.DEV) {
  console.log('🔧 Environment Configuration:', {
    isDemoMode: config.isDemoMode,
    apiUrl: config.apiUrl,
    mode: import.meta.env.MODE,
  });
}
