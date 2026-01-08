/**
 * Environment Configuration
 *
 * Centralized configuration for demo mode and API settings
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
