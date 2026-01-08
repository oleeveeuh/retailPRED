import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vite.dev/config/
export default defineConfig(({ mode }) => ({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  publicDir: 'public',
  // Additional static file serving
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  // Force demo mode in production builds
  define: {
    __DEMO_MODE__: mode === 'production' ? 'true' : 'false',
  },
}))
