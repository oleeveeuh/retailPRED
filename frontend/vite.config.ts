import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vite.dev/config/
export default defineConfig(() => ({
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
  // GitHub Pages serves the showcase from the repository subpath; the
  // Pages workflow sets VITE_PAGES_BASE=/retailPRED/. Local development
  // and other root deployments keep '/'.
  base: process.env.VITE_PAGES_BASE || '/',
  // Additional static file serving
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          react: ['react', 'react-dom', 'react-router-dom', '@tanstack/react-query'],
          charts: ['recharts'],
          motion: ['framer-motion'],
        },
      },
    },
  },
}))
