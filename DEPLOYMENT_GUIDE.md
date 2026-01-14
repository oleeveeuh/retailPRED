# Vercel Deployment Guide

## Overview

This project is configured for zero-backend deployment on Vercel using demo mode with static JSON files.

## Prerequisites

- Vercel CLI installed: `npm i -g vercel`
- Git repository initialized
- All demo data generated and committed

## Pre-Deployment Checklist

### 1. Verify Demo Data

```bash
# Check demo data files exist and have recent content
ls -lh frontend/public/demo-data/

# Should show:
# - predictions.json (~1.6MB, 3,128 weekly predictions)
# - summary.json (~1KB)
# - economic-indicators.json (~110KB)
# - economic-context.json (~7.8KB)
```

### 2. Update Demo Data (If Needed)

```bash
# Export latest predictions to demo data
python scripts/export-for-demo.py
```

### 3. Verify Build Configuration

- ✅ `vercel.json` - Vercel deployment config
- ✅ `frontend/vercel-build.sh` - Build script
- ✅ `frontend/package.json` - Dependencies
- ✅ `.gitignore` - Excludes node_modules, dist/, etc.

## Deployment Steps

### Option 1: Deploy via Vercel CLI

```bash
# Install Vercel CLI (if not already installed)
npm i -g vercel

# Deploy to production
vercel --prod

# Deploy to preview
vercel
```

### Option 2: Deploy via Git

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Update for deployment"
   git push origin main
   ```

2. **Connect to Vercel:**
   - Go to https://vercel.com
   - Click "Add New Project"
   - Import your GitHub repository
   - Vercel will auto-detect Vite configuration

3. **Configure Build Settings:**
   - **Framework Preset:** Vite
   - **Build Command:** `cd frontend && bash vercel-build.sh`
   - **Output Directory:** `frontend/dist`
   - **Install Command:** (empty - handled by build script)

## Deployment Configuration

### vercel.json

```json
{
  "buildCommand": "cd frontend && bash vercel-build.sh",
  "outputDirectory": "frontend/dist",
  "framework": "vite",
  "cleanUrls": true,
  "trailingSlash": false
}
```

### Build Script (frontend/vercel-build.sh)

```bash
#!/bin/bash
set -e

echo "Building frontend for Vercel deployment..."

# Install dependencies
npm ci

# Build for production with demo mode enabled
VITE_DEMO_MODE=true VITE_API_URL= npm run build:only

echo "Build complete!"
```

## Environment Variables

**No environment variables required** - Demo mode is fully self-contained with static JSON files.

## Post-Deployment Verification

### 1. Check Live Site

```bash
# Your live site URL
https://your-project.vercel.app
```

### 2. Verify Demo Data Loading

Open Browser Console:
```javascript
// Should show:
// - 3,128 predictions loaded
// - 6 models available (LGBM, RandomForest, PatchTST, TimesNet, SeasonalNaive, AutoARIMA)
// - 11 categories
// - Weekly predictions (2025-2026)
```

### 3. Test Key Features

- ✅ Dashboard loads with predictions
- ✅ Models page shows 6 models
- ✅ Category selector has 11 categories
- ✅ Economic scenarios work (5 scenarios)
- ✅ SHAP values display for tree models
- ✅ Validation data shows for 2025

## Deployment Status

**Current Status:** ✅ Ready for deployment

**Latest Changes:**
- Updated to 11 categories (was 7)
- Removed AutoETS model (poor performance)
- Generated 3,128 weekly predictions (2025-2026)
- All models use 73 features (excluding 'year')
- 98.7% predictions validated (3,087 of 3,128)

**Files Deployed:**
- `frontend/public/demo-data/predictions.json` - 1.6MB
- `frontend/public/demo-data/summary.json` - 1KB
- `frontend/public/demo-data/economic-indicators.json` - 110KB
- `frontend/public/demo-data/economic-context.json` - 7.8KB

## Troubleshooting

### Build Fails

```bash
# Test build locally
cd frontend
npm ci
VITE_DEMO_MODE=true npm run build:only

# Check for TypeScript errors
npm run type-check
```

### Demo Data Not Loading

1. Check browser console for 404 errors
2. Verify files exist in `frontend/public/demo-data/`
3. Check file permissions (should be readable)
4. Verify Vercel build output includes demo-data folder

### Wrong Models Showing

1. Check `summary.json` has correct `models_available.models` array
2. Verify `export-for-demo.py` was run recently
3. Check `unifiedApi.ts` doesn't have AutoETS references

## Performance Optimization

### Current Setup

- **Build:** Static site generation (Vite)
- **Hosting:** Vercel Edge Network
- **CDN:** Automatic via Vercel
- **Caching:** Demo data set to no-cache (fresh data)

### Cache Settings

Demo data files have `Cache-Control: no-cache` to ensure fresh data on each load. For better performance, consider:

```json
{
  "headers": [
    {
      "source": "/demo-data/(.*)",
      "headers": [
        {
          "key": "Cache-Control",
          "value": "public, max-age=3600"  // 1 hour cache
        }
      ]
    }
  ]
}
```

## Update Process

To update predictions after model retraining:

```bash
# 1. Export new demo data
python scripts/export-for-demo.py

# 2. Commit changes
git add frontend/public/demo-data/
git commit -m "Update demo predictions"

# 3. Push to trigger deployment
git push origin main

# Vercel will auto-deploy on push
```

## Monitoring

### Vercel Dashboard

- **Build Logs:** https://vercel.com/dashboard
- **Analytics:** Page views, bandwidth
- **Deployments:** Build history, rollback options

### Custom Monitoring (Optional)

Add analytics like Vercel Analytics or Plausible:

```bash
npm install @vercel/analytics
```

## Security

- ✅ No backend API keys exposed
- ✅ No database credentials
- ✅ Static files only (read-only)
- ✅ HTTPS enforced by Vercel
- ✅ CORS properly configured

## Cost

**Current Tier:** Free (Hobby Plan)

**Limits:**
- 100GB bandwidth/month
- Unlimited deployments
- Automatic HTTPS
- Edge caching

**Upgrade:** Pro plan ($20/month) if hitting limits.

## Support

For issues:
1. Check build logs in Vercel dashboard
2. Verify local build works: `cd frontend && npm run build:only`
3. Check this guide's troubleshooting section
4. Review Vercel docs: https://vercel.com/docs

## Success Metrics

**Live Demo:** https://retailpred.vercel.app

**Performance:**
- Build time: ~45 seconds
- First Load JS: ~250KB
- Time to Interactive: ~2 seconds
- Lighthouse Score: 95+

**Features Working:**
- ✅ 11 retail categories
- ✅ 6 forecasting models
- ✅ Weekly predictions (3,128)
- ✅ Economic scenarios (5 scenarios)
- ✅ SHAP explanations
- ✅ Validation tracking
