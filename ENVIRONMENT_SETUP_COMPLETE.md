# Environment Configuration Setup - COMPLETE ✅

## Overview

Environment configuration has been successfully set up for both local development and production deployment (Vercel).

## Files Created

### 1. `frontend/.env.development`
**Purpose**: Local development with backend API

```bash
VITE_DEMO_MODE=false
VITE_API_URL=http://localhost:8000
VITE_TABLEAU_EMBED_URL="https://public.tableau.com/views/Book1_17676501972860/Sheet1?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link"
```

**Usage**: Automatically used when running `npm run dev`

**Features**:
- ✅ Live backend API calls
- ✅ Real-time predictions
- ✅ Database queries
- ✅ Full feature access

### 2. `frontend/.env.production`
**Purpose**: Production/Vercel deployment with demo mode

```bash
VITE_DEMO_MODE=true
VITE_API_URL=
VITE_TABLEAU_EMBED_URL="https://public.tableau.com/views/Book1_17676501972860/Sheet1?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link"
```

**Usage**: Automatically used when running `npm run build`

**Features**:
- ✅ Static JSON data
- ✅ No backend required
- ✅ Fast loading
- ✅ Vercel-ready

### 3. `frontend/.env.example`
**Purpose**: Template for custom local overrides

```bash
VITE_DEMO_MODE=false
VITE_API_URL=http://localhost:8000
VITE_TABLEAU_EMBED_URL=""
VITE_DEBUG=false
```

**Usage**: Copy to `.env.local` for custom development configuration

## Environment Variables Explained

### `VITE_DEMO_MODE`
- **Type**: Boolean
- **Values**: `true` or `false`
- **Default**: `false`
- **Purpose**: Controls whether app uses static JSON or backend API

**When `true`**:
- Loads data from `public/demo-data/*.json`
- Shows DemoBanner component
- No API calls made
- Ideal for Vercel deployment

**When `false`**:
- Makes HTTP requests to backend API
- Live data fetching
- Full feature access
- Requires backend server running

### `VITE_API_URL`
- **Type**: URL string
- **Default**: `http://localhost:8000`
- **Purpose**: Backend API endpoint
- **Usage**: Only used when `VITE_DEMO_MODE=false`

### `VITE_TABLEAU_EMBED_URL`
- **Type**: URL string
- **Required**: Yes
- **Purpose**: Tableau Public visualization URL
- **Format**: `https://public.tableau.com/views/WORKBOOK/SHEET?:params`
- **Usage**: Business Dashboard page

### `VITE_DEBUG`
- **Type**: Boolean
- **Values**: `true` or `false`
- **Default**: `false`
- **Purpose**: Enable detailed console logs
- **Usage**: Optional, for debugging

## Git Configuration

### `.gitignore` Updates

```
# Environment files
.env.local
.env.*.local

# Keep example files
!.env.development
!.env.production
!.env.example
```

**What's tracked**:
- ✅ `.env.development` (team defaults)
- ✅ `.env.production` (production defaults)
- ✅ `.env.example` (documentation)

**What's ignored**:
- ❌ `.env.local` (personal overrides)
- ❌ `.env.development.local` (personal dev overrides)
- ❌ `.env.production.local` (personal prod overrides)

## Usage Examples

### Local Development (with backend)
```bash
# Uses .env.development automatically
npm run dev
# → VITE_DEMO_MODE=false
# → Calls http://localhost:8000
```

### Local Development (demo mode)
```bash
# Create personal override
echo "VITE_DEMO_MODE=true" > .env.local

npm run dev
# → VITE_DEMO_MODE=true
# → Uses static JSON
```

### Production Build
```bash
# Uses .env.production automatically
npm run build:prod
# → VITE_DEMO_MODE=true
# → Uses static JSON
```

### Custom Backend URL
```bash
# Create .env.local
cat > .env.local << EOF
VITE_DEMO_MODE=false
VITE_API_URL=https://my-backend-api.com
EOF

npm run dev
# → Calls https://my-backend-api.com
```

## Vercel Deployment

### Automatic Configuration
When deploying to Vercel, the production build will:
1. ✅ Use `.env.production` settings
2. ✅ Enable demo mode automatically
3. ✅ Load static JSON data
4. ✅ No backend required

### Manual Environment Variables (Optional)
If you want to override in Vercel dashboard:

1. Go to Project Settings → Environment Variables
2. Add variables:
   - `VITE_DEMO_MODE`: `true`
   - `VITE_TABLEAU_EMBED_URL`: (your URL)
   - `VITE_API_URL`: (leave empty for demo mode)

## Build Verification

✅ **Build Status**: PASSING

```bash
$ npm run build:prod

vite v5.4.11 building for production...
transforming...
✓ 2922 modules transformed.
rendering chunks...
dist/index.html                              2.17 kB
dist/assets/index-BpinWvTV.css              90.87 kB
dist/assets/index-C95l4wYF.js            1,068.37 kB
✓ built in 3.96s
```

**Notes**:
- ✅ All modules compiled successfully
- ⚠️  Large bundle size (1MB) - normal for React app with many dependencies
- ℹ️  Type import warnings are cosmetic and don't affect functionality

## Architecture

### Environment Switching

```
┌─────────────────────────────────────────┐
│         Vite Build Process              │
└──────────────┬──────────────────────────┘
               │
               ├─→ npm run dev  → .env.development
               ├─→ npm run build → .env.production
               └─→ .env.local (overrides all)
```

### Runtime Switching

```
┌──────────────────────────────────────────┐
│  frontend/src/config/environment.ts      │
│                                          │
│  export const config = {                 │
│    isDemoMode: import.meta.env.VITE_DEMO_MODE === 'true',  │
│    apiUrl: import.meta.env.VITE_API_URL, │
│  }                                       │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  frontend/src/api/unifiedApi.ts          │
│                                          │
│  export const api = config.isDemoMode    │
│    ? { ...demoAPI }                      │
│    : { ...realAPI }                      │
└──────────────────────────────────────────┘
```

## Next Steps

### For Local Development
1. ✅ Start backend server: `cd backend && uvicorn main:app --reload`
2. ✅ Start frontend: `cd frontend && npm run dev`
3. ✅ Open http://localhost:5173

### For Vercel Deployment
1. ✅ Build completed: `npm run build:prod`
2. ✅ Ready to deploy: `vercel` CLI or GitHub integration
3. ✅ Demo mode enabled automatically

### Optional Enhancements
- Configure custom domain in Vercel
- Set up staging environment (.env.staging)
- Add analytics (Google Analytics, Plausible)
- Configure error tracking (Sentry)

## Troubleshooting

### Environment Variables Not Working
**Problem**: Changes to `.env` files not reflected

**Solution**:
```bash
# Restart dev server
npm run dev

# Clear cache and rebuild
rm -rf node_modules/.vite
npm run dev
```

### Build Fails with "VITE_ Not Defined"
**Problem**: Environment variable not accessible

**Solution**:
- Ensure variable name starts with `VITE_`
- Check variable is in correct `.env` file
- Restart dev server after changes

### Demo Mode Not Activating
**Problem**: Banner not showing, still making API calls

**Solution**:
```bash
# Check current value
echo $VITE_DEMO_MODE

# Verify in browser console
console.log(import.meta.env.VITE_DEMO_MODE)
```

### Tableau Not Loading
**Problem**: Business Dashboard shows error

**Solution**:
- Verify URL format: `https://public.tableau.com/views/WORKBOOK/SHEET?...`
- Check URL is quoted in `.env` file
- Test URL in browser first

## Summary

✅ **Environment Configuration**: COMPLETE
✅ **Development Setup**: READY
✅ **Production Build**: WORKING
✅ **Git Configuration**: OPTIMIZED
✅ **Documentation**: COMPLETE

Your frontend now supports seamless switching between demo mode and live mode through environment variables!
