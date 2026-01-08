# 🚀 RetailPRED Deployment Checklist

## Pre-Deployment Checklist

### ✅ Environment Setup
- [x] `.env.development` created for local development
- [x] `.env.production` created for Vercel deployment
- [x] `.env.example` created as template
- [x] `.gitignore` configured for environment files
- [x] Build verification completed successfully

### ✅ Build Status
```
Build Time: 3.96s
Output: dist/
Bundle Size: 1.068 MB (gzipped: 310.92 kB)
Total Dist Size: 1.7 MB
Status: ✅ PASSING
```

### ✅ Demo Data Included
- `dist/demo-data/predictions.json` (414 KB)
- `dist/demo-data/economic-indicators.json` (110 KB)
- `dist/demo-data/summary.json` (883 B)

## Deployment Options

### Option 1: Vercel (Recommended for Demo)

**Setup Steps:**

1. **Install Vercel CLI**
   ```bash
   npm install -g vercel
   ```

2. **Deploy Frontend**
   ```bash
   cd frontend
   vercel
   # Follow prompts:
   # - Set project name: retailpred-frontend
   # - Framework: Vite
   # - Build directory: dist
   # - Output directory: (leave empty)
   ```

3. **Environment Variables in Vercel Dashboard**
   - Go to: https://vercel.com/your-username/retailpred-frontend/settings/environment-variables
   - Add variables:
     ```
     VITE_DEMO_MODE=true
     VITE_TABLEAU_EMBED_URL=https://public.tableau.com/views/Book1_17676501972860/Sheet1?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link
     ```

4. **Custom Domain (Optional)**
   - Add custom domain in Vercel dashboard
   - Update DNS records as instructed

**Benefits:**
- ✅ No backend required
- ✅ Fast global CDN
- ✅ Automatic HTTPS
- ✅ Free tier available

### Option 2: Docker Deployment (Full Stack)

**Frontend Only (Docker):**
```bash
# Build image
docker build -t retailpred-frontend frontend/

# Run container
docker run -d -p 3000:80 retailpred-frontend

# Access at http://localhost:3000
```

**Full Stack (with Docker Compose):**
```bash
# Start backend and frontend together
docker-compose up -d

# Access:
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### Option 3: Manual Deployment

**Build for Production:**
```bash
cd frontend
npm run build:prod
```

**Deploy to Any Static Host:**
- Upload `dist/` folder contents to:
  - AWS S3 + CloudFront
  - Netlify
  - GitHub Pages
  - Your own web server

## Verification Steps

### After Deployment

1. **Check Home Page**
   - Navigate to deployed URL
   - Should see Dashboard page
   - Demo banner visible (top of page)

2. **Test Predictions Page**
   - Click "Predictions" in sidebar
   - Should load prediction history
   - Filters should work
   - Data comes from static JSON

3. **Test Models Page**
   - Click "Models" in sidebar
   - Should show model comparison cards
   - Metrics should display correctly

4. **Test Explainability Page**
   - Click "Explainability" in sidebar
   - Should show SHAP value charts
   - Feature importance should load

5. **Test Business Dashboard**
   - Click "Business Dashboard" in sidebar
   - Tableau visualization should load
   - No iframe errors in console

6. **Check Browser Console**
   - Open DevTools (F12)
   - Look for errors (should be none)
   - Verify `VITE_DEMO_MODE: true` in logs

## Environment Variables Reference

### Development (Local)
```bash
# File: frontend/.env.development
VITE_DEMO_MODE=false              # Use backend API
VITE_API_URL=http://localhost:8000
VITE_TABLEAU_EMBED_URL="https://..."
```

### Production (Vercel)
```bash
# File: frontend/.env.production
VITE_DEMO_MODE=true               # Use static JSON
VITE_API_URL=
VITE_TABLEAU_EMBED_URL="https://..."
```

### Custom Overrides (Optional)
```bash
# File: frontend/.env.local (gitignored)
VITE_DEBUG=true                   # Enable debug logs
# Override any variable as needed
```

## Deployment Status

| Environment | Status | URL | Mode |
|------------|--------|-----|------|
| Local | ✅ Ready | http://localhost:5173 | Demo API |
| Production Build | ✅ Ready | - | Static Demo |
| Vercel | ⏳ Pending | - | Static Demo |
| Docker | ✅ Ready | - | Full Stack |

**Last Updated**: January 7, 2025
