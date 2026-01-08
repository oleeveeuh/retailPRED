# Quick Start: Deploy to Vercel

Fast-track deployment guide for RetailPRED.

## Prerequisites

✅ All preparation complete - ready to deploy in 5 minutes

## Step 1: Commit & Push (2 minutes)

```bash
# Add all changes
git add .

# Commit with descriptive message
git commit -m "Prepare repository for deployment

- Add CI/CD workflow for build validation
- Fix .gitignore to track package-lock.json
- Create comprehensive documentation (DEPLOYMENT.md, ARCHITECTURE.md)
- Add portfolio preparation guides
- Add GitHub topics and documentation structure
- Verify build and React 19 compatibility

🤖 Generated with Claude Code"

# Push to GitHub
git push origin main
```

## Step 2: Deploy to Vercel (3 minutes)

### Option A: Vercel CLI (Recommended)

```bash
# Install Vercel CLI (if not installed)
npm i -g vercel

# Login to Vercel
vercel login

# Deploy to production
vercel --prod
```

### Option B: Vercel Dashboard

1. Go to https://vercel.com/new
2. Import repository: `oleeveeuh/retailPRED`
3. Settings:
   - **Framework Preset**: Vite
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build:prod`
   - **Output Directory**: `dist`
4. Environment Variables (not needed for demo mode):
   ```
   VITE_DEMO_MODE=true
   VITE_API_URL=
   VITE_TABLEAU_EMBED_URL=
   ```
5. Click **Deploy**

## Step 3: Verify Deployment (1 minute)

After deployment completes:

1. **Visit your URL**: `https://retailpred.vercel.app` (or your custom URL)
2. **Test key features**:
   - ✅ Page loads without errors
   - ✅ Select a retail category (e.g., "Total Retail Sales")
   - ✅ View forecast visualization
   - ✅ Check SHAP explainability
   - ✅ Try economic scenarios

## Step 4: Post-Deployment Tasks (Optional but Recommended)

### Add GitHub Topics (1 minute)
1. Go to repository Settings → Topics
2. Add these 16 topics:
   ```
   machine-learning forecasting time-series economics react typescript
   python fastapi data-visualization shap explainable-ai retail
   sales-forecasting macroeconomics lightgbm dashboard
   ```

### Capture Screenshots (10 minutes)
1. Open deployed application
2. Take 5 screenshots:
   - Dashboard overview (11 categories)
   - Forecast chart (7 models)
   - SHAP feature importance
   - Economic scenarios
   - Model comparison
3. Save to `docs/images/screenshots/`

### Update README (5 minutes)
1. Add screenshots section:
   ```markdown
   ## 📸 Screenshots

   ![Dashboard Overview](docs/images/screenshots/dashboard-overview.png)
   ![Forecast Visualization](docs/images/screenshots/forecast-visualization.png)
   ```

2. Add live demo badge at top (if not already present):
   ```markdown
   [![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://retailpred.vercel.app)
   ```

## Troubleshooting

### Build Fails on Vercel

**Issue**: Build fails with error

**Solution**:
```bash
# Test build locally first
cd frontend
npm run build:prod

# If successful, check Vercel build logs
# Common issues:
# - Wrong root directory (should be "frontend")
# - Wrong build command (should be "npm run build:prod")
# - Missing environment variables (not needed for demo mode)
```

### Demo Data Not Loading

**Issue**: Predictions or economic indicators not showing

**Solution**:
```bash
# Verify demo-data files are built
ls -lh frontend/dist/demo-data/

# Should see:
# predictions.json (414 KB)
# economic-indicators.json (110 KB)
# summary.json (883 B)

# If missing, rebuild:
cd frontend
npm run build:prod
```

### CI/CD Workflow Fails

**Issue**: GitHub Actions workflow fails

**Solution**:
1. Check workflow logs in Actions tab
2. Common issue: npm ci fails if package-lock.json not tracked
3. Verify .gitignore doesn't ignore package-lock.json
4. Re-run workflow:

```bash
# Fix .gitignore if needed
git add .gitignore
git commit -m "Fix .gitignore for package-lock.json"
git push
```

## Success Checklist

- [ ] Code pushed to GitHub
- [ ] Deployed to Vercel successfully
- [ ] Live demo URL works
- [ ] All 11 retail categories load
- [ ] Forecast visualization displays
- [ ] SHAP explainability works
- [ ] Economic scenarios functional
- [ ] GitHub topics added
- [ ] Screenshots captured
- [ ] README updated with screenshots

## Next Steps

After successful deployment:

1. **Share on LinkedIn**:
   ```
   🚀 Just deployed RetailPRED - a macroeconomic retail sales forecasting platform
   using multi-model ensemble & SHAP explainability!

   Live Demo: https://retailpred.vercel.app
   GitHub: https://github.com/oleeveeuh/retailPRED

   #MachineLearning #Forecasting #React #TypeScript
   ```

2. **Add to Resume** (see [PORTFOLIO_PREPARATION.md](PORTFOLIO_PREPARATION.md))

3. **Prepare for Interviews** (talking points in Portfolio guide)

4. **Monitor Analytics**:
   - Vercel Analytics Dashboard
   - GitHub repository traffic
   - Google Analytics (if added)

## Support Documents

- [Full Deployment Guide](docs/DEPLOYMENT.md)
- [Pre-Deployment Checklist](PRE_DEPLOYMENT_CHECKLIST.md)
- [Portfolio Preparation](PORTFOLIO_PREPARATION.md)
- [Architecture Documentation](docs/ARCHITECTURE.md)

---

**Time Estimate**: 10-15 minutes total
**Difficulty**: Beginner-friendly
**Status**: ✅ Ready to Deploy

Good luck! 🚀
