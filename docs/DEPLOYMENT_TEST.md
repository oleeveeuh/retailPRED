# Pre-Deployment Test Checklist

## Build Information
- **Build Date**: January 7, 2025
- **Build Time**: 6.09s
- **Total Bundle Size**: 1.7 MB
- **Main JS Bundle**: 1,068 KB (gzipped: 310.94 kB)
- **CSS Bundle**: 90.87 KB (gzipped: 13.58 kB)

## Test the Preview

**URL**: http://localhost:4173

Start the preview server:
```bash
cd frontend
npm run preview
```

---

## Critical Tests

### ✅ Core Functionality

- [ ] **Site loads without errors**
  - Open browser to http://localhost:4173
  - Page should render within 2-3 seconds
  - No white screens or crash errors

- [ ] **Demo banner is visible**
  - Blue banner at top of page
  - Text: "📊 Demo Mode"
  - Link to GitHub repository

- [ ] **Dashboard displays**
  - Summary cards show metrics
  - Charts render (ForecastChart, ModelInfoCard)
  - No "undefined" or "NaN" values

### ✅ Predictions Page

- [ ] **Navigate to /predictions**
  - Click "Predictions" in sidebar
  - Prediction history table loads
  - **Data comes from static JSON** (check Network tab)

- [ ] **Filters work**
  - Try model filter
  - Try date range filter
  - Try category filter
  - Results update correctly

- [ ] **Pagination works**
  - Scroll to bottom
  - Load more button appears
  - Clicking loads more predictions

### ✅ Models Page

- [ ] **Navigate to /models**
  - Click "Models" in sidebar
  - Model comparison cards display
  - 7 model types shown:
    - LGBM
    - RandomForest
    - AutoARIMA
    - AutoETS
    - SeasonalNaive
    - PatchTST
    - TimesNet

- [ ] **Metrics display correctly**
  - RMSE values shown
  - MAE values shown
  - MAPE values shown
  - No "undefined" values

### ✅ Explainability Page

- [ ] **Navigate to /explain**
  - Click "Explainability" in sidebar
  - Category selector works
  - Model selector works

- [ ] **SHAP charts render**
  - Feature importance chart loads
  - Bar chart displays
  - Colors are correct (blue for positive, red for negative)

- [ ] **Feature explanations**
  - Top features listed
  - Descriptions show
  - Impact percentages display

### ✅ Business Dashboard

- [ ] **Navigate to /business-dashboard**
  - Click "Business Dashboard" in sidebar
  - **Tableau visualization loads**
  - Visualization fills container (not small)

- [ ] **Tableau embed works**
  - No iframe errors in console
  - Visualization is interactive
  - Toolbar is visible
  - No "X-Frame-Options" errors

### ✅ Validation Page

- [ ] **Navigate to /validation**
  - Click "Validation" in sidebar
  - Validation table loads
  - Filters work

### ✅ Navigation

- [ ] **All routes work**
  - http://localhost:4173/ - Dashboard ✅
  - http://localhost:4173/predictions - Predictions ✅
  - http://localhost:4173/models - Models ✅
  - http://localhost:4173/explain - Explainability ✅
  - http://localhost:4173/validation - Validation ✅
  - http://localhost:4173/business-dashboard - Business Dashboard ✅

- [ ] **Page refresh works**
  - Refresh each page
  - No 404 errors
  - Content still displays

- [ ] **Browser back/forward works**
  - Navigate between pages
  - Use browser back button
  - Use browser forward button
  - Correct page displays

---

## Console Checks (F12)

### ✅ No Errors

Open browser DevTools (F12) and check Console tab:

- [ ] **No 404 errors**
  - All demo-data files return 200
  - No "Failed to load resource" errors

- [ ] **No TypeScript errors**
  - No type errors in console
  - No "is not defined" errors

- [ ] **No runtime errors**
  - No "Cannot read property of undefined"
  - No "Cannot read property of null"
  - No React errors

### ✅ Network Verification

Check Network tab in DevTools:

- [ ] **Demo data files load**
  - `/demo-data/predictions.json` - Status: 200, Size: ~414 KB
  - `/demo-data/economic-indicators.json` - Status: 200, Size: ~110 KB
  - `/demo-data/summary.json` - Status: 200, Size: ~883 B

- [ ] **No API calls to backend**
  - No requests to `localhost:8000`
  - No requests to `127.0.0.1:8000`
  - All data from `/demo-data/` folder

- [ ] **Assets load correctly**
  - `index.html` - 200
  - `assets/index-*.js` - 200
  - `assets/index-*.css` - 200
  - `favicon.svg` - 200

### ✅ Performance Checks

Check Performance tab (optional):

- [ ] **Time to Interactive**: < 5 seconds
- [ ] **First Contentful Paint**: < 2 seconds
- [ ] **Largest Contentful Paint**: < 3 seconds

---

## Mobile Responsive Tests

### ✅ Responsive Design

- [ ] **Test on mobile viewport**
  - Open Chrome DevTools (F12)
  - Click device toolbar (Ctrl+Shift+M / Cmd+Shift+M)
  - Select iPhone 12 Pro or similar
  - Page layout adapts correctly

- [ ] **Sidebar behavior**
  - Hamburger menu appears on mobile
  - Sidebar collapses/expands correctly
  - Overlay works when sidebar open

- [ ] **Charts are readable**
  - Forecast chart not squashed
  - Model cards stack vertically
  - Tableau visualization usable

- [ ] **Touch interactions work**
  - Can tap navigation items
  - Can tap buttons
  - Can scroll tables

---

## Edge Cases

### ✅ Error Handling

- [ ] **Missing demo data**
  - Temporarily rename a demo-data file
  - Page shows error message
  - No crash/white screen

- [ ] **Invalid URL parameters**
  - Visit `/predictions?model=invalid`
  - Page still loads
  - No errors shown

- [ ] **Browser back button**
  - Navigate multiple pages
  - Use back button repeatedly
  - Correct state restored

---

## Browser Compatibility

### ✅ Test in Multiple Browsers

- [ ] **Chrome/Edge** (Chromium)
  - All features work
  - No console errors

- [ ] **Firefox**
  - All features work
  - No console errors

- [ ] **Safari** (if on Mac)
  - All features work
  - No console errors

---

## File Verification

### ✅ Build Output Structure

Verify the `dist/` folder contains:

```
dist/
├── index.html                    ✅
├── favicon.svg                   ✅
├── health.html                   ✅
├── manifest.json                 ✅
├── assets/
│   ├── index-DOJwkACc.js        ✅ (1.0 MB)
│   ├── index-BpinWvTV.css       ✅ (89 KB)
│   └── confetti.module-wUsLuJ1J.js  ✅ (10 KB)
└── demo-data/
    ├── predictions.json          ✅ (414 KB)
    ├── economic-indicators.json  ✅ (110 KB)
    └── summary.json              ✅ (883 B)
```

---

## Performance Metrics

### ✅ Bundle Size

- [ ] **Main JS bundle**: 1.0 MB ✅
  - Gzipped: 310.94 KB ✅
  - Acceptable for production

- [ ] **CSS bundle**: 89 KB ✅
  - Gzipped: 13.58 KB ✅
  - Good size

- [ ] **Demo data**: 525 KB ✅
  - Compressed by browser
  - Loaded once, cached

### ✅ Load Times

- [ ] **Initial page load**: < 3 seconds
- [ ] **Route transitions**: < 500ms
- [ ] **Data loading**: < 1 second (from static JSON)

---

## Accessibility Checks

### ✅ Basic Accessibility

- [ ] **Keyboard navigation works**
  - Tab through page elements
  - Enter key activates buttons
  - Focus indicators visible

- [ ] **Screen reader friendly**
  - Alt text on images
  - Semantic HTML elements
  - ARIA labels where needed

- [ ] **Color contrast**
  - Text is readable
  - No low-contrast issues

---

## Pre-Deployment Sign-Off

### ✅ Ready to Deploy?

- [ ] All critical tests pass ✅
- [ ] No console errors ✅
- [ ] All pages load correctly ✅
- [ ] Demo data loads successfully ✅
- [ ] Mobile responsive ✅
- [ ] Performance acceptable ✅

### 🔴 Issues Found

Document any issues discovered during testing:

```
Issue #1:
- Location: [Page/Component]
- Description: [What's wrong]
- Severity: [Critical/High/Medium/Low]
- Status: [Fixed/Needs Fixing/Acceptable]

```

### 📝 Notes

Add any additional notes or observations:

```

```

---

## Next Steps After Testing

If all tests pass:

1. ✅ **Deploy to Vercel**
   ```bash
   cd frontend
   vercel
   ```

2. ✅ **Set environment variables** in Vercel dashboard:
   - `VITE_DEMO_MODE=true`
   - `VITE_TABLEAU_EMBED_URL=https://...`

3. ✅ **Test production deployment**
   - Visit deployed URL
   - Repeat critical tests
   - Verify demo mode works

4. ✅ **Configure custom domain** (optional)

If issues found:

1. 🔧 **Fix critical issues**
2. 🔄 **Rebuild and retest**
3. ✅ **Document workarounds**
4. 🚀 **Deploy when ready**

---

**Test Date**: ___________

**Tester**: ___________

**Result**: ✅ PASS / 🔴 FAIL

**Approved for Deployment**: Yes / No

---

*Last Updated: January 7, 2025*
