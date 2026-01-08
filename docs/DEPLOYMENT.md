# RetailPRED Deployment Guide

Complete guide for deploying RetailPRED to various platforms. This guide covers Vercel (demo mode), Docker (full stack), and manual deployment options.

---

## Table of Contents

- [Deployment Options Overview](#deployment-options-overview)
- [Prerequisites](#prerequisites)
- [Option 1: Vercel Deployment (Demo Mode)](#option-1-vercel-deployment-demo-mode)
- [Option 2: Docker Deployment (Full Stack)](#option-2-docker-deployment-full-stack)
- [Option 3: Manual Deployment](#option-3-manual-deployment)
- [Updating Demo Data](#updating-demo-data)
- [Custom Domain Setup](#custom-domain-setup)
- [Troubleshooting](#troubleshooting)

---

## Deployment Options Overview

| Option | Use Case | Backend Required | Cost | Complexity |
|--------|----------|------------------|------|------------|
| **Vercel (Demo)** | Portfolio, demos, showcases | ❌ No | Free tier | ⭐ Easy |
| **Docker** | Production, on-premise | ✅ Yes | Varies | ⭐⭐ Medium |
| **Manual** | Custom infrastructure | ✅ Yes | Varies | ⭐⭐⭐ Hard |

### Demo Mode vs. Full Stack

**Demo Mode (Vercel)**:
- ✅ Static JSON data (no backend)
- ✅ Pre-generated predictions
- ✅ Fast loading
- ✅ Zero-cost deployment
- ❌ No real-time forecasting
- ❌ Static data only

**Full Stack (Docker/Manual)**:
- ✅ Real-time predictions
- ✅ Live API access
- ✅ Database integration
- ✅ Full feature set
- ❌ Requires backend infrastructure
- ❌ Higher complexity

---

## Prerequisites

### For Vercel Deployment:
- [ ] Vercel account (sign up at https://vercel.com)
- [ ] GitHub account (for automatic deployments)
- [ ] Node.js 18+ (local testing)

### For Docker Deployment:
- [ ] Docker and Docker Compose installed
- [ ] 4GB RAM minimum
- [ ] 10GB disk space

### For Manual Deployment:
- [ ] Node.js 18+ and Python 3.9+
- [ ] Web server (nginx/Apache) or hosting platform
- [ ] Database server (SQLite for development, PostgreSQL for production)

---

## Option 1: Vercel Deployment (Demo Mode)

### Overview

Vercel deployment uses **demo mode** - the frontend serves static JSON files containing pre-generated predictions. No backend is required.

### Step 1: Install Vercel CLI

```bash
npm install -g vercel
```

### Step 2: Build Frontend Locally

```bash
cd frontend
npm install
npm run build:prod
```

**Verify build**:
- Check `frontend/dist/` folder exists
- Verify `demo-data/` folder contains JSON files
- Total size should be ~1.7 MB

### Step 3: Deploy to Vercel

#### Option A: CLI Deployment

```bash
cd frontend
vercel
```

**Follow the prompts**:
```
? Set up and deploy "~/retailPRED/frontend"? [Y/n] Y
? Which scope do you want to deploy to? Your Username
? Link to existing project? [y/N] N
? What's your project's name? retailpred-frontend
? In which directory is your code located? ./
? Want to override the settings? [y/N] N
```

Vercel will detect the configuration from `vercel.json` automatically.

#### Option B: GitHub Integration (Recommended)

1. **Push code to GitHub**:
   ```bash
   git add .
   git commit -m "Add Vercel deployment"
   git push origin main
   ```

2. **Connect to Vercel**:
   - Go to https://vercel.com/dashboard
   - Click "Add New Project"
   - Import from GitHub
   - Select `retailpred` repository
   - Configure:
     - **Root Directory**: `frontend`
     - **Framework Preset**: Vite (auto-detected)
     - **Build Command**: `npm run build` (auto-detected)
     - **Output Directory**: `dist` (auto-detected)

3. **Deploy**:
   - Click "Deploy"
   - Vercel will build and deploy automatically
   - Get URL: `https://retailpred.vercel.app`

### Step 4: Configure Environment Variables

In Vercel Dashboard:
1. Go to **Settings → Environment Variables**
2. Add variables:
   ```
   VITE_DEMO_MODE=true
   VITE_TABLEAU_EMBED_URL=https://public.tableau.com/views/BOOK_NAME/SHEET_NAME?:params
   ```

**Note**: Replace the Tableau URL with your actual visualization URL.

### Step 5: Verify Deployment

1. **Visit your URL**: `https://your-project.vercel.app`
2. **Check**:
   - ✅ Page loads within 3 seconds
   - ✅ Demo banner visible at top
   - ✅ All navigation works
   - ✅ Demo data loads (check Network tab)
   - ✅ No console errors

### Automatic Deployments

With GitHub integration, every push to `main` triggers automatic deployment:

```bash
# Make changes
git add .
git commit -m "Update predictions"
git push origin main

# Vercel automatically redeploys
```

### Preview Deployments

Every pull request gets a preview URL:

```bash
git checkout -b feature/new-feature
# Make changes
git push origin feature/new-feature

# Vercel creates preview: https://retailpred-git-feature-new-feature.vercel.app
```

---

## Option 2: Docker Deployment (Full Stack)

### Overview

Docker deployment runs the complete application stack: backend API, frontend, and database.

### Step 1: Build Docker Images

```bash
# Build frontend
cd frontend
docker build -t retailpred-frontend .

# Build backend (if separate)
cd ../backend
docker build -t retailpred-backend .
```

### Step 2: Configure Environment

Create `.env` file in project root:

```bash
# Backend
DATABASE_URL=sqlite:///data/retailpred.db
API_HOST=0.0.0.0
API_PORT=8000

# Frontend (for local development)
VITE_API_URL=http://localhost:8000
VITE_DEMO_MODE=false
```

### Step 3: Start Services

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### Step 4: Access Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

### Docker Management Commands

```bash
# Stop services
docker-compose down

# Restart services
docker-compose restart

# View logs for specific service
docker-compose logs -f frontend
docker-compose logs -f backend

# Execute commands in container
docker-compose exec backend bash
docker-compose exec frontend sh

# Rebuild after changes
docker-compose up -d --build
```

### Production Considerations

1. **Use PostgreSQL** instead of SQLite
2. **Enable HTTPS** with nginx reverse proxy
3. **Set up monitoring** (Prometheus, Grafana)
4. **Configure backups** for database
5. **Use environment variables** for secrets

---

## Option 3: Manual Deployment

### Frontend Deployment

#### Step 1: Build Frontend

```bash
cd frontend
npm install
npm run build:prod
```

#### Step 2: Deploy to Static Host

**AWS S3 + CloudFront**:
```bash
# Install AWS CLI
pip install awscli

# Sync to S3
aws s3 sync dist/ s3://your-bucket-name --delete

# Invalidate CloudFront cache
aws cloudfront create-invalidation --distribution-id YOUR_DIST_ID --paths "/*"
```

**Netlify**:
```bash
# Install Netlify CLI
npm install -g netlify-cli

# Deploy
netlify deploy --prod --dir=frontend/dist
```

**GitHub Pages**:
1. Create `gh-pages` branch
2. Push `dist/` contents to branch
3. Enable GitHub Pages in repository settings

### Backend Deployment

#### Step 1: Prepare Backend

```bash
cd backend
pip install -r requirements.txt

# Or use Docker
docker build -t retailpred-backend .
```

#### Step 2: Deploy to Server

**Using systemd** (Linux):

1. Create service file `/etc/systemd/system/retailpred.service`:
```ini
[Unit]
Description=RetailPRED Backend API
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/var/www/retailpred/backend
Environment="PATH=/var/www/retailpred/venv/bin"
ExecStart=/var/www/retailpred/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

2. Enable and start:
```bash
sudo systemctl enable retailpred
sudo systemctl start retailpred
sudo systemctl status retailpred
```

**Using nginx reverse proxy**:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location / {
        root /var/www/retailpred/frontend/dist;
        try_files $uri $uri/ /index.html;
    }
}
```

---

## Updating Demo Data

### Export Predictions from Database

```bash
# Run export script
cd backend
python scripts/export_demo_data.py
```

This creates:
- `frontend/public/demo-data/predictions.json`
- `frontend/public/demo-data/economic-indicators.json`
- `frontend/public/demo-data/summary.json`

### Manual Export

```python
import sqlite3
import json

# Connect to database
conn = sqlite3.connect('data/retailpred.db')
cursor = conn.cursor()

# Export predictions
cursor.execute("""
    SELECT * FROM prediction_log
    WHERE prediction_date >= '2025-01-01'
""")
predictions = cursor.fetchall()

with open('frontend/public/demo-data/predictions.json', 'w') as f:
    json.dump(predictions, f, indent=2)

conn.close()
```

### Rebuild and Redeploy

After updating demo data:

```bash
cd frontend
npm run build:prod

# Deploy to Vercel
vercel --prod

# Or commit and push (triggers auto-deploy)
git add public/demo-data/
git commit -m "Update demo data"
git push origin main
```

---

## Custom Domain Setup

### Vercel Custom Domain

1. **Add domain in Vercel**:
   - Go to **Settings → Domains**
   - Click "Add Domain"
   - Enter: `retailpred.yourdomain.com`

2. **Configure DNS**:
   - If using A record:
     ```
     Type: A
     Name: retailpred
     Value: 76.76.21.21
     ```
   
   - If using CNAME (recommended):
     ```
     Type: CNAME
     Name: retailpred
     Value: cname.vercel-dns.com
     ```

3. **Wait for propagation** (up to 24 hours)

4. **Enable HTTPS** (automatic with Vercel)

### Docker with Custom Domain

Using nginx:

```nginx
server {
    listen 80;
    server_name retailpred.yourdomain.com;

    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl;
    server_name retailpred.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/retailpred/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/retailpred/privkey.pem;

    location /api {
        proxy_pass http://backend:8000;
    }

    location / {
        root /usr/share/nginx/html;
        try_files $uri $uri/ /index.html;
    }
}
```

---

## Troubleshooting

### Build Failures

**Problem**: Build fails on Vercel

**Solutions**:
1. Check build logs in Vercel dashboard
2. Verify all dependencies in `package.json`
3. Ensure build runs locally: `npm run build:prod`
4. Clear Vercel cache: Settings → Git → Ignored Build Step

### Demo Mode Not Working

**Problem**: App tries to call API instead of using static JSON

**Solutions**:
1. Verify `VITE_DEMO_MODE=true` in environment variables
2. Check case sensitivity (must be all caps)
3. Redeploy after adding environment variable
4. Check browser console for `config.isDemoMode` value

### 404 Errors on Routes

**Problem**: Page refresh shows 404 error

**Solutions**:
1. Verify `vercel.json` has rewrites configuration
2. Check `outputDirectory` is `dist`
3. Ensure `buildCommand` is `npm run build`
4. Clear cache and redeploy

### Tableau Not Loading

**Problem**: Business Dashboard shows error

**Solutions**:
1. Verify Tableau URL format is correct
2. Check URL is in quotes in environment variable
3. Test URL in browser first
4. Check for console errors (CORS, X-Frame-Options)

### Docker Container Won't Start

**Problem**: Services fail to start

**Solutions**:
1. Check logs: `docker-compose logs -f`
2. Verify ports are not in use: `lsof -i :8000`
3. Rebuild images: `docker-compose build --no-cache`
4. Check disk space: `df -h`

### Slow Performance

**Problem**: Site loads slowly

**Solutions**:
1. Enable gzip compression (Vercel does this automatically)
2. Optimize images
3. Implement code splitting
4. Use CDN for static assets
5. Enable browser caching

### Environment Variables Not Working

**Problem**: Changes to `.env` don't take effect

**Solutions**:
1. Restart dev server after changes
2. Clear cache: `rm -rf frontend/node_modules/.vite`
3. Verify variable names start with `VITE_`
4. Check variable is in correct `.env` file
5. For production, set variables in deployment platform dashboard

---

## Monitoring and Maintenance

### Vercel Analytics

Enable in Vercel Dashboard:
1. Go to **Analytics** tab
2. Enable Vercel Analytics
3. View metrics: page views, unique visitors, top pages

### Health Checks

For Docker deployments, add health check:

```yaml
# docker-compose.yml
services:
  backend:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Log Management

```bash
# View Vercel logs
vercel logs

# View Docker logs
docker-compose logs -f --tail=100

# Archive logs
docker-compose logs > logs_$(date +%Y%m%d).log
```

---

## Security Best Practices

### Environment Variables
- ✅ Never commit `.env.local` to git
- ✅ Use different values for dev/staging/prod
- ✅ Rotate secrets regularly
- ✅ Use Vercel/env vault for production

### API Security
- ✅ Enable HTTPS in production
- ✅ Implement rate limiting
- ✅ Add authentication for non-demo mode
- ✅ Validate all inputs

### Dependencies
- ✅ Run `npm audit` regularly
- ✅ Keep dependencies updated
- ✅ Use `npm ci` for production builds
- ✅ Lock file integrity

---

## Cost Estimate

### Vercel (Demo Mode)
- **Hobby Plan**: FREE
  - 100GB bandwidth
  - Unlimited deployments
  - Automatic HTTPS
  - Team collaboration

### Docker (Self-Hosted)
- **Minimum**: $5-10/month (DigitalOcean, Linode)
  - 2GB RAM, 1 CPU
  - 50GB SSD
- **Recommended**: $20-40/month
  - 4GB RAM, 2 CPUs
  - 80GB SSD
  - Better performance

### AWS (Production)
- **S3 + CloudFront**: $10-20/month
- **EC2 + RDS**: $50-100/month
- **Elastic Beanstalk**: $40-80/month

---

## Next Steps

1. **Deploy to Vercel** (easiest option)
2. **Test all features** on deployed URL
3. **Set up monitoring** and alerts
4. **Configure custom domain** (optional)
5. **Enable CI/CD** for automatic deployments

For detailed architecture information, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

**Last Updated**: January 7, 2025
**Maintained By**: RetailPRED Team
