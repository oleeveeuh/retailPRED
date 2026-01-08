# RetailPRED AWS Architecture Guide

## Table of Contents
- [Current Local Architecture](#current-local-architecture)
- [AWS Production Architecture](#aws-production-architecture)
- [Component Mapping](#component-mapping)
- [Data Flow Diagrams](#data-flow-diagrams)
- [Cost Estimates](#cost-estimates)
- [Security Best Practices](#security-best-practices)
- [Scaling Considerations](#scaling-considerations)

---

## Current Local Architecture

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LOCAL DEVELOPMENT                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐                                               │
│  │  Frontend (Dev)  │                                               │
│  │  React 19.2      │                                               │
│  │  Vite 7.2        │                                               │
│  │  Port: 5173      │                                               │
│  └────────┬─────────┘                                               │
│           │                                                          │
│           │ HTTP/JSON                                                │
│           ▼                                                          │
│  ┌──────────────────┐                                               │
│  │   Backend API    │                                               │
│  │   FastAPI        │                                               │
│  │   Uvicorn        │                                               │
│  │   Port: 8000     │                                               │
│  └────────┬─────────┘                                               │
│           │                                                          │
│           │                                                          │
│           ├──► SQLite Database (./data/retailpred.db)               │
│           │   - prediction_log                                       │
│           │   - model_metadata                                       │
│           │   - time_series_data                                     │
│           │   - categories                                           │
│           │                                                          │
│           ├──► ML Models (./backend/ml/models/)                      │
│           │   - LightGBM (.pkl)                                      │
│           │   - RandomForest (.pkl)                                  │
│           │   - PatchTST (.pth)                                      │
│           │   - TimesNet (.pth)                                      │
│           │   - AutoARIMA/ETS (pickle)                               │
│           │                                                          │
│           └──► Logs (stdout/stderr)                                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Descriptions

#### Frontend (Local)
- **Technology**: React 19.2 + TypeScript + Vite 7.2
- **Development Server**: `npm run dev` runs on localhost:5173
- **Features**:
  - Hot module replacement (HMR)
  - React Query for API calls
  - Recharts for visualizations
  - TailwindCSS v4 for styling
  - Framer Motion for animations
- **Build Output**: `/frontend/dist/` (static files)

#### Backend (Local)
- **Technology**: FastAPI + Python 3.12
- **Development Server**: `uvicorn main:app --reload` on localhost:8000
- **Features**:
  - REST API endpoints
  - Automatic OpenAPI docs (`/docs`)
  - CORS middleware for frontend
  - Prediction service with ML models
  - SHAP explainability
  - Counterfactual analysis
- **Entry Point**: `/backend/main.py`

#### Database (Local)
- **Technology**: SQLite 3
- **Location**: `./data/retailpred.db`
- **Size**: ~50-100 MB (with 1 year of data)
- **Tables**:
  - `prediction_log` (all forecasts)
  - `model_metadata` (model info)
  - `time_series_data` (historical sales)
  - `categories` (business categories)
  - `external_factors` (economic indicators)

#### ML Models (Local)
- **Location**: `./backend/ml/models/` and `./training_outputs/`
- **Formats**:
  - `.pkl` (pickle) for scikit-learn, LightGBM
  - `.pth` for PyTorch models (PatchTST, TimesNet)
  - `.pkl` for statsmodels (AutoARIMA, AutoETS)
- **Count**: 77 models total (11 categories × 7 architectures)
- **Loading**: `joblib.load()` or `torch.load()` from local paths

#### Data Pipeline (Local)
- **ETL Scripts**: `/project_root/etl/`
  - `fetch_fred.py` - Economic data from FRED API
  - `fetch_mrts.py` - Retail sales from US Census
  - `fetch_yahoo.py` - Stock data from Yahoo Finance
- **Storage**: Raw data in `./project_root/data_raw/`
- **Processing**: Processed data in `./project_root/data_processed/`

#### Logging (Local)
- **Method**: Print statements to stdout/stderr
- **Viewing**: Terminal console where server runs
- **Level**: INFO for most operations
- **Format**: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`

---

## AWS Production Architecture

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AWS PRODUCTION ENVIRONMENT                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                           CloudFront CDN                             │  │
│  │                    (Global Edge Locations)                            │  │
│  │                    URL: https://retailpred.com                        │  │
│  └───────────────────────────────────────┬───────────────────────────────┘  │
│                                          │                                  │
│                                          │ Cache static assets               │
│                                          ▼                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                        S3 Bucket (Frontend)                          │  │
│  │                 retailpred-frontend-static                           │  │
│  │                                                                   │  │
│  │  Contents:                                                           │  │
│  │  - index.html                                                        │  │
│  │  - assets/*.js (bundled React app)                                   │  │
│  │  - assets/*.css (Tailwind styles)                                    │  │
│  │  - favicon.ico                                                       │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│                                          │                                  │
│                                          │ API calls (/api/*)               │
│                                          ▼                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         Application Load Balancer                     │  │
│  │                              (ALB/ELB)                                │  │
│  │                         SSL/TLS Termination                           │  │
│  └───────────────────────────────────────┬───────────────────────────────┘  │
│                                          │                                  │
│                                          │ HTTP traffic                     │
│                                          ▼                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                       Auto Scaling Group                              │  │
│  │                   (t3.medium instances)                               │  │
│  │                                                                      │  │
│  │  ┌────────────────────────────────────────────────────────────────┐ │  │
│  │  │                  EC2 Instance 1 (Primary)                       │ │  │
│  │  │                  AMI: retailpred-backend                        │ │  │
│  │  │                                                                  │ │  │
│  │  │  ┌──────────────────────────────────────────────────────────┐  │ │  │
│  │  │  │               Nginx Reverse Proxy                         │  │  │
│  │  │  │               Port 80/443 → Port 8000                    │  │  │
│  │  │  │               Static file serving                         │  │  │
│  │  │  │               SSL certificates (Let's Encrypt)            │  │  │
│  │  │  └──────────────────────────────────────────────────────────┘  │ │  │
│  │  │                            │                                     │ │  │
│  │  │                            ▼                                     │ │  │
│  │  │  ┌──────────────────────────────────────────────────────────┐  │ │  │
│  │  │  │              FastAPI Application                          │  │ │  │
│  │  │  │              Gunicorn workers (4x)                         │  │ │  │
│  │  │  │              Systemd service: retailpred-api              │  │ │  │
│  │  │  └──────────────────────────────────────────────────────────┘  │ │  │
│  │  │                            │                                     │ │  │
│  │  │                            ▼                                     │ │  │
│  │  │  ┌──────────────────────────────────────────────────────────┐  │ │  │
│  │  │  │              EBS Volume (100 GB)                           │  │ │  │
│  │  │  │              Mounted: /mnt/ebs                             │  │ │  │
│  │  │  │              /data/retailpred.db                           │  │ │  │
│  │  │  │              /models/ (cached from S3)                     │  │ │  │
│  │  │  └──────────────────────────────────────────────────────────┘  │ │  │
│  │  └────────────────────────────────────────────────────────────────┘ │  │
│  │                                                                      │  │
│  │  ┌────────────────────────────────────────────────────────────────┐ │  │
│  │  │                  EC2 Instance 2 (Standby)                      │ │  │
│  │  │                      (Auto-scaling)                             │  │  │
│  │  └────────────────────────────────────────────────────────────────┘ │  │
│  └────────────────────────────────────────────────────────────────────┘ │  │
│                                                                              │
│                                          │                                  │
│                                          │                                  │
│  ┌──────────────────────────────────────┼───────────────────────────────┐  │
│  │                                      │                               │  │
│  │  ┌───────────────────────────────────▼────────────────────────────┐  │  │
│  │  │                   S3 Bucket (Models)                            │  │  │
│  │  │                 retailpred-ml-models                             │  │  │
│  │  │                                                                 │  │  │
│  │  │  Contents:                                                      │  │  │
│  │  │  - total_sales_lightgbm_model.pkl                               │  │  │
│  │  │  - total_sales_patchtst_model.pth                               │  │  │
│  │  │  - ... (77 model files)                                         │  │  │
│  │  │                                                                 │  │  │
│  │  │  Lifecycle Policy:                                              │  │  │
│  │  │  - Transition to IA after 30 days                               │  │  │
│  │  │  - Delete after 90 days (for old versions)                      │  │  │
│  │  └────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                      │  │
│  │  ┌────────────────────────────────────────────────────────────────┐  │  │
│  │  │                      CloudWatch Logs                            │  │  │
│  │  │                                                                 │  │  │
│  │  │  Log Groups:                                                    │  │  │
│  │  │  - /aws/ec2/retailpred-api/application                          │  │  │
│  │  │  - /aws/ec2/retailpred-api/nginx                                │  │  │
│  │  │                                                                 │  │  │
│  │  │  Metric Filters:                                                │  │  │
│  │  │  - ERROR count → CloudWatch Alarm                               │  │  │
│  │  │  - 5xx responses → CloudWatch Alarm                             │  │  │
│  │  └────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                      │  │
│  │  ┌────────────────────────────────────────────────────────────────┐  │  │
│  │  │                      CloudWatch Metrics                         │  │  │
│  │  │                                                                 │  │  │
│  │  │  Custom Metrics:                                                │  │  │
│  │  │  - PredictionLatency (ms)                                       │  │  │
│  │  │  - ModelLoadTime (ms)                                           │  │  │
│  │  │  - ActiveUsers (count)                                          │  │  │
│  │  │  - DatabaseSize (GB)                                            │  │  │
│  │  └────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                      │  │
│  │  ┌────────────────────────────────────────────────────────────────┐  │  │
│  │  │                    Route53 (DNS)                                │  │  │
│  │  │                                                                 │  │  │
│  │  │  Records:                                                       │  │  │
│  │  │  - retailpred.com → A → ALB                                     │  │  │
│  │  │  - www.retailpred.com → CNAME → retailpred.com                  │  │  │
│  │  │  - api.retailpred.com → A → ALB (optional)                      │  │  │
│  │  └────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                      │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Side-by-Side Comparison

```
┌──────────────────────┬─────────────────────────┬──────────────────────────────┐
│    Component         │      Local (Dev)        │         AWS (Prod)           │
├──────────────────────┼─────────────────────────┼──────────────────────────────┤
│ Frontend Hosting     │ npm run dev            │ S3 + CloudFront CDN          │
│                      │ (localhost:5173)       │ (global edge locations)       │
├──────────────────────┼─────────────────────────┼──────────────────────────────┤
│ Backend Server       │ uvicorn main:app       │ EC2 (t3.medium) + Nginx       │
│                      │ (localhost:8000)       │ + Gunicorn (4 workers)        │
├──────────────────────┼─────────────────────────┼──────────────────────────────┤
│ Database             │ ./data/retailpred.db   │ EBS Volume (100 GB)           │
│                      │ (local file)           │ mounted at /mnt/ebs           │
├──────────────────────┼─────────────────────────┼──────────────────────────────┤
│ ML Models Storage    │ ./backend/ml/models/   │ S3 Bucket + Local Cache       │
│                      │ (local directory)      │ (/mnt/ebs/models/)            │
├──────────────────────┼─────────────────────────┼──────────────────────────────┤
│ Logging              │ stdout/stderr          │ CloudWatch Logs               │
│                      │ (terminal output)      │ (centralized log management)  │
├──────────────────────┼─────────────────────────┼──────────────────────────────┤
│ Monitoring           │ Manual console checks  │ CloudWatch Metrics + Alarms  │
├──────────────────────┼─────────────────────────┼──────────────────────────────┤
│ SSL/TLS              │ N/A (HTTP only)        │ AWS Certificate Manager       │
│                      │                        │ (free SSL certs)              │
├──────────────────────┼─────────────────────────┼──────────────────────────────┤
│ DNS                  │ N/A (localhost)        │ Route53                       │
│                      │                        │ (retailpred.com)              │
├──────────────────────┼─────────────────────────┼──────────────────────────────┤
│ Auto-scaling         │ N/A (single instance)  │ Auto Scaling Group            │
│                      │                        │ (2-10 instances)              │
├──────────────────────┼─────────────────────────┼──────────────────────────────┤
│ Load Balancing       │ N/A                    │ Application Load Balancer     │
│                      │                        │ (health checks, failover)     │
└──────────────────────┴─────────────────────────┴──────────────────────────────┘
```

---

## Component Mapping

### 1. Frontend (React + TypeScript)

#### Local Development
```bash
cd frontend
npm run dev  # Runs on localhost:5173
```

**Characteristics:**
- Hot Module Replacement (HMR)
- Source maps for debugging
- Development-only optimizations
- No minification
- Inline CSS for Tailwind

#### AWS Production
```bash
cd frontend
npm run build  # Creates /dist/ directory
aws s3 sync dist/ s3://retailpred-frontend-static --delete
```

**AWS Components:**
- **S3 Bucket**: `retailpred-frontend-static`
  - Stores static files (HTML, JS, CSS, images)
  - Versioning enabled
  - Lifecycle policy for old versions

- **CloudFront CDN**: Distributes content globally
  - Cache behavior: Cache all files for 24 hours
  - Invalidate cache on deployment
  - Edge locations: 400+ globally

- **Route53**: Maps domain to CloudFront
  - `retailpred.com` → CloudFront distribution
  - `www.retailpred.com` → Redirect to root domain

**Changes Required:**
```typescript
// frontend/src/api/client.ts
// Update API base URL for production
const API_BASE_URL = import.meta.env.VITE_API_URL || 'https://api.retailpred.com';
```

```bash
# .env.production
VITE_API_URL=https://api.retailpred.com
```

**Deployment Process:**
1. Build: `npm run build` → Creates optimized `/dist/`
2. Upload: `aws s3 sync dist/ s3://retailpred-frontend-static`
3. Invalidate: `aws cloudfront create-invalidation --distribution-id XXX --paths "/*"`
4. Verify: Access https://retailpred.com

---

### 2. Backend (FastAPI)

#### Local Development
```bash
cd backend
python main.py  # Runs on localhost:8000
```

**Characteristics:**
- Single Uvicorn worker
- Auto-reload on code changes
- DEBUG mode enabled
- CORS allows all origins
- Logs to console

#### AWS Production

**EC2 Instance Configuration:**
```bash
# Instance Type: t3.medium
# - 2 vCPUs
# - 4 GB RAM
# - Cost: ~$30/month

# AMI: Ubuntu 22.04 LTS
# EBS Volume: 100 GB (gp3)
#   - /dev/sda1: 20 GB (OS)
#   - /dev/sdb: 80 GB (data, mounted at /mnt/ebs)
```

**Nginx Configuration:**
```nginx
# /etc/nginx/sites-available/retailpred
server {
    listen 80;
    server_name retailpred.com www.retailpred.com;

    location /api {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support (for future features)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    location /docs {
        proxy_pass http://127.0.0.1:8000;
    }
}
```

**Gunicorn Systemd Service:**
```ini
# /etc/systemd/system/retailpred-api.service
[Unit]
Description=RetailPRED FastAPI Application
After=network.target

[Service]
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/app/backend
Environment="PATH=/home/ubuntu/app/venv/bin"
Environment="PYTHONPATH=/home/ubuntu/app/backend"
ExecStart=/home/ubuntu/app/venv/bin/gunicorn \
    -w 4 \
    -k uvicorn.workers.UvicornWorker \
    -b 127.0.0.1:8000 \
    -access-logfile /var/log/retailpred/access.log \
    -error-logfile /var/log/retailpred/error.log \
    main:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**SSL/TLS with Let's Encrypt:**
```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Generate certificate
sudo certbot --nginx -d retailpred.com -d www.retailpred.com

# Auto-renewal (cron)
0 0 * * * certbot renew --quiet
```

**Application Load Balancer (ALB):**
- **Target Group**: EC2 instances on port 80
- **Health Check**: `GET /health` every 30 seconds
- **Stickiness**: Enabled (for user sessions)
- **Security Group**: Allows HTTP/HTTPS from internet

**Auto Scaling Group:**
- **Min instances**: 2
- **Max instances**: 10
- **Scaling policy**: CPU > 70% for 5 minutes → scale up
- **Scaling policy**: CPU < 30% for 5 minutes → scale down

**Changes Required:**

1. **Update CORS Settings:**
```python
# backend/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://retailpred.com",
        "https://www.retailpred.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

2. **Add CloudWatch Logging:**
```python
# backend/main.py
import watchtower
import logging

# CloudWatch handler
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        watchtower.CloudWatchLogHandler(
            log_group_name="/aws/ec2/retailpred-api/application",
            stream_name="production"
        )
    ]
)
```

3. **Health Check Endpoint:**
```python
# backend/api/routes.py
@router.get("/health")
async def health_check():
    """Health check for ALB"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "database": db.is_connected(),
        "models_loaded": len(ml_integrator.models)
    }
```

---

### 3. Database (SQLite)

#### Local Development
```bash
# Location: ./data/retailpred.db
# Connection string: sqlite:///./data/retailpred.db
```

**Characteristics:**
- Single file
- No concurrent write handling needed
- Direct file I/O
- Backed up manually

#### AWS Production

**EBS Volume Setup:**
```bash
# Create EBS volume (80 GB, gp3)
aws ec2 create-volume \
    --volume-type gp3 \
    --size 80 \
    --availability-zone us-east-1a

# Attach to EC2 instance
aws ec2 attach-volume \
    --volume-id vol-1234567890 \
    --instance-id i-1234567890 \
    --device /dev/sdb

# On EC2 instance
sudo mkfs -t xfs /dev/sdb
sudo mkdir /mnt/ebs
sudo mount /dev/sdb /mnt/ebs
sudo chown ubuntu:ubuntu /mnt/ebs

# Add to /etc/fstab for auto-mount on reboot
/dev/sdb /mnt/ebs xfs defaults,nofail 0 2
```

**Directory Structure:**
```
/mnt/ebs/
├── data/
│   └── retailpred.db
├── models/
│   ├── total_sales_lightgbm_model.pkl
│   ├── total_sales_patchtst_model.pth
│   └── ...
└── backups/
    ├── retailpred.db.2025-01-01
    ├── retailpred.db.2025-01-02
    └── ...
```

**Automated Backups:**
```bash
# /home/ubuntu/scripts/backup-db.sh
#!/bin/bash
DATE=$(date +%Y-%m-%d)
DB_PATH="/mnt/ebs/data/retailpred.db"
BACKUP_PATH="/mnt/ebs/backups/retailpred.db.$DATE"

# Backup to local
cp $DB_PATH $BACKUP_PATH

# Backup to S3 (cross-region redundancy)
aws s3 cp $DB_PATH s3://retailpred-db-backups/$DATE/retailpred.db

# Delete backups older than 30 days
find /mnt/ebs/backups -name "retailpred.db.*" -mtime +30 -delete
```

**Systemd Timer for Daily Backups:**
```ini
# /etc/systemd/system/retailpred-backup.service
[Unit]
Description=RetailPRED Database Backup

[Service]
Type=oneshot
User=ubuntu
ExecStart=/home/ubuntu/scripts/backup-db.sh
```

```ini
# /etc/systemd/system/retailpred-backup.timer
[Unit]
Description=Daily RetailPRED Backup
Requires=retailpred-backup.service

[Timer]
OnCalendar=daily
Persistent=true

[Install]
WantedBy=timers.target
```

**Why Keep SQLite on AWS?**

**Advantages:**
- Zero operational overhead (no RDS management)
- Fast local reads (no network latency)
- Simple backup (file copy)
- Low cost ($8/month for 80 GB EBS)

**When to Migrate to RDS PostgreSQL:**
- Multiple concurrent writers needed
- > 10 GB database size
- Need for advanced features (JSONB, full-text search)
- Compliance requirements (multi-AZ, read replicas)

**Migration Path to RDS:**
```python
# Change connection string only!
# From:
DATABASE_URL = "sqlite:////mnt/ebs/data/retailpred.db"

# To:
DATABASE_URL = "postgresql://user:pass@retailpred-db.xxxx.us-east-1.rds.amazonaws.com:5432/retailpred"

# No code changes needed (SQLAlchemy handles the rest)
```

---

### 4. ML Models (.pkl files)

#### Local Development
```python
# backend/ml/multi_resolution_inference.py
model_path = "./training_outputs/total_sales/lightgbm_model.pkl"
model = joblib.load(model_path)
```

**Characteristics:**
- Models stored locally
- Loaded on startup
- No versioning
- No caching strategy

#### AWS Production

**S3 Bucket for Models:**
```
s3://retailpred-ml-models/
├── total_sales/
│   ├── lightgbm_model.pkl
│   ├── patchtst_model.pth
│   └── ...
├── general_merchandise/
│   ├── lightgbm_model.pkl
│   └── ...
└── archived/
    └── v1.0/
        └── old_model.pkl
```

**Model Loading with S3 + Local Cache:**
```python
# backend/ml/model_loader.py
import boto3
import joblib
from pathlib import Path

s3_client = boto3.client('s3')
LOCAL_CACHE_DIR = Path("/mnt/ebs/models")
LOCAL_CACHE_DIR.mkdir(exist_ok=True)

def load_model(model_name: str, category: str):
    """Load model with S3 + local cache strategy"""

    local_path = LOCAL_CACHE_DIR / category / f"{model_name}.pkl"

    # Check local cache first
    if local_path.exists():
        return joblib.load(local_path)

    # Download from S3 if not in cache
    s3_key = f"{category}/{model_name}.pkl"
    s3_client.download_file(
        "retailpred-ml-models",
        s3_key,
        str(local_path)
    )

    # Load from local copy
    return joblib.load(local_path)
```

**Benefits of S3 + Cache:**
1. **Fast first load**: Download once, cache locally
2. **Versioning**: S3 versioning for rollback
3. **Lifecycle policies**: Auto-archive old models
4. **Cross-instance sharing**: All EC2 instances access same models

**Lifecycle Policy for S3 Bucket:**
```json
{
  "Rules": [
    {
      "Id": "ArchiveOldModels",
      "Status": "Enabled",
      "Transitions": [
        {
          "Days": 30,
          "StorageClass": "STANDARD_IA"
        }
      ],
      "NoncurrentVersionTransitions": [
        {
          "NoncurrentDays": 7,
          "StorageClass": "GLACIER"
        }
      ]
    }
  ]
}
```

---

### 5. Logging and Monitoring

#### Local Development
```python
# backend/main.py
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

**Characteristics:**
- Logs to console
- No persistence
- Manual searching
- No alerting

#### AWS Production

**CloudWatch Logs Integration:**
```python
# backend/main.py
import watchtower
import logging

# Configure logging with CloudWatch handler
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Still log to console
        watchtower.CloudWatchLogHandler(
            log_group_name="/aws/ec2/retailpred-api/application",
            stream_name="production",
            boto3_client=boto3.client('logs', region_name='us-east-1')
        )
    ]
)
```

**CloudWatch Metrics (Custom):**
```python
# backend/api/routes.py
import boto3
cloudwatch = boto3.client('cloudwatch')

@router.post("/api/predict")
async def create_prediction(request: PredictionRequest):
    start_time = time.time()

    # ... generate prediction ...

    # Record latency metric
    latency_ms = (time.time() - start_time) * 1000
    cloudwatch.put_metric_data(
        Namespace='RetailPRED',
        MetricData=[{
            'MetricName': 'PredictionLatency',
            'Value': latency_ms,
            'Unit': 'Milliseconds'
        }]
    )

    return prediction
```

**CloudWatch Alarms:**
```bash
# Alarm: High Error Rate
aws cloudwatch put-metric-alarm \
    --alarm-name retailpred-high-errors \
    --alarm-description "Alert when error rate > 5%" \
    --metric-name Errors \
    --namespace AWS/ApplicationELB \
    --statistic Average \
    --period 300 \
    --threshold 5 \
    --comparison-operator GreaterThanThreshold

# Alarm: High CPU Usage
aws cloudwatch put-metric-alarm \
    --alarm-name retailpred-high-cpu \
    --alarm-description "Alert when CPU > 80%" \
    --metric-name CPUUtilization \
    --namespace AWS/EC2 \
    --statistic Average \
    --period 300 \
    --threshold 80 \
    --comparison-operator GreaterThanThreshold
```

**Log Insights Queries:**
```sql
-- Find all prediction errors
fields @timestamp, @message
| filter @type = "ERROR"
| filter @message like /prediction/
| sort @timestamp desc

-- Calculate average prediction latency
fields @timestamp, prediction_latency
| filter @type = "INFO"
| filter @message like /PredictionLatency/
| stats avg(prediction_latency) as avg_latency

-- Top 10 slowest predictions
fields prediction_id, latency
| sort latency desc
| limit 10
```

---

## Data Flow Diagrams

### Request Flow: User Makes Prediction

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PRODUCTION REQUEST FLOW                              │
└─────────────────────────────────────────────────────────────────────────────┘

User Browser
    │
    │ 1. GET https://retailpred.com/dashboard/predictions
    ▼
CloudFront CDN (Edge Location in NYC)
    │
    │ 2. Cache HIT → Return index.html + assets
    │    (or Cache MISS → Fetch from S3)
    ▼
S3 Bucket: retailpred-frontend-static
    │
    │ 3. Return React application bundle
    ▼
User Browser (React App Loaded)
    │
    │ 4. User selects category = "total_sales", weeks_ahead = 52
    │    Clicks "Generate Forecast"
    ▼
User Browser (API Call)
    │
    │ 5. POST https://retailpred.com/api/predict
    ▼
Route53 DNS
    │
    │ 6. Resolve retailpred.com → ALB DNS name
    ▼
Application Load Balancer (us-east-1)
    │
    │ 7. Route to healthy target (EC2 instance 1)
    ▼
Nginx Reverse Proxy (EC2 Instance 1)
    │
    │ 8. Proxy pass to http://127.0.0.1:8000/api/predict
    ▼
Gunicorn Worker 1 (FastAPI)
    │
    │ 9. Receive request, validate with Pydantic
    ▼
Prediction Service
    │
    │ 10. Check if model loaded in memory
    │     - If not: Load from /mnt/ebs/models/
    │     - If not cached: Download from S3 → Cache → Load
    ▼
ML Model (LightGBM)
    │
    │ 11. Run inference with 242 features
    │     - Temporal features updated for target dates
    │     - External factors from database
    ▼
Prediction Result (52 weeks of forecasts)
    │
    │ 12. Save to database: INSERT INTO prediction_log
    ▼
SQLite Database (/mnt/ebs/data/retailpred.db)
    │
    │ 13. Return prediction results + SHAP values
    ▼
FastAPI Response
    │
    │ 14. Gunicorn → Nginx → ALB → CloudFront → User
    ▼
User Browser (Display Results)
    │
    │ 15. Render forecast chart with Recharts
    │     Display SHAP explanations
    │     Show confidence intervals
```

### Model Loading Flow (First Request)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       MODEL LOADING FLOW (Cold Start)                        │
└─────────────────────────────────────────────────────────────────────────────┘

FastAPI Startup
    │
    │ 1. Check /mnt/ebs/models/ directory
    ▼
Local Cache Check
    │
    │ 2. Model not found locally
    ▼
S3 Download Request
    │
    │ 3. GET s3://retailpred-ml-models/total_sales/lightgbm_model.pkl
    ▼
S3 Bucket (us-east-1)
    │
    │ 4. Return model file (~5 MB)
    ▼
EC2 Instance
    │
    │ 5. Save to /mnt/ebs/models/total_sales/lightgbm_model.pkl
    ▼
Joblib Load
    │
    │ 6. Load model into memory
    ▼
Model Ready
    │
    │ 7. Cache in memory (Python dict)
    ▼
Prediction Proceeds
    │
    │ 8. Model already loaded → No S3 call needed
    ▼
Fast Response (< 100ms)
```

---

## Cost Estimates

### Monthly Cost Breakdown

| Service | Specification | Monthly Cost | Annual Cost |
|---------|---------------|--------------|-------------|
| **EC2 Instances** | 2 × t3.medium (Linux) | $60.00 | $720.00 |
| **EBS Storage** | 100 GB gp3 volumes | $8.00 | $96.00 |
| **S3 Storage** | 20 GB frontend + models | $0.46 | $5.52 |
| **CloudFront** | 100 GB data transfer | $8.50 | $102.00 |
| **Route53** | Hosted zone + queries | $0.60 | $7.20 |
| **Load Balancer** | 1 ALB + 2 targets | $18.25 | $219.00 |
| **CloudWatch Logs** | 5 GB logs ingestion | $2.50 | $30.00 |
| **Data Transfer** | 500 GB out to internet | $45.00 | $540.00 |
| **TOTAL** | | **$143.31** | **$1,719.72** |

### Cost Optimization Strategies

1. **Use Reserved Instances** (t3.medium)
   - 1-year commitment: $38/month (37% savings)
   - 3-year commitment: $30/month (50% savings)

2. **Use Spot Instances** for worker nodes
   - t3.medium spot: ~70% discount
   - Risk: Interruption (not recommended for primary instances)

3. **Optimize CloudFront**
   - Use Origin Access Identity (restrict S3 access)
   - Enable compression (reduce data transfer by ~50%)
   - Set appropriate cache headers

4. **Lifecycle Policies**
   - Transition old logs to Glacier after 30 days
   - Delete old models after 90 days
   - Archive old backups to S3 IA

5. **Right-Sizing**
   - Start with t3.medium
   - Monitor CPU/memory usage
   - Scale down if < 20% utilization

### Free Tier Eligibility (First 12 Months)

| Service | Free Tier Limit | Monthly Savings |
|---------|----------------|-----------------|
| EC2 | 750 hours/month (t2.micro only) | $0 |
| S3 | 5 GB storage, 20K requests | $0.23 |
| CloudFront | 1 TB data transfer | $8.50 |
| CloudWatch | 5 GB logs ingestion | $2.50 |
| **Total Savings** | | **$11.23/month** |

---

## Security Best Practices

### 1. Network Security

**Security Group Rules:**
```json
{
  "InboundRules": [
    {
      "Type": "HTTP",
      "Protocol": "TCP",
      "PortRange": "80",
      "Source": "0.0.0.0/0"
    },
    {
      "Type": "HTTPS",
      "Protocol": "TCP",
      "PortRange": "443",
      "Source": "0.0.0.0/0"
    },
    {
      "Type": "SSH",
      "Protocol": "TCP",
      "PortRange": "22",
      "Source": "YOUR_IP/32"
    }
  ],
  "OutboundRules": [
    {
      "Type": "All Traffic",
      "Protocol": "All",
      "PortRange": "All",
      "Destination": "0.0.0.0/0"
    }
  ]
}
```

**Best Practices:**
- ✅ Use security groups, not security group IDs
- ✅ Restrict SSH to your IP only
- ✅ Use AWS Systems Manager Session Manager (no SSH needed)
- ✅ Enable VPC Flow Logs for monitoring
- ✅ Use Network ACLs for additional layer of security

### 2. Data Security

**S3 Bucket Policies:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "DenyUnencryptedObjectUploads",
      "Effect": "Deny",
      "Principal": "*",
      "Action": "s3:PutObject",
      "Resource": "arn:aws:s3:::retailpred-ml-models/*",
      "Condition": {
        "StringNotEquals": {
          "s3:x-amz-server-side-encryption": "AES256"
        }
      }
    },
    {
      "Sid": "DenyHTTP",
      "Effect": "Deny",
      "Principal": "*",
      "Action": "s3:*",
      "Resource": "arn:aws:s3:::retailpred-ml-models/*",
      "Condition": {
        "Bool": {
          "aws:SecureTransport": "false"
        }
      }
    }
  ]
}
```

**Database Security:**
- ✅ Encrypt EBS volumes (AWS-managed keys)
- ✅ Enable S3 server-side encryption
- ✅ Rotate database credentials weekly
- ✅ Use IAM roles instead of access keys
- ✅ Enable S3 versioning (recover from accidental deletes)

### 3. Application Security

**Environment Variables (Never commit to git):**
```bash
# .env.production (on EC2 instance)
DATABASE_URL=sqlite:////mnt/ebs/data/retailpred.db
AWS_REGION=us-east-1
S3_MODELS_BUCKET=retailpred-ml-models
CLOUDWATCH_LOG_GROUP=/aws/ec2/retailpred-api/application

# API Keys (for external services)
FRED_API_KEY=your_secret_key_here
```

**Secrets Management (Better):**
```bash
# Use AWS Systems Manager Parameter Store
aws ssm put-parameter \
    --name "/retailprod/fred-api-key" \
    --value "your_secret_key" \
    --type "SecureString"

# Retrieve in code
import boto3
ssm = boto3.client('ssm')
fred_key = ssm.get_parameter(
    Name='/retailprod/fred-api-key',
    WithDecryption=True
)['Parameter']['Value']
```

**API Security:**
- ✅ Implement rate limiting (100 req/min per IP)
- ✅ Add authentication (JWT or API keys)
- ✅ Validate all inputs with Pydantic
- ✅ Sanitize database queries (SQLAlchemy handles this)
- ✅ Add CORS whitelist
- ✅ Use HTTPS only (redirect HTTP → HTTPS)

**Dependencies Security:**
```bash
# Scan for vulnerabilities
cd frontend
npm audit
cd ../backend
pip install safety
safety check

# Auto-update dependencies
npm update
pip-compile requirements.txt --upgrade
```

### 4. Access Control

**IAM Roles (Least Privilege):**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:us-east-1:*:log-group:/aws/ec2/retailpred-api/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject"
      ],
      "Resource": [
        "arn:aws:s3:::retailpred-ml-models/*",
        "arn:aws:s3:::retailpred-db-backups/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "cloudwatch:PutMetricData"
      ],
      "Resource": "*"
    }
  ]
}
```

### 5. Monitoring and Alerting

**CloudWatch Alarms:**
- CPU > 80% for 5 minutes → Scale up
- Error rate > 5% for 5 minutes → Alert via SNS
- Disk space < 20% → Alert via email
- Prediction latency > 2 seconds → Alert via Slack

**Logging Best Practices:**
- ✅ Never log sensitive data (API keys, passwords)
- ✅ Use structured logging (JSON format)
- ✅ Include request ID for tracing
- ✅ Log at appropriate levels (DEBUG, INFO, WARNING, ERROR)
- ✅ Set retention policies (30 days for application logs)

---

## Scaling Considerations

### Vertical Scaling (Scale Up)

**Current Instance: t3.medium**
- 2 vCPUs
- 4 GB RAM
- Handles: ~100 predictions/minute

**Upgrade Path:**

| Instance | vCPUs | RAM | Predictions/min | Cost/month |
|----------|-------|-----|-----------------|------------|
| t3.medium | 2 | 4 GB | 100 | $30 |
| t3.large | 2 | 8 GB | 200 | $60 |
| t3.xlarge | 4 | 16 GB | 500 | $120 |
| m5.xlarge | 4 | 16 GB | 800 | $192 |

**When to Scale Up:**
- CPU > 70% sustained
- Memory > 80% (OOM errors)
- Model loading taking too long

**When to Scale Out:**
- Spiky traffic (not sustained)
- Need > 1000 predictions/minute
- Want high availability (multi-AZ)

### Horizontal Scaling (Scale Out)

**Auto Scaling Group Configuration:**
```json
{
  "MinSize": 2,
  "MaxSize": 10,
  "DesiredCapacity": 2,
  "TargetTrackingPolicies": [
    {
      "TargetValue": 70.0,
      "PredefinedMetricSpecification": {
        "PredefinedMetricType": "ASGAverageCPUUtilization"
      }
    }
  ]
}
```

**Scaling Events:**
1. Traffic increases → CPU > 70% → Launch new instance
2. New instance boots up → Downloads models from S3
3. Health check passes → Instance added to ALB
4. Traffic distributed across 3 instances

**Challenges with Horizontal Scaling:**
- ⚠️ Model loading on each instance (cold start delay)
- ⚠️ Database locking (SQLite has write conflicts)
- ⚠️ Session state (need sticky sessions or Redis)

**Solution: Use Read Replicas + Write Master**
```
┌──────────────────┐
│  EC2 Instance 1  │ ──┐
│  (Write Master)  │   │
└──────────────────┘   │
                       │
                       ├─► EBS (Read-Only Mount)
                       │
┌──────────────────┐   │
│  EC2 Instance 2  │ ──┘
│  (Read Replica)  │
└──────────────────┘
```

### Database Scaling

**When SQLite Becomes a Bottleneck:**

| Metric | SQLite Limit | Action |
|--------|--------------|--------|
| Database size | < 10 GB | Stay with SQLite |
| Concurrent writers | 1 | Stay with SQLite |
| Read queries | < 100/sec | Stay with SQLite |
| Write latency | < 10ms | Stay with SQLite |
| **Any exceeded** | | **Migrate to RDS** |

**Migration to RDS PostgreSQL:**
```bash
# 1. Export SQLite to SQL
sqlite3 retailpred.db .dump > retailpred.sql

# 2. Create RDS instance
aws rds create-db-instance \
    --db-instance-identifier retailpred-db \
    --db-instance-class db.t3.micro \
    --engine postgres \
    --master-username admin \
    --allocated-storage 20

# 3. Import to RDS
psql -h retailpred-db.xxxx.us-east-1.rds.amazonaws.com \
    -U admin \
    -d retailpred \
    -f retailpred.sql

# 4. Update application connection string
```

**Benefits of RDS:**
- Multi-AZ deployment (automatic failover)
- Read replicas (scale reads)
- Automated backups (7-35 days retention)
- Point-in-time recovery (restore to any second)
- Enhanced monitoring (Performance Insights)
- **Cost**: db.t3.micro = $15/month

### Caching Strategy

**Use Elasticache (Redis) for:**
1. **Session storage** (user authentication state)
2. **API response caching** (expensive predictions)
3. **Rate limiting** (request counts per IP)

**Example: Cache Prediction Results**
```python
# backend/api/routes.py
import redis

redis_client = redis.Redis(
    host='retailpred-cache.xxxxx.use1.cache.amazonaws.com',
    port=6379,
    db=0
)

@router.post("/api/predict")
async def create_prediction(request: PredictionRequest):
    # Generate cache key
    cache_key = f"prediction:{request.category}:{request.weeks_ahead}:{hash(str(request.features))}"

    # Check cache
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    # Generate prediction
    prediction = await generate_prediction(request)

    # Cache for 1 hour
    redis_client.setex(cache_key, 3600, json.dumps(prediction))

    return prediction
```

**Cost:**
- cache.t3.micro = $12/month
- cache.t3.small = $18/month

### CDN Optimization

**CloudFront Cache Behaviors:**
```
Path Pattern: /assets/*
Cache Policy: CachingOptimized
TTL: 86400 seconds (24 hours)

Path Pattern: /api/*
Cache Policy: Disabled (no caching)

Path Pattern: index.html
Cache Policy: CachingDisabled
TTL: 0 (always fetch latest)
```

**Invalidation Strategy:**
```bash
# On deployment, invalidate all HTML/JS
aws cloudfront create-invalidation \
    --distribution-id XXXXXX \
    --paths "/index.html" "/assets/*.js"

# Don't invalidate images (let them expire naturally)
```

---

## Migration Checklist

### Phase 1: Infrastructure Setup (Week 1)

- [ ] Create AWS account (if not exists)
- [ ] Set up billing alerts
- [ ] Create IAM user for deployment
- [ ] Configure AWS CLI locally
- [ ] Create S3 buckets (frontend, models, backups)
- [ ] Create EC2 key pair
- [ ] Launch EC2 instance (t3.medium)
- [ ] Configure security groups
- [ ] Mount EBS volume
- [ ] Install dependencies (Python, Nginx, Gunicorn)

### Phase 2: Application Deployment (Week 2)

- [ ] Clone repository to EC2
- [ ] Install Python dependencies (venv)
- [ ] Configure Nginx reverse proxy
- [ ] Set up Gunicorn systemd service
- [ ] Configure environment variables
- [ ] Download ML models from S3
- [ ] Initialize database with seed data
- [ ] Test API endpoints locally
- [ ] Set up SSL certificate (Let's Encrypt)

### Phase 3: Frontend Deployment (Week 2)

- [ ] Build React app (`npm run build`)
- [ ] Upload to S3 (`aws s3 sync`)
- [ ] Create CloudFront distribution
- [ ] Configure custom domain (Route53)
- [ ] Update API base URL in frontend
- [ ] Test frontend deployment
- [ ] Enable HTTPS only

### Phase 4: Load Balancing & Scaling (Week 3)

- [ ] Create AMI from EC2 instance
- [ ] Launch Auto Scaling Group (min=2, max=10)
- [ ] Create Application Load Balancer
- [ ] Configure health checks
- [ ] Set up scaling policies
- [ ] Configure Route53 for ALB
- [ ] Test failover (terminate one instance)

### Phase 5: Monitoring & Backups (Week 3)

- [ ] Configure CloudWatch Logs
- [ ] Set up CloudWatch Alarms
- [ ] Create SNS topics for alerts
- [ ] Configure database backup scripts
- [ ] Set up S3 lifecycle policies
- [ ] Test backup restoration
- [ ] Set up log aggregation

### Phase 6: Security Hardening (Week 4)

- [ ] Enable HTTPS only (no HTTP)
- [ ] Configure IAM roles (least privilege)
- [ ] Set up S3 bucket policies
- [ ] Enable EBS encryption
- [ ] Rotate database credentials
- [ ] Configure rate limiting
- [ ] Set up WAF (optional)
- [ ] Run security audit

### Phase 7: Performance Optimization (Week 4)

- [ ] Enable CloudFront compression
- [ ] Configure cache headers
- [ ] Set up Redis caching (optional)
- [ ] Optimize database queries
- [ ] Enable gzip compression in Nginx
- [ ] Run load tests (Locust)
- [ ] Tune Gunicorn workers
- [ ] Right-size EC2 instances

---

## Summary

This architecture provides a **production-ready AWS deployment** for RetailPRED with:

✅ **High Availability** (Multi-AZ, Auto Scaling)
✅ **Scalability** (Horizontal + vertical scaling)
✅ **Security** (HTTPS, IAM roles, encryption)
✅ **Monitoring** (CloudWatch logs + metrics + alarms)
✅ **Cost-Effective** ($143/month, or $96/month with RIs)
✅ **Disaster Recovery** (Automated backups, versioning)
✅ **Performance** (CloudFront CDN, model caching)
✅ **Maintainability** (Infrastructure as Code, automated deployments)

**Next Steps:**
1. Review [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for step-by-step instructions
2. Examine [infrastructure/terraform/](infrastructure/terraform/) for Infrastructure as Code
3. Run [scripts/deploy-aws.sh](scripts/deploy-aws.sh) for automated deployment
