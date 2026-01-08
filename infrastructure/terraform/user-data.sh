#!/bin/bash
# User Data Script for RetailPRED EC2 Instances
# This script runs on first boot of each EC2 instance

set -e  # Exit on error

# Log all output to /var/log/user-data.log
exec > >(tee /var/log/user-data.log)
exec 2>&1

echo "========================================="
echo "RetailPRED EC2 User Data Script"
echo "Started at: $(date)"
echo "========================================="

# =============================================================================
# SYSTEM UPDATE
# =============================================================================

echo "[1/10] Updating system packages..."
apt-get update
apt-get upgrade -y

# =============================================================================
# INSTALL DEPENDENCIES
# =============================================================================

echo "[2/10] Installing system dependencies..."
apt-get install -y \
    python3.10 \
    python3-pip \
    python3-venv \
    nginx \
    certbot \
    python3-certbot-nginx \
    git \
    curl \
    wget \
    htop \
    jq \
    awscli \
    xfsprogs

# =============================================================================
# MOUNT EBS VOLUME
# =============================================================================

echo "[3/10] Mounting EBS data volume..."
# Find EBS volume (usually /dev/xvdb or /dev/nvme1n1)
EBS_DEVICE=""
for device in /dev/xvdb /dev/nvme1n1 /dev/sdb; do
    if [ -e "$device" ]; then
        EBS_DEVICE="$device"
        break
    fi
done

if [ -z "$EBS_DEVICE" ]; then
    echo "ERROR: No EBS volume found!"
    exit 1
fi

echo "Found EBS device: $EBS_DEVICE"

# Check if volume is already formatted
if ! file -s $EBS_DEVICE | grep -q "XFS"; then
    echo "Formatting EBS volume with XFS..."
    mkfs -t xfs $EBS_DEVICE
fi

# Create mount point
mkdir -p /mnt/ebs

# Check if already mounted
if ! mountpoint -q /mnt/ebs; then
    echo "Mounting EBS volume..."
    mount $EBS_DEVICE /mnt/ebs

    # Add to fstab for auto-mount on reboot
    if ! grep -q "$EBS_DEVICE" /etc/fstab; then
        echo "$EBS_DEVICE /mnt/ebs xfs defaults,nofail 0 2" >> /etc/fstab
        echo "Added EBS volume to /etc/fstab"
    fi
else
    echo "EBS volume already mounted"
fi

# Create directory structure
mkdir -p /mnt/ebs/data
mkdir -p /mnt/ebs/models
mkdir -p /mnt/ebs/backups
mkdir -p /var/log/retailpred

# =============================================================================
# CLONE APPLICATION CODE
# =============================================================================

echo "[4/10] Cloning application repository..."
cd /var/www

if [ ! -d "retailpred" ]; then
    # In production, use your actual repository URL
    # git clone https://github.com/your-username/retailPRED.git retailpred
    # For now, create placeholder
    mkdir -p retailpred
    echo "NOTE: Clone your repository manually or update this script"
fi

# =============================================================================
# SETUP PYTHON VIRTUAL ENVIRONMENT
# =============================================================================

echo "[5/10] Setting up Python virtual environment..."
cd /var/www/retailpred

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate

# Install Python dependencies
if [ -f "requirements.txt" ]; then
    pip install --upgrade pip
    pip install -r requirements.txt
else
    # Install essential packages if requirements.txt not found
    pip install \
        fastapi \
        uvicorn[standard] \
        gunicorn \
        sqlalchemy \
        sqlite \
        pandas \
        numpy \
        scikit-learn \
        joblib \
        lightgbm \
        boto3 \
        watchtower \
        python-multipart
fi

# =============================================================================
# DOWNLOAD ML MODELS FROM S3
# =============================================================================

echo "[6/10] Downloading ML models from S3..."

# Get S3 bucket name from instance metadata or environment
S3_MODELS_BUCKET="${S3_MODELS_BUCKET:-retailprod-models}"

if aws s3 ls "s3://$S3_MODELS_BUCKET/" 2>&1 | grep -q "Unable to locate"; then
    echo "WARNING: S3 bucket $S3_MODELS_BUCKET not found or not accessible"
    echo "Models will need to be downloaded manually"
else
    aws s3 sync "s3://$S3_MODELS_BUCKET/" /mnt/ebs/models/ \
        --exclude "*.html" \
        --exclude "*.png" \
        --exclude "*.json" \
        --exclude "*.md"

    echo "Models downloaded successfully"
    ls -lh /mnt/ebs/models/
fi

# =============================================================================
# CREATE ENVIRONMENT FILE
# =============================================================================

echo "[7/10] Creating environment file..."
cat > /var/www/retailpred/.env.production << 'EOF'
# Application
APP_NAME=RetailPRED
APP_ENV=production
DEBUG=false

# Database
DATABASE_URL=sqlite:////mnt/ebs/data/retailpred.db

# AWS
AWS_REGION=us-east-1
S3_MODELS_BUCKET=S3_MODELS_BUCKET_PLACEHOLDER

# CloudWatch
CLOUDWATCH_LOG_GROUP=/aws/ec2/retailpred/application
EOF

# Replace S3 bucket placeholder
sed -i "s/S3_MODELS_BUCKET_PLACEHOLDER/$S3_MODELS_BUCKET/" /var/www/retailpred/.env.production

# =============================================================================
# CONFIGURE NGINX
# =============================================================================

echo "[8/10] Configuring Nginx..."
cat > /etc/nginx/sites-available/retailpred << 'EOF'
upstream retailpred_backend {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name _;

    client_max_body_size 100M;

    access_log /var/log/nginx/retailpred-access.log;
    error_log /var/log/nginx/retailpred-error.log;

    location /api {
        proxy_pass http://retailpred_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    location /docs {
        proxy_pass http://retailpred_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    location /health {
        proxy_pass http://retailpred_backend;
        access_log off;
    }
}
EOF

# Remove default site
rm -f /etc/nginx/sites-enabled/default

# Enable RetailPRED site
ln -sf /etc/nginx/sites-available/retailpred /etc/nginx/sites-enabled/

# Test Nginx configuration
nginx -t

# Restart Nginx
systemctl restart nginx

# =============================================================================
# CONFIGURE GUNICORN SERVICE
# =============================================================================

echo "[9/10] Configuring Gunicorn service..."
cat > /etc/systemd/system/retailpred-api.service << 'EOF'
[Unit]
Description=RetailPRED FastAPI Application
After=network.target

[Service]
Type=notify
User=ubuntu
Group=ubuntu
WorkingDirectory=/var/www/retailpred
Environment="PATH=/var/www/retailpred/venv/bin"
EnvironmentFile=/var/www/retailpred/.env.production
ExecStart=/var/www/retailpred/venv/bin/gunicorn \
    -w 4 \
    -k uvicorn.workers.UvicornWorker \
    -b 127.0.0.1:8000 \
    --access-logfile /var/log/retailpred/access.log \
    --error-logfile /var/log/retailpred/error.log \
    --log-level info \
    --timeout 120 \
    --worker-tmp-dir /dev/shm \
    main:app
Restart=always
RestartSec=10
KillMode=mixed
TimeoutStopSec=30
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
systemctl daemon-reload

# Enable service (but don't start yet - need to deploy code first)
systemctl enable retailpred-api

# =============================================================================
# CONFIGURE CLOUDWATCH LOGS
# =============================================================================

echo "[10/10] Configuring CloudWatch Logs..."

# Install CloudWatch agent (optional)
# wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
# dpkg -i -E ./amazon-cloudwatch-agent.deb

# Create log groups
aws logs create-log-group --log-group-name /aws/ec2/retailpred/application --region us-east-1 || true
aws logs create-log-group --log-group-name /aws/ec2/retailpred/nginx --region us-east-1 || true

# =============================================================================
# CREATE DATABASE BACKUP SCRIPT
# =============================================================================

cat > /home/ubuntu/scripts/backup-db.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y-%m-%d)
DB_PATH="/mnt/ebs/data/retailpred.db"
BACKUP_PATH="/mnt/ebs/backups/retailpred.db.$DATE"
S3_BACKUPS_BUCKET="${S3_BACKUPS_BUCKET:-retailprod-backups}"

# Backup to local
if [ -f "$DB_PATH" ]; then
    cp $DB_PATH $BACKUP_PATH

    # Backup to S3
    aws s3 cp $DB_PATH s3://${S3_BACKUPS_BUCKET}/$DATE/retailpred.db

    # Delete local backups older than 30 days
    find /mnt/ebs/backups -name "retailpred.db.*" -mtime +30 -delete

    echo "Backup completed: $DATE"
else
    echo "Database not found at $DB_PATH"
fi
EOF

chmod +x /home/ubuntu/scripts/backup-db.sh

# Create systemd timer for daily backups
cat > /etc/systemd/system/retailpred-backup.service << 'EOF'
[Unit]
Description=RetailPRED Database Backup
After=network.target

[Service]
Type=oneshot
User=ubuntu
ExecStart=/home/ubuntu/scripts/backup-db.sh
EOF

cat > /etc/systemd/system/retailpred-backup.timer << 'EOF'
[Unit]
Description=Daily RetailPRED Backup
Requires=retailpred-backup.service

[Timer]
OnCalendar=*-*-* 02:00:00
Persistent=true

[Install]
WantedBy=timers.target
EOF

systemctl daemon-reload
systemctl enable retailpred-backup.timer
systemctl start retailpred-backup.timer

# =============================================================================
# COMPLETE
# =============================================================================

echo "========================================="
echo "User Data Script Complete!"
echo "Completed at: $(date)"
echo "========================================="
echo ""
echo "Next Steps:"
echo "1. SSH into the instance"
echo "2. Clone your application code to /var/www/retailpred"
echo "3. Install Python dependencies"
echo "4. Initialize database"
echo "5. Start the service: sudo systemctl start retailpred-api"
echo ""
echo "Application will be available at:"
echo "  HTTP: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)"
echo ""
echo "Health check:"
echo "  curl http://localhost:8000/health"
echo ""
