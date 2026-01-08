# RetailPRED AWS Deployment Guide

## Table of Contents
- [Prerequisites](#prerequisites)
- [Phase 1: AWS Account Setup](#phase-1-aws-account-setup)
- [Phase 2: Infrastructure Deployment](#phase-2-infrastructure-deployment)
- [Phase 3: Application Deployment](#phase-3-application-deployment)
- [Phase 4: Frontend Deployment](#phase-4-frontend-deployment)
- [Phase 5: Load Balancing & Scaling](#phase-5-load-balancing--scaling)
- [Phase 6: Monitoring & Backups](#phase-6-monitoring--backups)
- [Phase 7: Security Hardening](#phase-7-security-hardening)
- [Rollback Procedures](#rollback-procedures)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Accounts & Tools

1. **AWS Account** (create at https://aws.amazon.com)
   - Valid credit card required
   - Free tier eligible for first 12 months

2. **Local Tools:**
   ```bash
   # Install AWS CLI v2
   # Mac:
   brew install awscli

   # Linux:
   sudo apt install awscli

   # Windows (Chocolatey):
   choco install awscli

   # Verify installation
   aws --version
   # Output: aws-cli/2.x.x Python/3.x.x
   ```

3. **Terraform** (for Infrastructure as Code):
   ```bash
   # Mac:
   brew install terraform

   # Linux:
   wget https://releases.hashicorp.com/terraform/1.5.0/terraform_1.5.0_linux_amd64.zip
   unzip terraform_1.5.0_linux_amd64.zip
   sudo mv terraform /usr/local/bin/

   # Verify
   terraform --version
   ```

4. **Domain Name** (optional but recommended):
   - Purchase from Route53 or other registrar
   - Example: `retailpred.com`

### Configure AWS Credentials

```bash
# Create IAM user in AWS Console
# - Go to IAM → Users → Add user
# - Username: "retailpred-deployer"
# - Access type: Programmatic access
# - Attach policy: AdministratorAccess (for deployment only)
# - Save Access Key ID and Secret Access Key

# Configure AWS CLI
aws configure

# Enter your credentials:
# AWS Access Key ID: AKIAIOSFODNN7EXAMPLE
# AWS Secret Access Key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
# Default region name: us-east-1
# Default output format: json

# Verify configuration
aws sts get-caller-identity

# Expected output:
# {
#   "UserId": "AIDAI...EXAMPLE",
#   "Account": "123456789012",
#   "Arn": "arn:aws:iam::123456789012:user/retailpred-deployer"
# }
```

### Set Up SSH Key Pair

```bash
# Generate SSH key for EC2 access
ssh-keygen -t rsa -b 4096 -f ~/.ssh/retailpred-deploy

# Import key into AWS
aws ec2 import-key-pair \
    --key-name retailpred-deploy \
    --public-key-material fileb://~/.ssh/retailpred-deploy.pub

# Verify
aws ec2 describe-key-pairs --key-names retailpred-deploy
```

---

## Phase 1: AWS Account Setup

### 1.1 Create Billing Alerts

Prevent unexpected charges:

```bash
# Create AWS Budgets alert
aws budgets create-budget \
    --account-id $(aws sts get-caller-identity --query Account --output text) \
    --budget '{
        "BudgetName": "retailpred-monthly-budget",
        "BudgetLimit": {
            "Amount": "200",
            "Unit": "USD"
        },
        "TimeUnit": "MONTHLY",
        "BudgetType": "COST",
        "CostFilters": {
            "AZ": ["us-east-1"]
        }
    }'
```

**Via Console (easier):**
1. Go to Billing → Budgets
2. Create budget: $200/month
3. Set alert at 80% ($160) and 100% ($200)
4. Notify via email

### 1.2 Create S3 Buckets

```bash
# Set your bucket name (must be globally unique)
BUCKET_PREFIX="retailprod-$(whoami)-$(date +%s)"

# Frontend static files
aws s3 mb s3://${BUCKET_PREFIX}-frontend --region us-east-1

# ML models storage
aws s3 mb s3://${BUCKET_PREFIX}-models --region us-east-1

# Database backups
aws s3 mb s3://${BUCKET_PREFIX}-backups --region us-east-1

# Enable versioning on models bucket
aws s3 put-bucket-versioning \
    --bucket ${BUCKET_PREFIX}-models \
    --versioning-configuration Status=Enabled

# Save bucket names to .env
echo "S3_FRONTEND_BUCKET=${BUCKET_PREFIX}-frontend" >> .env.aws
echo "S3_MODELS_BUCKET=${BUCKET_PREFIX}-models" >> .env.aws
echo "S3_BACKUPS_BUCKET=${BUCKET_PREFIX}-backups" >> .env.aws
```

### 1.3 Upload ML Models to S3

```bash
# Upload all models from training_outputs
S3_MODELS_BUCKET=$(grep S3_MODELS_BUCKET .env.aws | cut -d= -f2)

aws s3 sync training_outputs/ s3://${S3_MODELS_BUCKET}/ \
    --exclude "*.html" \
    --exclude "*.png" \
    --exclude "*.json"

# Verify upload
aws s3 ls s3://${S3_MODELS_BUCKET}/ --recursive --summarize
```

---

## Phase 2: Infrastructure Deployment

### Option A: Automated (Terraform) - RECOMMENDED

**See [`infrastructure/terraform/README.md`](infrastructure/terraform/README.md) for detailed instructions.**

```bash
cd infrastructure/terraform

# Initialize Terraform
terraform init

# Review the plan
terraform plan \
    -var="domain_name=retailpred.com" \
    -var="bucket_prefix=retailprod"

# Deploy infrastructure
terraform apply \
    -var="domain_name=retailpred.com" \
    -var="bucket_prefix=retailprod"

# Save outputs
terraform output -json > terraform-outputs.json
```

**What gets created:**
- VPC with public subnets
- EC2 instances (t3.medium)
- Application Load Balancer
- Security Groups
- IAM roles
- CloudWatch Log Groups
- Route53 hosted zone

### Option B: Manual (AWS Console) - FOR LEARNING

#### 2.1 Create VPC

```bash
# Create VPC
VPC_ID=$(aws ec2 create-vpc \
    --cidr-block 10.0.0.0/16 \
    --query 'Vpc.VpcId' \
    --output text)

# Enable DNS support
aws ec2 modify-vpc-attribute \
    --vpc-id $VPC_ID \
    --enable-dns-support '{"Value":true}'

aws ec2 modify-vpc-attribute \
    --vpc-id $VPC_ID \
    --enable-dns-hostnames '{"Value":true}'

# Create Internet Gateway
IGW_ID=$(aws ec2 create-internet-gateway \
    --query 'InternetGateway.InternetGatewayId' \
    --output text)

aws ec2 attach-internet-gateway \
    --vpc-id $VPC_ID \
    --internet-gateway-id $IGW_ID

# Create subnet
SUBNET_ID=$(aws ec2 create-subnet \
    --vpc-id $VPC_ID \
    --cidr-block 10.0.1.0/24 \
    --availability-zone us-east-1a \
    --query 'Subnet.SubnetId' \
    --output text)

# Create route table
RT_ID=$(aws ec2 create-route-table \
    --vpc-id $VPC_ID \
    --query 'RouteTable.RouteTableId' \
    --output text)

# Add route to internet
aws ec2 create-route \
    --route-table-id $RT_ID \
    --destination-cidr-block 0.0.0.0/0 \
    --gateway-id $IGW_ID

# Associate with subnet
aws ec2 associate-route-table \
    --subnet-id $SUBNET_ID \
    --route-table-id $RT_ID

echo "VPC_ID=$VPC_ID" >> .env.aws
echo "SUBNET_ID=$SUBNET_ID" >> .env.aws
```

#### 2.2 Create Security Group

```bash
VPC_ID=$(grep VPC_ID .env.aws | cut -d= -f2)

# Create security group
SG_ID=$(aws ec2 create-security-group \
    --group-name retailpred-sg \
    --description "Security group for RetailPRED" \
    --vpc-id $VPC_ID \
    --query 'GroupId' \
    --output text)

# Allow HTTP (from anywhere)
aws ec2 authorize-security-group-ingress \
    --group-id $SG_ID \
    --protocol tcp \
    --port 80 \
    --cidr 0.0.0.0/0

# Allow HTTPS (from anywhere)
aws ec2 authorize-security-group-ingress \
    --group-id $SG_ID \
    --protocol tcp \
    --port 443 \
    --cidr 0.0.0.0/0

# Allow SSH (from your IP only)
MY_IP=$(curl -s https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress \
    --group-id $SG_ID \
    --protocol tcp \
    --port 22 \
    --cidr ${MY_IP}/32

echo "SECURITY_GROUP_ID=$SG_ID" >> .env.aws
```

#### 2.3 Launch EC2 Instance

```bash
SUBNET_ID=$(grep SUBNET_ID .env.aws | cut -d= -f2)
SG_ID=$(grep SECURITY_GROUP_ID .env.aws | cut -d= -f2)

# Find latest Ubuntu 22.04 AMI
AMI_ID=$(aws ec2 describe-images \
    --owners 099720109477 \
    --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" "Name=state,Values=available" \
    --query "sort_by(Images, &CreationDate)[-1].ImageId" \
    --output text)

# Create EBS volume for data
VOLUME_ID=$(aws ec2 create-volume \
    --volume-type gp3 \
    --size 100 \
    --availability-zone us-east-1a \
    --query 'VolumeId' \
    --output text)

# Launch instance
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id $AMI_ID \
    --count 1 \
    --instance-type t3.medium \
    --key-name retailpred-deploy \
    --security-group-ids $SG_ID \
    --subnet-id $SUBNET_ID \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=RetailPRED-Backend}]' \
    --query 'Instances[0].InstanceId' \
    --output text)

# Wait for instance to be running
aws ec2 wait instance-running --instance-ids $INSTANCE_ID

# Attach data volume
aws ec2 attach-volume \
    --volume-id $VOLUME_ID \
    --instance-id $INSTANCE_ID \
    --device /dev/sdb

echo "INSTANCE_ID=$INSTANCE_ID" >> .env.aws
echo "VOLUME_ID=$VOLUME_ID" >> .env.aws
```

#### 2.4 Allocate Elastic IP (Optional)

```bash
# Allocate Elastic IP
ALLOCATION_ID=$(aws ec2 allocate-address \
    --domain vpc \
    --query 'AllocationId' \
    --output text)

# Associate with instance
aws ec2 associate-address \
    --allocation-id $ALLOCATION_ID \
    --instance-id $INSTANCE_ID

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "PUBLIC_IP=$PUBLIC_IP" >> .env.aws
```

---

## Phase 3: Application Deployment

### 3.1 Connect to EC2 Instance

```bash
# Get instance public IP
INSTANCE_ID=$(grep INSTANCE_ID .env.aws | cut -d= -f2)
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

# SSH into instance
ssh -i ~/.ssh/retailpred-deploy ubuntu@$PUBLIC_IP

# (Or use AWS Systems Manager Session Manager - no SSH needed)
aws ssm start-session --target $INSTANCE_ID
```

### 3.2 Initial Server Setup

```bash
# On EC2 instance

# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y \
    python3.10 \
    python3-pip \
    python3-venv \
    nginx \
    certbot \
    python3-certbot-nginx \
    git \
    htop \
    curl

# Create app directory
sudo mkdir -p /var/www/retailpred
sudo chown ubuntu:ubuntu /var/www/retailpred
cd /var/www/retailpred
```

### 3.3 Mount EBS Volume

```bash
# Find EBS volume
lsblk
# You should see /dev/xvdb (or /dev/nvme1n1)

# Format volume
sudo mkfs -t xfs /dev/xvdb

# Create mount point
sudo mkdir -p /mnt/ebs

# Mount volume
sudo mount /dev/xvdb /mnt/ebs

# Set ownership
sudo chown ubuntu:ubuntu /mnt/ebs

# Add to fstab for auto-mount on reboot
echo "/dev/xvdb /mnt/ebs xfs defaults,nofail 0 2" | sudo tee -a /etc/fstab

# Create directory structure
mkdir -p /mnt/ebs/data
mkdir -p /mnt/ebs/models
mkdir -p /mnt/ebs/backups
```

### 3.4 Clone Repository

```bash
# Clone application code
cd /var/www/retailpred
git clone https://github.com/oleeveeuh/retailPRED.git

# Or download from your private repo
# git clone git@github.com:your-username/retailPRED.git

cd retailPRED

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r backend/requirements.txt
pip install gunicorn uvicorn watchtower boto3

# Install additional production dependencies
pip install python-multipart  # For form uploads
```

### 3.5 Configure Environment

```bash
# Create environment file
cat > /var/www/retailpred/retailPRED/.env.production << 'EOF'
# Application
APP_NAME=RetailPRED
APP_ENV=production
DEBUG=false

# Database
DATABASE_URL=sqlite:////mnt/ebs/data/retailpred.db

# AWS
AWS_REGION=us-east-1
S3_MODELS_BUCKET=$(grep S3_MODELS_BUCKET /home/ubuntu/.env.aws | cut -d= -f2)

# CloudWatch
CLOUDWATCH_LOG_GROUP=/aws/ec2/retailpred-api/application

# API Keys (store in SSM Parameter Store instead!)
# FRED_API_KEY=your_key_here
EOF

# Source environment in bash profile
echo "source /var/www/retailpred/retailPRED/.env.production" >> ~/.bashrc
source ~/.bashrc
```

### 3.6 Initialize Database

```bash
cd /var/www/retailpred/retailPRED/backend

# Initialize database
python3 - << 'EOF'
import sys
sys.path.insert(0, '/var/www/retailpred/retailPRED/backend')

from db.database import RetailPREDDatabase
from pathlib import Path

db_path = Path("/mnt/ebs/data/retailpred.db")
db = RetailPREDDatabase(db_path=str(db_path))

# Initialize schema
db.initialize_schema()

# Load initial data (categories, etc.)
# db.load_initial_data()

print("Database initialized successfully!")
EOF
```

### 3.7 Download Models from S3

```bash
# Download all models to local cache
S3_MODELS_BUCKET=$(grep S3_MODELS_BUCKET .env.aws | cut -d= -f2)

aws s3 sync s3://${S3_MODELS_BUCKET}/ /mnt/ebs/models/ \
    --exclude "*.html" \
    --exclude "*.png"

# Verify download
ls -lh /mnt/ebs/models/
```

### 3.8 Configure Gunicorn Service

```bash
# Create systemd service file
sudo tee /etc/systemd/system/retailpred-api.service > /dev/null << 'EOF'
[Unit]
Description=RetailPRED FastAPI Application
After=network.target

[Service]
Type=notify
User=ubuntu
Group=ubuntu
WorkingDirectory=/var/www/retailpred/retailPRED/backend
Environment="PATH=/var/www/retailpred/retailPRED/venv/bin"
EnvironmentFile=/var/www/retailpred/retailPRED/.env.production
ExecStart=/var/www/retailpred/retailPRED/venv/bin/gunicorn \
    -w 4 \
    -k uvicorn.workers.UvicornWorker \
    -b 127.0.0.1:8000 \
    --access-logfile /var/log/retailpred/access.log \
    --error-logfile /var/log/retailpred/error.log \
    --log-level info \
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

# Create log directory
sudo mkdir -p /var/log/retailpred
sudo chown ubuntu:ubuntu /var/log/retailpred

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable retailpred-api
sudo systemctl start retailpred-api

# Check status
sudo systemctl status retailpred-api

# View logs
sudo journalctl -u retailpred-api -f
```

### 3.9 Configure Nginx

```bash
# Remove default site
sudo rm /etc/nginx/sites-enabled/default

# Create RetailPRED site config
sudo tee /etc/nginx/sites-available/retailpred > /dev/null << 'EOF'
upstream retailpred_backend {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name _;

    # Increase upload size for model files
    client_max_body_size 100M;

    # Logging
    access_log /var/log/nginx/retailpred-access.log;
    error_log /var/log/nginx/retailpred-error.log;

    # API proxy
    location /api {
        proxy_pass http://retailpred_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Docs proxy
    location /docs {
        proxy_pass http://retailpred_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    # Health check
    location /health {
        proxy_pass http://retailpred_backend;
        access_log off;
    }
}
EOF

# Enable site
sudo ln -s /etc/nginx/sites-available/retailpred /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Restart Nginx
sudo systemctl restart nginx
```

### 3.10 Test API Locally

```bash
# From EC2 instance, test API
curl http://localhost:8000/health

# Expected output:
# {"status":"healthy","timestamp":"2025-01-04T12:00:00","database":true,"models_loaded":77}

# Test from your local machine
PUBLIC_IP=$(grep PUBLIC_IP .env.aws | cut -d= -f2)
curl http://$PUBLIC_IP/health

# If successful, proceed to SSL setup
```

### 3.11 Configure SSL/TLS

```bash
# Obtain SSL certificate from Let's Encrypt
sudo certbot --nginx -d retailpred.com -d www.retailpred.com

# Certbot will automatically:
# 1. Generate SSL certificate
# 2. Update Nginx configuration
# 3. Set up auto-renewal

# Test SSL configuration
curl https://retailpred.com/health

# Verify auto-renewal
sudo certbot renew --dry-run
```

**Nginx config after Certbot (automatically updated):**
```nginx
server {
    listen 80;
    server_name retailpred.com www.retailpred.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl;
    server_name retailpred.com www.retailpred.com;

    ssl_certificate /etc/letsencrypt/live/retailpred.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/retailpred.com/privkey.pem;
    include /etc/letsencrypt/options-ssl-nginx.conf;
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;

    # ... rest of configuration
}
```

---

## Phase 4: Frontend Deployment

### 4.1 Build React Application

```bash
# On your local machine

cd frontend

# Install dependencies
npm install

# Create production environment file
cat > .env.production << EOF
VITE_API_URL=https://retailpred.com
EOF

# Build for production
npm run build

# Verify build output
ls -lh dist/
```

### 4.2 Upload to S3

```bash
# Get S3 bucket name
S3_FRONTEND_BUCKET=$(grep S3_FRONTEND_BUCKET .env.aws | cut -d= -f2)

# Sync to S3
aws s3 sync dist/ s3://${S3_FRONTEND_BUCKET}/ \
    --delete \
    --cache-control "public, max-age=31536000, immutable"

# Set index.html to no-cache
aws s3 cp dist/index.html s3://${S3_FRONTEND_BUCKET}/index.html \
    --cache-control "public, max-age=0, must-revalidate"

# Enable static website hosting
aws s3 website s3://${S3_FRONTEND_BUCKET}/ \
    --index-document index.html \
    --error-document index.html

# Verify upload
aws s3 ls s3://${S3_FRONTEND_BUCKET}/
```

### 4.3 Create CloudFront Distribution

```bash
# Get S3 bucket region
S3_REGION=$(aws s3api get-bucket-location \
    --bucket ${S3_FRONTEND_BUCKET} \
    --query LocationConstraint \
    --output text)

# Create CloudFront distribution (save as cloudfront-config.json)
cat > cloudfront-config.json << EOF
{
  "CallerReference": "retailpred-frontend-$(date +%s)",
  "Aliases": {
    "Quantity": 2,
    "Items": ["retailpred.com", "www.retailpred.com"]
  },
  "DefaultRootObject": "index.html",
  "Origins": {
    "Quantity": 1,
    "Items": [
      {
        "Id": "S3-${S3_FRONTEND_BUCKET}",
        "DomainName": "${S3_FRONTEND_BUCKET}.s3-website-${S3_REGION}.amazonaws.com",
        "CustomOriginConfig": {
          "HTTPPort": 80,
          "HTTPSPort": 443,
          "OriginProtocolPolicy": "http-only"
        }
      }
    ]
  },
  "DefaultCacheBehavior": {
    "TargetOriginId": "S3-${S3_FRONTEND_BUCKET}",
    "ViewerProtocolPolicy": "redirect-to-https",
    "AllowedMethods": {
      "Quantity": 3,
      "Items": ["HEAD", "GET", "OPTIONS"]
    },
    "ForwardedValues": {
      "QueryString": false,
      "Cookies": {"Forward": "none"}
    },
    "MinTTL": 0,
    "DefaultTTL": 86400,
    "MaxTTL": 31536000,
    "Compress": true
  },
  "ViewerCertificate": {
    "ACMCertificateArn": "arn:aws:acm:us-east-1:123456789012:certificate/abcdef123456",
    "SSLSupportMethod": "sni-only",
    "MinimumProtocolVersion": "TLSv1.2_2018"
  },
  "PriceClass": "PriceClass_100",
  "Enabled": true
}
EOF

# Create distribution
DISTRIBUTION_ID=$(aws cloudfront create-distribution \
    --distribution-config file://cloudfront-config.json \
    --query 'Distribution.Id' \
    --output text)

echo "CLOUDFRONT_DISTRIBUTION_ID=$DISTRIBUTION_ID" >> .env.aws

# Wait for distribution to deploy (takes ~15 minutes)
aws cloudfront wait distribution-deployed \
    --id $DISTRIBUTION_ID

# Get distribution domain name
DOMAIN_NAME=$(aws cloudfront get-distribution \
    --id $DISTRIBUTION_ID \
    --query 'Distribution.DomainName' \
    --output text)

echo "CloudFront Domain: https://$DOMAIN_NAME"
```

### 4.4 Configure DNS (Route53)

```bash
# Create hosted zone (if you purchased domain from Route53)
HOSTED_ZONE_ID=$(aws route53 create-hosted-zone \
    --name retailpred.com \
    --caller-reference "$(date +%s)" \
    --query 'HostedZone.Id' \
    --output text)

# Or get existing hosted zone ID
HOSTED_ZONE_ID=$(aws route53 list-hosted-zones \
    --query "HostedZones[?Name=='retailpred.com.'].Id" \
    --output text | sed 's/\/hostedzone\///')

# Create A record for CloudFront
cat > route53-a-record.json << EOF
{
  "Comment": "A record for retailpred.com",
  "Changes": [
    {
      "Action": "CREATE",
      "ResourceRecordSet": {
        "Name": "retailpred.com",
        "Type": "A",
        "AliasTarget": {
          "HostedZoneId": "Z2FDTNDATAQYW2",  # CloudFront hosted zone ID
          "DNSName": "${DOMAIN_NAME}",
          "EvaluateTargetHealth": false
        }
      }
    }
  ]
}
EOF

aws route53 change-resource-record-sets \
    --hosted-zone-id $HOSTED_ZONE_ID \
    --change-batch file://route53-a-record.json

# Create CNAME for www
cat > route53-cname-record.json << EOF
{
  "Comment": "CNAME record for www.retailpred.com",
  "Changes": [
    {
      "Action": "CREATE",
      "ResourceRecordSet": {
        "Name": "www.retailpred.com",
        "Type": "CNAME",
        "TTL": 300,
        "ResourceRecords": [
          {"Value": "retailpred.com"}
        ]
      }
    }
  ]
}
EOF

aws route53 change-resource-record-sets \
    --hosted-zone-id $HOSTED_ZONE_ID \
    --change-batch file://route53-cname-record.json
```

### 4.5 Invalidate CloudFront Cache

```bash
# Invalidate all files on first deployment
DISTRIBUTION_ID=$(grep CLOUDFRONT_DISTRIBUTION_ID .env.aws | cut -d= -f2)

aws cloudfront create-invalidation \
    --distribution-id $DISTRIBUTION_ID \
    --paths "/*"

# Verify deployment
curl https://retailpred.com
```

---

## Phase 5: Load Balancing & Scaling

### 5.1 Create AMI from EC2 Instance

```bash
INSTANCE_ID=$(grep INSTANCE_ID .env.aws | cut -d= -f2)

# Create AMI
AMI_ID=$(aws ec2 create-image \
    --instance-id $INSTANCE_ID \
    --name "retailpred-backend-$(date +%Y%m%d-%H%M%S)" \
    --description "RetailPRED Backend Server" \
    --query 'ImageId' \
    --output text)

echo "AMI_ID=$AMI_ID" >> .env.aws

# Wait for AMI to be available
aws ec2 wait image-available --image-ids $AMI_ID
```

### 5.2 Create Target Group

```bash
VPC_ID=$(grep VPC_ID .env.aws | cut -d= -f2)

# Create target group
TARGET_GROUP_ARN=$(aws elbv2 create-target-group \
    --name retailpred-backend-tg \
    --protocol HTTP \
    --port 80 \
    --vpc-id $VPC_ID \
    --health-check Protocol=HTTP,Path=/health,IntervalSeconds=30,UnhealthyThreshold=3,HealthyThreshold=2 \
    --query 'TargetGroups[0].TargetGroupArn' \
    --output text)

echo "TARGET_GROUP_ARN=$TARGET_GROUP_ARN" >> .env.aws
```

### 5.3 Create Application Load Balancer

```bash
SUBNET_ID=$(grep SUBNET_ID .env.aws | cut -d= -f2)
SG_ID=$(grep SECURITY_GROUP_ID .env.aws | cut -d= -f2)

# Create load balancer
ALB_ARN=$(aws elbv2 create-load-balancer \
    --name retailprod-alb \
    --subnets $SUBNET_ID \
    --security-groups $SG_ID \
    --scheme internet-facing \
    --type application \
    --query 'LoadBalancers[0].LoadBalancerArn' \
    --output text)

ALB_DNS_NAME=$(aws elbv2 describe-load-balancers \
    --load-balancer-arns $ALB_ARN \
    --query 'LoadBalancers[0].DNSName' \
    --output text)

echo "ALB_ARN=$ALB_ARN" >> .env.aws
echo "ALB_DNS_NAME=$ALB_DNS_NAME" >> .env.aws

# Wait for ALB to be active
aws elbv2 wait load-balancer-available \
    --load-balancer-arns $ALB_ARN

# Create listener
aws elbv2 create-listener \
    --load-balancer-arn $ALB_ARN \
    --protocol HTTP \
    --port 80 \
    --default-actions Type=forward,TargetGroupArn=$TARGET_GROUP_ARN
```

### 5.4 Create Launch Template

```bash
AMI_ID=$(grep AMI_ID .env.aws | cut -d= -f2)
SG_ID=$(grep SECURITY_GROUP_ID .env.aws | cut -d= -f2)
SUBNET_ID=$(grep SUBNET_ID .env.aws | cut -d= -f2)

# Create launch template
LAUNCH_TEMPLATE_ID=$(aws ec2 create-launch-template \
    --launch-template-name retailpred-backend-lt \
    --launch-template-data "{
        \"ImageId\": \"${AMI_ID}\",
        \"InstanceType\": \"t3.medium\",
        \"KeyName\": \"retailpred-deploy\",
        \"SecurityGroupIds\": [\"${SG_ID}\"],
        \"SubnetId\": \"${SUBNET_ID}\",
        \"IamInstanceProfile\": {
            \"Name\": \"RetailPRED-Instance-Profile\"
        }
    }" \
    --query 'LaunchTemplate.LaunchTemplateId' \
    --output text)

echo "LAUNCH_TEMPLATE_ID=$LAUNCH_TEMPLATE_ID" >> .env.aws
```

### 5.5 Create Auto Scaling Group

```bash
TARGET_GROUP_ARN=$(grep TARGET_GROUP_ARN .env.aws | cut -d= -f2)
LAUNCH_TEMPLATE_ID=$(grep LAUNCH_TEMPLATE_ID .env.aws | cut -d= -f2)
SUBNET_ID=$(grep SUBNET_ID .env.aws | cut -d= -f2)

# Create ASG
ASG_NAME="retailpred-backend-asg"

aws autoscaling create-auto-scaling-group \
    --auto-scaling-group-name $ASG_NAME \
    --launch-template "LaunchTemplateId=${LAUNCH_TEMPLATE_ID},Version=\$Latest" \
    --min-size 2 \
    --max-size 10 \
    --desired-capacity 2 \
    --target-group-arns $TARGET_GROUP_ARN \
    --vpc-zone-identifier $SUBNET_ID \
    --health-check-type ELB \
    --health-check-grace-period 300 \
    --default-instance-warmup 300

# Create scaling policy (CPU-based)
aws autoscaling put-scaling-policy \
    --auto-scaling-group-name $ASG_NAME \
    --policy-name retailpred-scale-out \
    --scaling-adjustment 1 \
    --adjustment-type ChangeInCapacity \
    --cooldown 300 \
    --metric-aggregation-type Average \
    --step-adjustments "MetricIntervalLowerBound=0,ScalingAdjustment=1"

# Create CloudWatch alarm for scaling
aws cloudwatch put-metric-alarm \
    --alarm-name retailpred-high-cpu \
    --alarm-description "Scale out when CPU > 70%" \
    --metric-name CPUUtilization \
    --namespace AWS/EC2 \
    --statistic Average \
    --period 300 \
    --evaluation-periods 2 \
    --threshold 70 \
    --comparison-operator GreaterThanThreshold \
    --dimensions Name=AutoScalingGroupName,Value=$ASG_NAME

echo "Auto Scaling Group created: $ASG_NAME"
```

---

## Phase 6: Monitoring & Backups

### 6.1 Configure CloudWatch Logs

```bash
# Create log group
aws logs create-log-group \
    --log-group-name /aws/ec2/retailpred-api/application \
    --retention-in-days 30

aws logs create-log-group \
    --log-group-name /aws/ec2/retailpred-api/nginx \
    --retention-in-days 30
```

**Update backend/main.py on EC2:**
```python
import watchtower
import logging

# Add CloudWatch handler
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

### 6.2 Create CloudWatch Alarms

```bash
# High error rate alarm
aws cloudwatch put-metric-alarm \
    --alarm-name retailpred-high-errors \
    --alarm-description "Alert when error rate > 5%" \
    --metric-name Errors \
    --namespace AWS/ApplicationELB \
    --statistic Sum \
    --period 300 \
    --evaluation-periods 1 \
    --threshold 10 \
    --comparison-operator GreaterThanThreshold

# High CPU alarm
aws cloudwatch put-metric-alarm \
    --alarm-name retailpred-high-cpu \
    --alarm-description "Alert when CPU > 80%" \
    --metric-name CPUUtilization \
    --namespace AWS/EC2 \
    --statistic Average \
    --period 300 \
    --evaluation-periods 2 \
    --threshold 80 \
    --comparison-operator GreaterThanThreshold

# Low disk space alarm
aws cloudwatch put-metric-alarm \
    --alarm-name retailpred-low-disk \
    --alarm-description "Alert when disk space < 20%" \
    --metric-name DiskSpaceUtilization \
    --namespace System/Linux \
    --statistic Average \
    --period 300 \
    --evaluation-periods 1 \
    --threshold 80 \
    --comparison-operator GreaterThanThreshold
```

### 6.3 Set Up Database Backups

```bash
# On EC2 instance, create backup script
cat > /home/ubuntu/scripts/backup-db.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y-%m-%d)
DB_PATH="/mnt/ebs/data/retailpred.db"
BACKUP_PATH="/mnt/ebs/backups/retailpred.db.$DATE"
S3_BUCKET=$(grep S3_BACKUPS_BUCKET /home/ubuntu/.env.aws | cut -d= -f2)

# Backup to local
cp $DB_PATH $BACKUP_PATH

# Backup to S3
aws s3 cp $DB_PATH s3://${S3_BUCKET}/$DATE/retailpred.db

# Delete local backups older than 30 days
find /mnt/ebs/backups -name "retailpred.db.*" -mtime +30 -delete

echo "Backup completed: $DATE"
EOF

chmod +x /home/ubuntu/scripts/backup-db.sh

# Create systemd service
sudo tee /etc/systemd/system/retailpred-backup.service > /dev/null << 'EOF'
[Unit]
Description=RetailPRED Database Backup
After=network.target

[Service]
Type=oneshot
User=ubuntu
ExecStart=/home/ubuntu/scripts/backup-db.sh

[Install]
WantedBy=multi-user.target
EOF

# Create systemd timer (daily at 2 AM)
sudo tee /etc/systemd/system/retailpred-backup.timer > /dev/null << 'EOF'
[Unit]
Description=Daily RetailPRED Backup
Requires=retailpred-backup.service

[Timer]
OnCalendar=*-*-* 02:00:00
Persistent=true

[Install]
WantedBy=timers.target
EOF

# Enable and start timer
sudo systemctl daemon-reload
sudo systemctl enable retailpred-backup.timer
sudo systemctl start retailpred-backup.timer

# Verify
sudo systemctl list-timers | grep retailpred
```

### 6.4 Configure S3 Lifecycle Policies

```bash
S3_BACKUPS_BUCKET=$(grep S3_BACKUPS_BUCKET .env.aws | cut -d= -f2)

# Transition old backups to Glacier
aws s3api put-bucket-lifecycle-configuration \
    --bucket ${S3_BACKUPS_BUCKET} \
    --lifecycle-configuration '{
        "Rules": [
            {
                "Id": "BackupLifecycle",
                "Status": "Enabled",
                "Filter": {"Prefix": ""},
                "Transitions": [
                    {
                        "Days": 30,
                        "StorageClass": "STANDARD_IA"
                    },
                    {
                        "Days": 90,
                        "StorageClass": "GLACIER"
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
    }'
```

---

## Phase 7: Security Hardening

### 7.1 Configure IAM Roles

```bash
# Create IAM role for EC2 instances
cat > trust-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

ROLE_ARN=$(aws iam create-role \
    --role-name RetailPRED-Instance-Role \
    --assume-role-policy-document file://trust-policy.json \
    --query 'Role.Arn' \
    --output text)

# Attach policies
aws iam attach-role-policy \
    --role-name RetailPRED-Instance-Role \
    --policy-arn arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy

aws iam attach-role-policy \
    --role-name RetailPRED-Instance-Role \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess

# Create instance profile
aws iam create-instance-profile \
    --instance-profile-name RetailPRED-Instance-Profile

aws iam add-role-to-instance-profile \
    --instance-profile-name RetailPRED-Instance-Profile \
    --role-name RetailPRED-Instance-Role
```

### 7.2 Configure S3 Bucket Policies

```bash
S3_MODELS_BUCKET=$(grep S3_MODELS_BUCKET .env.aws | cut -d= -f2)

# Deny unencrypted uploads
aws s3api put-bucket-policy \
    --bucket ${S3_MODELS_BUCKET} \
    --policy '{
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "DenyUnencryptedObjectUploads",
                "Effect": "Deny",
                "Principal": "*",
                "Action": "s3:PutObject",
                "Resource": "arn:aws:s3:::'${S3_MODELS_BUCKET}'/*",
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
                "Resource": "arn:aws:s3:::'${S3_MODELS_BUCKET}'/*",
                "Condition": {
                    "Bool": {
                        "aws:SecureTransport": "false"
                    }
                }
            }
        ]
    }'
```

### 7.3 Enable EBS Encryption

```bash
# Create encryption key (optional - use AWS managed key)
KMS_KEY_ID=$(aws kms create-key \
    --description "RetailPRED EBS Encryption Key" \
    --query 'KeyMetadata.KeyId' \
    --output text)

# (For new instances, use this key when creating volumes)
```

### 7.4 Configure Rate Limiting (Nginx)

```bash
# On EC2 instance, update Nginx config
sudo tee -a /etc/nginx/sites-available/retailpred > /dev/null << 'EOF'

# Rate limiting
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
limit_req_zone $binary_remote_addr zone=predict_limit:10m rate=1r/s;

location /api/predict {
    limit_req zone=predict_limit burst=5 nodelay;
    # ... rest of config
}
EOF

sudo systemctl restart nginx
```

---

## Rollback Procedures

### Rollback Application Code

```bash
# SSH into EC2 instance
ssh ubuntu@retailpred.com

# Navigate to app directory
cd /var/www/retailpred/retailPRED

# View previous commits
git log --oneline -10

# Reset to previous commit
git reset --hard <previous-commit-hash>

# Restart services
sudo systemctl restart retailpred-api
```

### Rollback Database

```bash
# List available backups
aws s3 ls s3://${S3_BACKUPS_BUCKET}/

# Download backup
aws s3 cp s3://${S3_BACKUPS_BUCKET}/2025-01-03/retailpred.db /tmp/restore.db

# Stop application
sudo systemctl stop retailpred-api

# Restore database
cp /mnt/ebs/data/retailpred.db /mnt/ebs/data/retailpred.db.backup
cp /tmp/restore.db /mnt/ebs/data/retailpred.db

# Restart application
sudo systemctl start retailpred-api
```

### Rollback Frontend

```bash
# Upload previous build
aws s3 sync dist-previous/ s3://${S3_FRONTEND_BUCKET}/ --delete

# Invalidate CloudFront cache
aws cloudfront create-invalidation \
    --distribution-id $DISTRIBUTION_ID \
    --paths "/*"
```

### Rollback Infrastructure (Terraform)

```bash
cd infrastructure/terraform

# View previous state
terraform state list

# Rollback to previous infrastructure version
terraform apply \
    -var="domain_name=retailpred.com" \
    -var="bucket_prefix=retailprod" \
    -var="ami_id=<previous-ami-id>"
```

---

## Troubleshooting

### Issue: API returns 502 Bad Gateway

**Diagnosis:**
```bash
# Check if Gunicorn is running
sudo systemctl status retailpred-api

# Check Nginx error logs
sudo tail -f /var/log/nginx/retailpred-error.log

# Check if port 8000 is listening
sudo netstat -tlnp | grep 8000
```

**Solutions:**
1. Restart Gunicorn: `sudo systemctl restart retailpred-api`
2. Check database connection
3. Verify environment variables
4. Check firewall rules

### Issue: Frontend not loading

**Diagnosis:**
```bash
# Check CloudFront distribution status
aws cloudfront get-distribution --id $DISTRIBUTION_ID

# Verify S3 objects are public
aws s3 ls s3://${S3_FRONTEND_BUCKET}/ --recursive

# Check origin is accessible
curl http://${S3_FRONTEND_BUCKET}.s3-website-us-east-1.amazonaws.com/
```

**Solutions:**
1. Invalidate CloudFront cache
2. Verify S3 bucket policy allows public read
3. Check DNS propagation
4. Clear browser cache

### Issue: Database locked

**Diagnosis:**
```bash
# Check for long-running transactions
sqlite3 /mnt/ebs/data/retailpred.db "SELECT * FROM sqlite_master WHERE type='table';"

# Check database integrity
sqlite3 /mnt/ebs/data/retailpred.db "PRAGMA integrity_check;"
```

**Solutions:**
1. Restart application (releases locks)
2. Increase timeout settings
3. Consider migrating to PostgreSQL

### Issue: High CPU usage

**Diagnosis:**
```bash
# Check CPU utilization
aws cloudwatch get-metric-statistics \
    --namespace AWS/EC2 \
    --metric-name CPUUtilization \
    --dimensions Name=InstanceId,Value=$INSTANCE_ID \
    --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) \
    --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
    --period 300 \
    --statistics Average

# Check top processes on EC2
ssh ubuntu@retailpred.com "top -b -n1 | head -20"
```

**Solutions:**
1. Scale out (add more instances)
2. Optimize model loading
3. Increase instance size
4. Add caching layer

### Issue: Out of memory

**Diagnosis:**
```bash
# Check memory usage
ssh ubuntu@retailpred.com "free -h"

# Check for memory leaks
ssh ubuntu@retailpred.com "ps aux --sort=-%mem | head"
```

**Solutions:**
1. Increase instance memory
2. Limit model cache size
3. Add swap space
4. Restart services

---

## Cost Monitoring

### View Current Month's Costs

```bash
# Get current month's costs
aws ce get-cost-and-usage \
    --time-start $(date -u -d "$(date +%Y-%m-01)" +%Y-%m-%d) \
    --time-end $(date -u +%Y-%m-%d) \
    --granularity MONTHLY \
    --metrics BlendedCost \
    --group-by Type=DIMENSION,Key=SERVICE
```

### Set Up Cost Alerts

```bash
# Create billing alarm
aws cloudwatch put-metric-alarm \
    --alarm-name retailpred-high-costs \
    --alarm-description "Alert when monthly costs > $150" \
    --metric-name EstimatedCharges \
    --namespace AWS/Billing \
    --statistic Maximum \
    --period 21600 \
    --evaluation-periods 1 \
    --threshold 150 \
    --comparison-operator GreaterThanThreshold
```

---

## Maintenance Tasks

### Weekly Tasks

- [ ] Review CloudWatch logs for errors
- [ ] Check database size
- [ ] Verify backups are running
- [ ] Review SSL certificate expiry

### Monthly Tasks

- [ ] Review and optimize AWS costs
- [ ] Update dependencies (npm, pip)
- [ ] Review CloudWatch alarms
- [ ] Test backup restoration

### Quarterly Tasks

- [ ] Security audit
- [ ] Performance review
- [ ] Disaster recovery drill
- [ ] Architecture review

---

## Summary

This deployment guide provides a complete roadmap for deploying RetailPRED to AWS with:

✅ **Automated Infrastructure** (Terraform)
✅ **High Availability** (ALB, Auto Scaling)
✅ **Security** (SSL, IAM, encryption)
✅ **Monitoring** (CloudWatch)
✅ **Backups** (Automated daily)
✅ **Rollback** (Git, database, infrastructure)

**Estimated Time to Deploy:**
- **Manual**: 6-8 hours
- **Automated (Terraform)**: 1-2 hours

**Monthly Cost:**
- **Development** (single EC2): ~$60/month
- **Production** (ALB + 2 EC2): ~$143/month
- **With Reserved Instances**: ~$96/month (33% savings)

**Next Steps:**
1. Complete Phase 1 (Account Setup)
2. Choose deployment method (Terraform vs Manual)
3. Follow Phase 2-7 sequentially
4. Test all functionality
5. Set up monitoring and alerts
6. Document your deployment specifics

For questions or issues, refer to [AWS_ARCHITECTURE.md](AWS_ARCHITECTURE.md) or create an issue in the repository.
