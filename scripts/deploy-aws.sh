#!/bin/bash
# RetailPRED AWS Deployment Script
# Automates the deployment of RetailPRED to AWS

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TERRAFORM_DIR="$PROJECT_ROOT/infrastructure/terraform"
FRONTEND_DIR="$PROJECT_ROOT/frontend"

# Default values
ENVIRONMENT="production"
DOMAIN_NAME=""
SSH_KEY_PATH="$HOME/.ssh/id_rsa.pub"
AUTO_APPROVE=false

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy RetailPRED to AWS

OPTIONS:
    -e, --environment ENV       Environment (dev, staging, production) [default: production]
    -d, --domain DOMAIN         Domain name (optional)
    -k, --ssh-key PATH          Path to SSH public key [default: ~/.ssh/id_rsa.pub]
    -a, --auto-approve          Auto-approve Terraform apply (use with caution)
    -h, --help                  Show this help message

EXAMPLES:
    # Deploy to production
    $0 --environment production --domain retailpred.com

    # Deploy to dev with auto-approve
    $0 --environment dev --auto-approve

EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -d|--domain)
                DOMAIN_NAME="$2"
                shift 2
                ;;
            -k|--ssh-key)
                SSH_KEY_PATH="$2"
                shift 2
                ;;
            -a|--auto-approve)
                AUTO_APPROVE=true
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

validate_prerequisites() {
    log_info "Validating prerequisites..."

    # Check if AWS CLI is installed
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI is not installed. Please install it first."
        exit 1
    fi

    # Check if Terraform is installed
    if ! command -v terraform &> /dev/null; then
        log_error "Terraform is not installed. Please install it first."
        exit 1
    fi

    # Check if Node.js is installed (for frontend build)
    if ! command -v npm &> /dev/null; then
        log_error "Node.js/npm is not installed. Please install it first."
        exit 1
    fi

    # Check if SSH key exists
    if [ ! -f "$SSH_KEY_PATH" ]; then
        log_error "SSH key not found at $SSH_KEY_PATH"
        exit 1
    fi

    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured. Run 'aws configure' first."
        exit 1
    fi

    log_success "All prerequisites met"
}

initialize_terraform() {
    log_info "Initializing Terraform..."

    cd "$TERRAFORM_DIR"

    # Create S3 bucket for Terraform state (if it doesn't exist)
    STATE_BUCKET="retailpred-terraform-state"
    REGION=$(terraform output -raw aws_region 2>/dev/null || echo "us-east-1")

    if ! aws s3 ls "s3://$STATE_BUCKET" 2>&1 | grep -q "Unable to locate"; then
        log_info "Creating Terraform state bucket..."
        aws s3 mb "s3://$STATE_BUCKET" --region "$REGION" 2>/dev/null || log_warning "State bucket may already exist"
        aws s3api put-bucket-versioning \
            --bucket "$STATE_BUCKET" \
            --versioning-configuration Status=Enabled \
            --region "$REGION" 2>/dev/null || true
    fi

    # Create DynamoDB table for state locking (if it doesn't exist)
    if ! aws dynamodb describe-table \
        --table-name retailpred-terraform-locks \
        --region "$REGION" &> /dev/null; then
        log_info "Creating Terraform state lock table..."
        aws dynamodb create-table \
            --table-name retailpred-terraform-locks \
            --attribute-definitions AttributeName=LockID,AttributeType=S \
            --key-schema AttributeName=LockID,KeyType=HASH \
            --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5 \
            --region "$REGION" \
            --tags Key=Name,Value=TerraformStateLock \
            2>/dev/null || log_warning "Lock table may already exist"
    fi

    # Initialize Terraform
    terraform init \
        -backend-config="bucket=$STATE_BUCKET" \
        -backend-config="key=infrastructure/terraform.tfstate" \
        -backend-config="region=$REGION" \
        -backend-config="dynamodb_table=retailpred-terraform-locks"

    log_success "Terraform initialized"
}

deploy_infrastructure() {
    log_info "Deploying AWS infrastructure..."

    cd "$TERRAFORM_DIR"

    # Read SSH public key
    SSH_PUBLIC_KEY=$(cat "$SSH_KEY_PATH")

    # Build Terraform plan
    TF_VARS="-var=\"environment=$ENVIRONMENT\""
    TF_VARS="$TF_VARS -var=\"ssh_public_key=$SSH_PUBLIC_KEY\""

    if [ -n "$DOMAIN_NAME" ]; then
        TF_VARS="$TF_VARS -var=\"domain_name=$DOMAIN_NAME\""
    fi

    # Run Terraform apply
    log_info "Running Terraform apply..."
    if [ "$AUTO_APPROVE" = true ]; then
        terraform apply $TF_VARS -auto-approve
    else
        terraform apply $TF_VARS
    fi

    log_success "Infrastructure deployed successfully"

    # Save outputs to file
    terraform output -json > "$PROJECT_ROOT/terraform-outputs.json"
    log_success "Terraform outputs saved to terraform-outputs.json"
}

build_frontend() {
    log_info "Building frontend application..."

    cd "$FRONTEND_DIR"

    # Install dependencies
    log_info "Installing frontend dependencies..."
    npm install

    # Create production environment file
    log_info "Creating production environment..."

    # Get API URL from Terraform outputs or use default
    if [ -f "$PROJECT_ROOT/terraform-outputs.json" ]; then
        ALB_DNS=$(jq -r '.load_balancer_dns_name.value' "$PROJECT_ROOT/terraform-outputs.json")
        API_URL="https://$ALB_DNS"
    else
        API_URL="http://localhost:8000"
    fi

    cat > .env.production << EOF
VITE_API_URL=$API_URL
EOF

    # Build frontend
    log_info "Building frontend bundle..."
    npm run build

    log_success "Frontend built successfully"
}

deploy_frontend() {
    log_info "Deploying frontend to S3..."

    # Check if terraform outputs exist
    if [ ! -f "$PROJECT_ROOT/terraform-outputs.json" ]; then
        log_error "Terraform outputs not found. Please deploy infrastructure first."
        exit 1
    fi

    # Get S3 bucket name
    S3_BUCKET=$(jq -r '.s3_frontend_bucket.value' "$PROJECT_ROOT/terraform-outputs.json")

    # Sync to S3
    log_info "Uploading frontend files to S3..."
    aws s3 sync "$FRONTEND_DIR/dist/" "s3://$S3_BUCKET/" --delete

    # Set cache headers
    log_info "Setting cache headers..."
    aws s3 cp "$FRONTEND_DIR/dist/index.html" "s3://$S3_BUCKET/index.html" \
        --cache-control "public, max-age=0, must-revalidate"

    # Invalidate CloudFront cache (if enabled)
    CLOUDFRONT_ID=$(jq -r '.cloudfront_distribution_id.value?' "$PROJECT_ROOT/terraform-outputs.json")
    if [ "$CLOUDFRONT_ID" != "null" ] && [ -n "$CLOUDFRONT_ID" ]; then
        log_info "Invalidating CloudFront cache..."
        aws cloudfront create-invalidation \
            --distribution-id "$CLOUDFRONT_ID" \
            --paths "/*"

        log_success "CloudFront cache invalidated"
    fi

    log_success "Frontend deployed successfully"
}

deploy_backend() {
    log_info "Deploying backend to EC2..."

    # Check if terraform outputs exist
    if [ ! -f "$PROJECT_ROOT/terraform-outputs.json" ]; then
        log_error "Terraform outputs not found. Please deploy infrastructure first."
        exit 1
    fi

    # Get EC2 instance IP
    ASG_NAME=$(jq -r '.autoscaling_group_name.value' "$PROJECT_ROOT/terraform-outputs.json")

    # Get running instances in ASG
    log_info "Finding EC2 instances in Auto Scaling Group..."
    INSTANCE_IDS=$(aws autoscaling describe-auto-scaling-groups \
        --auto-scaling-group-names "$ASG_NAME" \
        --query "AutoScalingGroups[0].Instances[?HealthStatus=='Healthy'].InstanceId" \
        --output text)

    if [ -z "$INSTANCE_IDS" ]; then
        log_error "No healthy instances found in ASG"
        exit 1
    fi

    # Get first instance IP
    INSTANCE_ID=$(echo "$INSTANCE_IDS" | awk '{print $1}')
    PUBLIC_IP=$(aws ec2 describe-instances \
        --instance-ids "$INSTANCE_ID" \
        --query "Reservations[0].Instances[0].PublicIpAddress" \
        --output text)

    log_info "Deploying to EC2 instance: $INSTANCE_ID ($PUBLIC_IP)"

    # SSH into instance and deploy
    log_warning "This script requires manual SSH access. Please:"
    echo ""
    echo "1. SSH into the instance:"
    echo "   ssh -i $(dirname "$SSH_KEY_PATH")/$(basename "$SSH_KEY_PATH" .pub) ubuntu@$PUBLIC_IP"
    echo ""
    echo "2. Run these commands:"
    echo "   cd /var/www/retailpred"
    echo "   git clone https://github.com/oleeveeuh/retailPRED.git ."
    echo "   source venv/bin/activate"
    echo "   pip install -r backend/requirements.txt"
    echo "   sudo systemctl start retailpred-api"
    echo ""
    echo "3. Check health:"
    echo "   curl http://localhost:8000/health"
    echo ""
}

show_summary() {
    log_success "Deployment complete!"

    echo ""
    echo "========================================="
    echo "      Deployment Summary"
    echo "========================================="
    echo ""

    if [ -f "$PROJECT_ROOT/terraform-outputs.json" ]; then
        ALB_DNS=$(jq -r '.load_balancer_dns_name.value' "$PROJECT_ROOT/terraform-outputs.json")
        CLOUDFRONT_DNS=$(jq -r '.cloudfront_domain_name.value?' "$PROJECT_ROOT/terraform-outputs.json")
        S3_BUCKET=$(jq -r '.s3_frontend_bucket.value' "$PROJECT_ROOT/terraform-outputs.json")

        echo "Infrastructure:"
        echo "  - Load Balancer: http://$ALB_DNS"
        if [ "$CLOUDFRONT_DNS" != "null" ] && [ -n "$CLOUDFRONT_DNS" ]; then
            echo "  - CloudFront: https://$CLOUDFRONT_DNS"
        fi
        echo "  - S3 Bucket: s3://$S3_BUCKET"
        echo ""
    fi

    echo "Next Steps:"
    echo "  1. SSH into an EC2 instance and deploy the backend code"
    echo "  2. Test the API health endpoint"
    echo "  3. Configure DNS (if using custom domain)"
    echo "  4. Set up SSL certificate with Let's Encrypt"
    echo ""
    echo "For detailed instructions, see: docs/DEPLOYMENT_GUIDE.md"
    echo ""
}

# Main execution
main() {
    echo "========================================="
    echo "  RetailPRED AWS Deployment Script"
    echo "========================================="
    echo ""

    parse_args "$@"
    validate_prerequisites
    initialize_terraform
    deploy_infrastructure
    build_frontend
    deploy_frontend
    deploy_backend
    show_summary
}

# Run main function
main "$@"
