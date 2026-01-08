# RetailPRED AWS Infrastructure - Terraform

This directory contains the Terraform configuration for deploying RetailPRED to AWS.

## Quick Start

```bash
# 1. Configure AWS credentials
aws configure

# 2. Initialize Terraform
terraform init

# 3. Review the plan
terraform plan \
    -var="environment=production" \
    -var="domain_name=retailpred.com" \
    -var="ssh_public_key=$(cat ~/.ssh/id_rsa.pub)"

# 4. Deploy infrastructure
terraform apply \
    -var="environment=production" \
    -var="domain_name=retailpred.com" \
    -var="ssh_public_key=$(cat ~/.ssh/id_rsa.pub)"
```

## Architecture

This Terraform configuration creates:

- **VPC & Networking**: VPC, subnets, route tables, internet gateway
- **Security Groups**: Web security group, ALB security group
- **IAM Roles**: EC2 instance role with S3 and CloudWatch permissions
- **S3 Buckets**: Frontend static files, ML models, database backups
- **EC2 Instances**: Launch template for backend servers
- **Load Balancing**: Application Load Balancer with target group
- **Auto Scaling**: ASG with target tracking policy (CPU-based)
- **CloudWatch**: Log groups and metric alarms
- **Route53**: Hosted zone and DNS records (optional)
- **CloudFront**: CDN for frontend (optional)

## Files

- [`main.tf`](main.tf) - Main Terraform configuration
- [`variables.tf`](variables.tf) - Input variables
- [`outputs.tf`](outputs.tf) - Output values
- [`user-data.sh`](user-data.sh) - EC2 user data script
- [`README.md`](README.md) - This file

## Variables

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `ssh_public_key` | SSH public key for EC2 access | `~/.ssh/id_rsa.pub` |

### Optional Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `environment` | Environment name | `production` |
| `aws_region` | AWS region | `us-east-1` |
| `project_name` | Project name for resources | `retailpred` |
| `bucket_prefix` | Prefix for S3 buckets | `retailprod` |
| `domain_name` | Domain name (optional) | `""` |
| `ssl_certificate_arn` | SSL certificate ARN | `""` |
| `ec2_instance_type` | EC2 instance type | `t3.medium` |
| `asg_min_size` | Minimum ASG size | `2` |
| `asg_max_size` | Maximum ASG size | `10` |
| `asg_desired_capacity` | Desired ASG capacity | `2` |
| `ebs_data_size` | EBS data volume size (GB) | `100` |
| `enable_cloudfront` | Enable CloudFront CDN | `true` |
| `log_retention_days` | CloudWatch log retention | `30` |
| `ssh_allowed_ips` | IPs allowed to SSH | `["0.0.0.0/0"]` |

## Outputs

After deployment, Terraform outputs:

- `vpc_id` - VPC ID
- `s3_frontend_bucket` - Frontend S3 bucket name
- `s3_models_bucket` - ML models S3 bucket name
- `s3_backups_bucket` - Backups S3 bucket name
- `load_balancer_dns_name` - ALB DNS name
- `cloudfront_domain_name` - CloudFront domain name
- `autoscaling_group_name` - ASG name
- `deployment_instructions` - Deployment instructions

## Usage Examples

### Deploy to Production

```bash
terraform apply \
    -var="environment=production" \
    -var="domain_name=retailpred.com" \
    -var="ssh_public_key=$(cat ~/.ssh/id_rsa.pub)"
```

### Deploy to Development (Single Instance)

```bash
terraform apply \
    -var="environment=dev" \
    -var="asg_min_size=1" \
    -var="asg_max_size=1" \
    -var="asg_desired_capacity=1" \
    -var="enable_cloudfront=false" \
    -var="ssh_public_key=$(cat ~/.ssh/id_rsa.pub)"
```

### Deploy with Custom Instance Type

```bash
terraform apply \
    -var="ec2_instance_type=t3.large" \
    -var="ssh_public_key=$(cat ~/.ssh/id_rsa.pub)"
```

### Deploy with Larger EBS Volume

```bash
terraform apply \
    -var="ebs_data_size=200" \
    -var="ssh_public_key=$(cat ~/.ssh/id_rsa.pub)"
```

## State Management

Terraform state is stored in S3 with DynamoDB locking for team collaboration:

```
s3://retailpred-terraform-state/infrastructure/terraform.tfstate
```

### Initialize Backend

```bash
terraform init \
    -backend-config="bucket=retailpred-terraform-state" \
    -backend-config="key=infrastructure/terraform.tfstate" \
    -backend-config="region=us-east-1" \
    -backend-config="dynamodb_table=retailpred-terraform-locks"
```

### Import Existing Resources

If you have existing AWS resources, import them:

```bash
# Import existing S3 bucket
terraform import aws_s3_bucket.frontend retailprod-frontend

# Import existing EC2 instance
terraform import aws_autoscaling_group.backend retailpred-backend-asg
```

## Modules

This configuration uses no external modules - everything is defined in [`main.tf`](main.tf) for transparency and ease of modification.

## Security

### SSH Access

**IMPORTANT**: Restrict SSH access to your IP only:

```bash
# Get your current IP
MY_IP=$(curl -s https://checkip.amazonaws.com)/32

terraform apply \
    -var="ssh_allowed_ips=[\"$MY_IP\"]" \
    -var="ssh_public_key=$(cat ~/.ssh/id_rsa.pub)"
```

### SSL Certificates

For production use with a custom domain:

1. Create SSL certificate in AWS Certificate Manager
2. Get certificate ARN
3. Deploy with certificate:

```bash
terraform apply \
    -var="domain_name=retailpred.com" \
    -var="ssl_certificate_arn=arn:aws:acm:us-east-1:123456789012:certificate/abcdef123456" \
    -var="ssh_public_key=$(cat ~/.ssh/id_rsa.pub)"
```

### Security Groups

Security groups follow the principle of least privilege:

- **Web SG**: Allows HTTP (80) and HTTPS (443) from anywhere, SSH from specific IPs
- **ALB SG**: Allows HTTP (80) and HTTPS (443) from anywhere

## Cost Estimation

Estimated monthly costs (us-east-1):

| Service | Specification | Monthly Cost |
|---------|---------------|--------------|
| EC2 | 2 × t3.medium | $60.00 |
| EBS | 100 GB gp3 | $8.00 |
| S3 | 20 GB storage | $0.46 |
| ALB | 1 ALB | $18.25 |
| CloudFront | 100 GB transfer | $8.50 |
| Route53 | Hosted zone | $0.50 |
| CloudWatch | 5 GB logs | $2.50 |
| Data Transfer | 500 GB | $45.00 |
| **TOTAL** | | **$143.21** |

**Cost Optimization Tips:**
- Use Reserved Instances (save 30-50%)
- Use Spot Instances for worker nodes (save 70%)
- Enable S3 lifecycle policies (archive old data)
- Use CloudFront compression (reduce transfer by 50%)

## Monitoring

### CloudWatch Alarms

The following alarms are created automatically:

- `retailpred-high-cpu` - Triggers when CPU > 80%
- `retailpred-high-errors` - Triggers when 5XX errors > 10 in 5 minutes

### Log Groups

- `/aws/ec2/retailpred/application` - Application logs
- `/aws/ec2/retailpred/nginx` - Nginx logs

View logs:
```bash
aws logs tail /aws/ec2/retailpred/application --follow
```

## Scaling

### Auto Scaling Policy

Target tracking policy maintains CPU at 70%:

- **Scale Out**: CPU > 70% for 5 minutes → Add instance
- **Scale In**: CPU < 30% for 5 minutes → Remove instance

### Manual Scaling

```bash
# Update desired capacity
aws autoscaling set-desired-capacity \
    --auto-scaling-group-name retailpred-backend-asg \
    --desired-capacity 4

# Update min/max size
aws autoscaling update-auto-scaling-group \
    --auto-scaling-group-name retailpred-backend-asg \
    --min-size 2 \
    --max-size 20
```

## Troubleshooting

### Terraform State Lock

If you get a state lock error:

```bash
# Force unlock (use with caution!)
terraform force-unlock <LOCK_ID>

# Or check who has the lock:
aws dynamodb get-item \
    --table-name retailpred-terraform-locks \
    --key '{"LockID": {"S": "infrastructure/terraform.tfstate-md5"}}'
```

### Resource Not Found

If Terraform can't find a resource:

```bash
# Refresh state
terraform refresh

# Re-create state if needed
terraform plan -refresh=false
```

### Apply Fails

If `terraform apply` fails:

```bash
# View detailed error
terraform apply -trace

# Check Terraform version
terraform version

# Re-initialize
terraform init -reconfigure
```

## Updates and Maintenance

### Update Infrastructure

```bash
# Make changes to main.tf or variables.tf

# Review the plan
terraform plan

# Apply changes
terraform apply
```

### Add New Resource

```bash
# Add resource to main.tf

# Import existing resource (if needed)
terraform import aws_<resource_type>.<resource_name> <resource_id>

# Apply
terraform apply
```

### Destroy Infrastructure

**WARNING**: This will delete all resources!

```bash
terraform destroy
```

Or destroy specific resources:

```bash
terraform destroy -target=aws_autoscaling_group.backend
```

## Best Practices

### 1. Use Separate Workspaces

```bash
terraform workspace new dev
terraform apply -var="environment=dev"

terraform workspace new production
terraform apply -var="environment=production"
```

### 2. Version Control

- Commit `.tf` files to Git
- Add `.tfstate` to `.gitignore`
- Use remote state (S3 + DynamoDB)

### 3. Validate Configuration

```bash
# Validate syntax
terraform validate

# Format code
terraform fmt -recursive

# Check for security issues
terraform fmt -check=false
```

### 4. Use Variables File

Create `terraform.tfvars`:

```hcl
environment     = "production"
domain_name     = "retailpred.com"
ssh_public_key  = "~/.ssh/id_rsa.pub"
ec2_instance_type = "t3.medium"
```

Then run:
```bash
terraform apply
```

### 5. Tag Resources

All resources are automatically tagged:

```hcl
tags = {
  Project     = "RetailPRED"
  Environment = var.environment
  ManagedBy   = "Terraform"
}
```

## Testing

### Test Configuration Locally

```bash
# Validate syntax
terraform validate

# Format check
terraform fmt -check

# Plan without applying
terraform plan
```

### Test in Dev Environment

```bash
terraform workspace new dev
terraform apply \
    -var="environment=dev" \
    -var="asg_min_size=1" \
    -var="asg_max_size=1"
```

### Load Testing

After deployment, test with:

```bash
# Install siege
sudo apt install siege

# Test health endpoint
siege -c 10 -t 30S http://<ALB-DNS>/health

# Test prediction endpoint
siege -c 5 -t 60S "http://<ALB-DNS>/api/predict POST content-type:application/json '{\"category\":\"total_sales\",\"weeks_ahead\":4}'"
```

## Support

For issues or questions:

1. Check the [main documentation](../../docs/AWS_ARCHITECTURE.md)
2. Review the [deployment guide](../../docs/DEPLOYMENT_GUIDE.md)
3. Create an issue in the repository
4. Check Terraform logs: `terraform apply -trace`

## Contributing

When modifying Terraform configuration:

1. Follow Terraform best practices
2. Add comments for complex logic
3. Update this README
4. Test in dev environment first
5. Update `variables.tf` if adding new variables
6. Update `outputs.tf` if adding new outputs

## License

This infrastructure code is part of RetailPRED. See main project LICENSE.
