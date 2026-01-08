# Terraform Outputs for RetailPRED AWS Infrastructure

output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "public_subnet_ids" {
  description = "IDs of public subnets"
  value       = aws_subnet.public[*].id
}

output "security_group_web_id" {
  description = "ID of the web security group"
  value       = aws_security_group.web.id
}

output "s3_frontend_bucket" {
  description = "Name of the S3 bucket for frontend static files"
  value       = aws_s3_bucket.frontend.id
}

output "s3_frontend_bucket_arn" {
  description = "ARN of the S3 bucket for frontend static files"
  value       = aws_s3_bucket.frontend.arn
}

output "s3_models_bucket" {
  description = "Name of the S3 bucket for ML models"
  value       = aws_s3_bucket.models.id
}

output "s3_models_bucket_arn" {
  description = "ARN of the S3 bucket for ML models"
  value       = aws_s3_bucket.models.arn
}

output "s3_backups_bucket" {
  description = "Name of the S3 bucket for database backups"
  value       = aws_s3_bucket.backups.id
}

output "s3_backups_bucket_arn" {
  description = "ARN of the S3 bucket for database backups"
  value       = aws_s3_bucket.backups.arn
}

output "load_balancer_dns_name" {
  description = "DNS name of the Application Load Balancer"
  value       = aws_lb.main.dns_name
}

output "load_balancer_zone_id" {
  description = "Zone ID of the Application Load Balancer (for Route53 alias records)"
  value       = aws_lb.main.zone_id
}

output "load_balancer_arn" {
  description = "ARN of the Application Load Balancer"
  value       = aws_lb.main.arn
}

output "target_group_arn" {
  description = "ARN of the ALB Target Group"
  value       = aws_lb_target_group.backend.arn
}

output "autoscaling_group_name" {
  description = "Name of the Auto Scaling Group"
  value       = aws_autoscaling_group.backend.name
}

output "launch_template_id" {
  description = "ID of the EC2 Launch Template"
  value       = aws_launch_template.backend.id
}

output "key_pair_name" {
  description = "Name of the SSH key pair"
  value       = aws_key_pair.deploy.key_name
}

output "cloudfront_distribution_id" {
  description = "ID of the CloudFront distribution (if enabled)"
  value       = var.enable_cloudfront ? aws_cloudfront_distribution.frontend[0].id : null
}

output "cloudfront_domain_name" {
  description = "Domain name of the CloudFront distribution (if enabled)"
  value       = var.enable_cloudfront ? aws_cloudfront_distribution.frontend[0].domain_name : null
}

output "route53_name_servers" {
  description = "Name servers for the Route53 hosted zone (if domain provided)"
  value       = var.domain_name != "" ? aws_route53_zone.main[0].name_servers : null
}

output "cloudwatch_log_group_application" {
  description = "Name of the application log group"
  value       = aws_cloudwatch_log_group.application.name
}

output "cloudwatch_log_group_nginx" {
  description = "Name of the Nginx log group"
  value       = aws_cloudwatch_log_group.nginx.name
}

output "ec2_instance_role_arn" {
  description = "ARN of the IAM role for EC2 instances"
  value       = aws_iam_role.ec2_role.arn
}

output "ebs_volume_id" {
  description = "ID of the EBS data volume"
  value       = aws_ebs_volume.data.id
}

# =============================================================================
# USAGE INSTRUCTIONS
# =============================================================================

output "deployment_instructions" {
  description = "Instructions for deploying the application"
  value       = <<EOT
=======================================
RetailPRED AWS Deployment Complete!
=======================================

1. Deploy Backend:
   SSH into one of the EC2 instances and run:
   - git clone https://github.com/oleeveeuh/retailPRED.git
   - Configure environment variables
   - Start services

2. Deploy Frontend:
   npm run build
   aws s3 sync dist/ s3://${aws_s3_bucket.frontend.id}/
   aws cloudfront create-invalidation --distribution-id ${var.enable_cloudfront ? aws_cloudfront_distribution.frontend[0].id : "N/A"} --paths "/*"

3. Access Application:
   - Load Balancer: http://${aws_lb.main.dns_name}
   - CloudFront: https://${var.enable_cloudfront ? aws_cloudfront_distribution.frontend[0].domain_name : "disabled"}

4. Configure DNS (if using Route53):
   - Point your domain to the ALB DNS name
   - Or use the Route53 records created automatically

5. Monitor:
   - CloudWatch: https://console.aws.amazon.com/cloudwatch/
   - Auto Scaling Group: ${aws_autoscaling_group.backend.name}
   - Target Group: ${aws_lb_target_group.backend.name}

For detailed deployment steps, see: docs/DEPLOYMENT_GUIDE.md
EOT
}
