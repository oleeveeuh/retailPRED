# Terraform Variables for RetailPRED AWS Infrastructure

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "retailpred"
}

variable "environment" {
  description = "Environment (dev, staging, production)"
  type        = string
  default     = "production"

  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "Environment must be dev, staging, or production."
  }
}

variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-east-1"

  validation {
    condition     = can(regex("^us-[a-z]+-[1-9]$", var.aws_region))
    error_message = "AWS region must be a valid US region (e.g., us-east-1)."
  }
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "List of availability zones"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b"]
}

variable "bucket_prefix" {
  description = "Prefix for S3 bucket names (must be globally unique)"
  type        = string
  default     = "retailprod"

  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9-]{1,61}[a-z0-9]$", var.bucket_prefix))
    error_message = "Bucket prefix must be 3-63 characters, lowercase, alphanumeric, with hyphens."
  }
}

variable "ec2_instance_type" {
  description = "EC2 instance type for backend servers"
  type        = string
  default     = "t3.medium"

  validation {
    condition     = can(regex("^t[23]\\.(micro|small|medium|large|xlarge|2xlarge)$", var.ec2_instance_type))
    error_message = "Instance type must be a valid t2 or t3 instance."
  }
}

variable "ebs_data_size" {
  description = "Size of EBS volume for data (GB)"
  type        = number
  default     = 100

  validation {
    condition     = var.ebs_data_size >= 20 && var.ebs_data_size <= 1000
    error_message = "EBS size must be between 20 and 1000 GB."
  }
}

variable "asg_min_size" {
  description = "Minimum number of instances in Auto Scaling Group"
  type        = number
  default     = 2

  validation {
    condition     = var.asg_min_size >= 1
    error_message = "Minimum size must be at least 1."
  }
}

variable "asg_max_size" {
  description = "Maximum number of instances in Auto Scaling Group"
  type        = number
  default     = 10

  validation {
    condition     = var.asg_max_size >= var.asg_min_size
    error_message = "Maximum size must be greater than or equal to minimum size."
  }
}

variable "asg_desired_capacity" {
  description = "Desired number of instances in Auto Scaling Group"
  type        = number
  default     = 2

  validation {
    condition     = var.asg_desired_capacity >= var.asg_min_size && var.asg_desired_capacity <= var.asg_max_size
    error_message = "Desired capacity must be between min and max size."
  }
}

variable "ssh_public_key" {
  description = "SSH public key for EC2 access (e.g., '~/.ssh/id_rsa.pub')"
  type        = string
  sensitive   = true
}

variable "ssh_allowed_ips" {
  description = "List of CIDR blocks allowed to SSH (use your IP only!)"
  type        = list(string)
  default     = ["0.0.0.0/0"]

  validation {
    condition     = length(var.ssh_allowed_ips) > 0
    error_message = "At least one SSH allowed IP must be specified."
  }
}

variable "domain_name" {
  description = "Domain name for the application (optional)"
  type        = string
  default     = ""
}

variable "ssl_certificate_arn" {
  description = "ARN of SSL certificate in AWS Certificate Manager"
  type        = string
  default     = ""
}

variable "enable_cloudfront" {
  description = "Enable CloudFront CDN for frontend"
  type        = bool
  default     = true
}

variable "cloudfront_price_class" {
  description = "CloudFront price class (PriceClass_100, PriceClass_200, or PriceClass_All)"
  type        = string
  default     = "PriceClass_100" # US, Canada, Europe

  validation {
    condition     = contains(["PriceClass_100", "PriceClass_200", "PriceClass_All"], var.cloudfront_price_class)
    error_message = "Price class must be PriceClass_100, PriceClass_200, or PriceClass_All."
  }
}

variable "log_retention_days" {
  description = "Retention period for CloudWatch Logs (days)"
  type        = number
  default     = 30

  validation {
    condition     = contains([1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365, 400, 545, 731, 1827, 3653], var.log_retention_days)
    error_message = "Log retention must be a valid CloudWatch retention period."
  }
}

variable "sns_topic_arn" {
  description = "SNS topic ARN for CloudWatch alarm notifications"
  type        = string
  default     = ""
}

variable "tags" {
  description = "Additional tags for all resources"
  type        = map(string)
  default     = {
    Owner = "DevOps"
  }
}
