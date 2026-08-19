# Cluster-mode object storage backing the Data Access Layer's `s3://` URI
# path (swaps transparently with the standalone `file://` local-Parquet
# path — see src/fig_quant/pit/bitemporal.py).
resource "aws_s3_bucket" "lake" {
  bucket = "${var.project_prefix}-${var.environment}-lake"
}

resource "aws_s3_bucket_versioning" "lake" {
  bucket = aws_s3_bucket.lake.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "lake" {
  bucket = aws_s3_bucket.lake.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "lake" {
  bucket                  = aws_s3_bucket.lake.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ECS Fargate service hosting the FastAPI web backend + Taipy frontend.
resource "aws_ecr_repository" "api" {
  name                 = "${var.project_prefix}-${var.environment}-api"
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecs_cluster" "pipeline" {
  name = "${var.project_prefix}-${var.environment}-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# Ray/PyIceberg distributed training compute, sized independently of the
# always-on API service so batch retraining jobs can scale to zero.
resource "aws_batch_compute_environment" "ray_train" {
  compute_environment_name = "${var.project_prefix}-${var.environment}-ray-train"
  type                      = "MANAGED"
  service_role              = aws_iam_role.batch_service_role.arn

  compute_resources {
    type               = "FARGATE"
    max_vcpus          = 64
    subnets            = var.subnet_ids
    security_group_ids = var.security_group_ids
  }

  depends_on = [aws_iam_role_policy_attachment.batch_service_role]
}

resource "aws_iam_role" "batch_service_role" {
  name = "${var.project_prefix}-${var.environment}-batch-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "batch.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "batch_service_role" {
  role       = aws_iam_role.batch_service_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole"
}

variable "subnet_ids" {
  description = "Subnet IDs for the Batch/Ray compute environment."
  type        = list(string)
  default     = []
}

variable "security_group_ids" {
  description = "Security group IDs for the Batch/Ray compute environment."
  type        = list(string)
  default     = []
}
