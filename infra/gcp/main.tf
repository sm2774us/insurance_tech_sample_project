# Cluster-mode object storage backing the Data Access Layer's URI-swapped
# storage path.
resource "google_storage_bucket" "lake" {
  name                        = "${var.project_prefix}-${var.environment}-lake"
  location                    = var.gcp_region
  uniform_bucket_level_access = true
  force_destroy               = false

  versioning {
    enabled = true
  }

  encryption {
    default_kms_key_name = google_kms_crypto_key.lake.id
  }
}

resource "google_kms_key_ring" "lake" {
  name     = "${var.project_prefix}-${var.environment}-keyring"
  location = var.gcp_region
}

resource "google_kms_crypto_key" "lake" {
  name     = "${var.project_prefix}-${var.environment}-lake-key"
  key_ring = google_kms_key_ring.lake.id
}

# Cloud Run service hosting the FastAPI web backend + Taipy frontend.
resource "google_cloud_run_v2_service" "api" {
  name     = "${var.project_prefix}-${var.environment}-api"
  location = var.gcp_region

  template {
    containers {
      image = var.api_container_image
      resources {
        limits = {
          cpu    = "2"
          memory = "4Gi"
        }
      }
    }
  }
}

# GKE node pool for Ray Train / distributed PyTorch retraining jobs.
resource "google_container_cluster" "ray_train" {
  name                     = "${var.project_prefix}-${var.environment}-ray"
  location                 = var.gcp_region
  remove_default_node_pool = true
  initial_node_count       = 1
}

resource "google_container_node_pool" "ray_train_pool" {
  name       = "${var.project_prefix}-${var.environment}-ray-pool"
  cluster    = google_container_cluster.ray_train.name
  location   = var.gcp_region
  node_count = 0

  autoscaling {
    min_node_count = 0
    max_node_count = 8
  }

  node_config {
    machine_type = "n2-standard-8"
  }
}

variable "api_container_image" {
  description = "Fully qualified container image for the FastAPI backend."
  type        = string
  default     = "gcr.io/PROJECT_ID/fig-quant-api:latest"
}
