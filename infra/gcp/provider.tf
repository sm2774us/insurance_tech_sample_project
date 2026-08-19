terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
}

variable "gcp_project_id" {
  description = "GCP project ID to deploy the pipeline into."
  type        = string
}

variable "gcp_region" {
  description = "GCP region to deploy the pipeline into."
  type        = string
  default     = "us-central1"
}
