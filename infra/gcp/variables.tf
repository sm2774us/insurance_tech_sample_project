variable "environment" {
  description = "Deployment environment name (e.g. dev, staging, prod)."
  type        = string
  default     = "dev"
}

variable "project_prefix" {
  description = "Naming prefix applied to all resources for this stack."
  type        = string
  default     = "fig-quant"
}
