resource "azurerm_resource_group" "pipeline" {
  name     = "${var.project_prefix}-${var.environment}-rg"
  location = var.azure_location
}

# Cluster-mode object storage backing the Data Access Layer's URI-swapped
# storage path (ADLS Gen2, hierarchical namespace enabled for Arrow/Iceberg
# partition-aware reads).
resource "azurerm_storage_account" "lake" {
  name                     = replace("${var.project_prefix}${var.environment}lake", "-", "")
  resource_group_name      = azurerm_resource_group.pipeline.name
  location                 = azurerm_resource_group.pipeline.location
  account_tier             = "Standard"
  account_replication_type = "GRS"
  is_hns_enabled            = true
  min_tls_version           = "TLS1_2"

  blob_properties {
    versioning_enabled = true
  }
}

resource "azurerm_storage_container" "lake" {
  name                  = "lake"
  storage_account_name  = azurerm_storage_account.lake.name
  container_access_type = "private"
}

# Container Apps environment hosting the FastAPI web backend + Taipy frontend.
resource "azurerm_container_app_environment" "api" {
  name                       = "${var.project_prefix}-${var.environment}-env"
  resource_group_name        = azurerm_resource_group.pipeline.name
  location                   = azurerm_resource_group.pipeline.location
}

resource "azurerm_container_app" "api" {
  name                         = "${var.project_prefix}-${var.environment}-api"
  container_app_environment_id = azurerm_container_app_environment.api.id
  resource_group_name          = azurerm_resource_group.pipeline.name
  revision_mode                 = "Single"

  template {
    container {
      name   = "api"
      image  = var.api_container_image
      cpu    = 2.0
      memory = "4Gi"
    }
  }
}

# AKS node pool for Ray Train / distributed PyTorch retraining jobs.
resource "azurerm_kubernetes_cluster" "ray_train" {
  name                = "${var.project_prefix}-${var.environment}-aks"
  location            = azurerm_resource_group.pipeline.location
  resource_group_name = azurerm_resource_group.pipeline.name
  dns_prefix          = "${var.project_prefix}${var.environment}"

  default_node_pool {
    name       = "system"
    node_count = 1
    vm_size    = "Standard_D4s_v5"
  }

  identity {
    type = "SystemAssigned"
  }
}

variable "api_container_image" {
  description = "Fully qualified container image for the FastAPI backend."
  type        = string
  default     = "myregistry.azurecr.io/fig-quant-api:latest"
}
