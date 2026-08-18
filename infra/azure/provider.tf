terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
  }
}

provider "azurerm" {
  features {}
}

variable "azure_location" {
  description = "Azure region to deploy the pipeline into."
  type        = string
  default     = "eastus"
}
