# Azure Infrastructure

Maps the CLUSTER MODE half of the architecture diagram onto Azure:

| Diagram component | Azure resource |
|---|---|
| Cloud Object Storage | `azurerm_storage_account.lake` (ADLS Gen2 / HNS-enabled, GRS, versioned, TLS1.2-minimum) + `azurerm_storage_container.lake` |
| Web Backend: FastAPI + Pydantic | `azurerm_container_app.api` inside `azurerm_container_app_environment.api` |
| AI Engines: Ray Train / Distributed PyTorch | `azurerm_kubernetes_cluster.ray_train` (AKS, system-assigned identity) |

## Layout

- `provider.tf` — azurerm provider + region variable.
- `variables.tf` / `outputs.tf` — shared stack inputs/outputs.
- `main.tf` — the resources above (includes `azurerm_resource_group.pipeline`).

## Usage

```bash
terraform -chdir=infra/azure init
terraform -chdir=infra/azure plan -var="environment=dev"
terraform -chdir=infra/azure apply
```

Configure an Azure Storage-backed remote state via `-backend-config` per
environment in CI, per `TERRAFORM_BEST_PRACTICES.md`.
