# GCP Infrastructure

Maps the CLUSTER MODE half of the architecture diagram onto GCP:

| Diagram component | GCP resource |
|---|---|
| Cloud Object Storage | `google_storage_bucket.lake` (versioned, CMEK-encrypted via `google_kms_crypto_key.lake`, uniform bucket-level access) |
| Web Backend: FastAPI + Pydantic | `google_cloud_run_v2_service.api` |
| AI Engines: Ray Train / Distributed PyTorch | `google_container_cluster.ray_train` + autoscaling `google_container_node_pool` (0-8 nodes, scales to zero) |

## Layout

- `provider.tf` — Google provider + project/region variables.
- `variables.tf` / `outputs.tf` — shared stack inputs/outputs.
- `main.tf` — the resources above.

## Usage

```bash
terraform -chdir=infra/gcp init
terraform -chdir=infra/gcp plan -var="gcp_project_id=my-project" -var="environment=dev"
terraform -chdir=infra/gcp apply
```

Configure a GCS-backed remote state via `-backend-config` per environment in
CI, per `TERRAFORM_BEST_PRACTICES.md`.
