# Infrastructure (Terraform)

Three independent, parallel cloud stacks (`aws/`, `gcp/`, `azure/`), each
implementing the CLUSTER MODE half of the architecture diagram in the top-level
README: object storage for the Arrow/Parquet data lake, a managed container
runtime for the FastAPI web backend, and an autoscaling Kubernetes/Batch pool for
Ray Train distributed PyTorch retraining. See each folder's own `README.md` for
the resource-by-resource mapping.

Validated in CI via `terraform fmt -check`, `terraform init -backend=false`, and
`terraform validate` per cloud (`.github/workflows/ci.yml` → `terraform-validate`
matrix job), plus a `tftest`-style pytest harness in `infra/tests/`.
