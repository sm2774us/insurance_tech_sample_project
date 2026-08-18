# AWS Infrastructure

Maps the CLUSTER MODE half of the architecture diagram onto AWS:

| Diagram component | AWS resource |
|---|---|
| Cloud Object Storage / S3-Compatible Ceph | `aws_s3_bucket.lake` (versioned, KMS-encrypted, public access blocked) — the `s3://` URI target for `pit.bitemporal.BitemporalStore` |
| Web Backend: FastAPI + Pydantic | `aws_ecs_cluster` + `aws_ecr_repository.api` (Fargate service, container-insights enabled) |
| AI Engines: Ray Train / Distributed PyTorch | `aws_batch_compute_environment.ray_train` (Fargate-backed, scales to zero between retraining runs) |

## Layout

- `provider.tf` — AWS provider + default resource tags.
- `variables.tf` / `outputs.tf` — shared stack inputs/outputs.
- `main.tf` — the resources above.

## Usage

```bash
terraform -chdir=infra/aws init
terraform -chdir=infra/aws plan -var="aws_region=us-east-1" -var="environment=dev"
terraform -chdir=infra/aws apply
```

State backend is intentionally left unconfigured here (see
`TERRAFORM_BEST_PRACTICES.md`); wire an `S3` + `DynamoDB` lock backend via
`-backend-config` per environment in CI rather than hardcoding it in source.
