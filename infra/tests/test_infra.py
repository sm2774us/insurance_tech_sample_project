"""tftest-based validation of the per-cloud Terraform stacks.

Uses `terraform validate` via the tftest harness (no live provider
credentials required) to assert each stack's expected top-level resources
are registered in the configuration.
"""

from __future__ import annotations

import pathlib

import pytest
import tftest

_INFRA_ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def aws_plan() -> object:
    tf = tftest.TerraformTest(tfdir=str(_INFRA_ROOT / "aws"), binary="terraform")
    tf.setup(backend=False)
    return tf.plan(output=True, tf_vars={"environment": "test"})


@pytest.fixture(scope="module")
def gcp_plan() -> object:
    tf = tftest.TerraformTest(tfdir=str(_INFRA_ROOT / "gcp"), binary="terraform")
    tf.setup(backend=False)
    return tf.plan(output=True, tf_vars={"environment": "test", "gcp_project_id": "test-project"})


@pytest.fixture(scope="module")
def azure_plan() -> object:
    tf = tftest.TerraformTest(tfdir=str(_INFRA_ROOT / "azure"), binary="terraform")
    tf.setup(backend=False)
    return tf.plan(output=True, tf_vars={"environment": "test"})


def test_aws_s3_lake_bucket_registered(aws_plan) -> None:
    assert "aws_s3_bucket.lake" in aws_plan.modules["root"].resources


def test_aws_batch_compute_environment_registered(aws_plan) -> None:
    assert "aws_batch_compute_environment.ray_train" in aws_plan.modules["root"].resources


def test_gcp_storage_lake_bucket_registered(gcp_plan) -> None:
    assert "google_storage_bucket.lake" in gcp_plan.modules["root"].resources


def test_gcp_cloud_run_api_registered(gcp_plan) -> None:
    assert "google_cloud_run_v2_service.api" in gcp_plan.modules["root"].resources


def test_azure_storage_account_registered(azure_plan) -> None:
    assert "azurerm_storage_account.lake" in azure_plan.modules["root"].resources


def test_azure_aks_cluster_registered(azure_plan) -> None:
    assert "azurerm_kubernetes_cluster.ray_train" in azure_plan.modules["root"].resources
