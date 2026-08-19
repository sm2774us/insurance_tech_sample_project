"""Tests for the FastAPI web backend."""

from __future__ import annotations

from fastapi.testclient import TestClient

from fig_quant.web.app import api

client = TestClient(api)


def test_root_redirects_to_docs() -> None:
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code in (302, 307)
    assert resp.headers["location"] == "/docs"


def test_healthz() -> None:
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_signal_validation_endpoint() -> None:
    n = 300
    body = {
        "feature_values": {"f1": [float(i % 7) for i in range(n)]},
        "target": [float((i % 7) * 0.5 + (i % 3)) for i in range(n)],
        "controls": [[1.0, float(i % 3)] for i in range(n)],
        "groups": [i % 10 for i in range(n)],
        "n_permutations": 50,
    }
    resp = client.post("/v1/signal-validation", json=body)
    assert resp.status_code == 200
    payload = resp.json()
    assert payload[0]["feature_name"] == "f1"


def test_small_sample_sizing_endpoint() -> None:
    body = {"log_severities": [6.2, 6.5, 6.1, 6.4], "confidence": 0.95}
    resp = client.post("/v1/small-sample-sizing", json=body)
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["expected_severity_lcb"] > 0
