"""FastAPI + Pydantic async web backend for the FIG quant pipeline.

Serves the JSON API under `/v1/*`. The Taipy GUI (`fig_quant.web.gui`) is
deliberately NOT mounted into this ASGI app: Taipy's frontend depends on a
live WebSocket (Flask-SocketIO) connection for state sync, and a plain WSGI
mount (e.g. `a2wsgi.WSGIMiddleware`) cannot proxy the WebSocket upgrade --
the handshake gets rejected (HTTP 403), the client falls back to nothing,
and the page renders blank. Taipy is run as its own native server instead
(see `fig_quant.web.server`), on a separate port, which is also what the
architecture diagram shows: FastAPI and Taipy are drawn as two distinct
boxes, not one merged process.
"""

from __future__ import annotations

import numpy as np
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from fig_quant.capital.bayesian_sizing import expected_severity_lcb, update_nig_posterior
from fig_quant.data.signal_validation import validate_features

api = FastAPI(title="FIG Quant Pod API", version="0.1.0")


class SignalValidationRequest(BaseModel):
    """Request body for the signal-validation endpoint."""

    feature_values: dict[str, list[float]] = Field(..., description="Candidate feature arrays.")
    target: list[float]
    controls: list[list[float]]
    groups: list[int]
    q: float = 0.10
    n_permutations: int = 500


class SignalValidationResponseItem(BaseModel):
    """One feature's validation outcome."""

    feature_name: str
    incremental_ic: float
    p_value: float
    rejected: bool


@api.post("/v1/signal-validation", response_model=list[SignalValidationResponseItem])
async def signal_validation(req: SignalValidationRequest) -> list[SignalValidationResponseItem]:
    """Runs the incremental IC / permutation / BHY validation harness."""
    features = {k: np.array(v) for k, v in req.feature_values.items()}
    target = np.array(req.target)
    controls = np.array(req.controls)
    groups = np.array(req.groups)
    results = validate_features(features, target, controls, groups, req.q, req.n_permutations)
    return [
        SignalValidationResponseItem(
            feature_name=r.feature_name,
            incremental_ic=r.incremental_ic,
            p_value=r.p_value,
            rejected=r.rejected,
        )
        for r in results
    ]


class SmallSampleSizingRequest(BaseModel):
    """Request body for the Bayesian small-sample sizing endpoint."""

    log_severities: list[float]
    confidence: float = 0.95


class SmallSampleSizingResponse(BaseModel):
    """Bayesian posterior LCB sizing outcome."""

    posterior_mu: float
    expected_severity_lcb: float


@api.post("/v1/small-sample-sizing", response_model=SmallSampleSizingResponse)
async def small_sample_sizing(req: SmallSampleSizingRequest) -> SmallSampleSizingResponse:
    """Computes the Bayesian posterior LCB severity estimate."""
    posterior = update_nig_posterior(np.array(req.log_severities))
    lcb = expected_severity_lcb(posterior, req.confidence)
    return SmallSampleSizingResponse(posterior_mu=posterior.mu, expected_severity_lcb=lcb)


@api.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@api.get("/")
async def root() -> RedirectResponse:
    """Redirects the API-root landing hit to the interactive docs.

    The human-facing Taipy console runs as a separate server (default
    port 8000 via `fig-quant serve`); this API server's own root simply
    points visitors at `/docs` rather than 404-ing or attempting to serve
    GUI assets it doesn't own.
    """
    return RedirectResponse(url="/docs")
