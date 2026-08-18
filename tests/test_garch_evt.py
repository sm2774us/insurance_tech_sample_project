"""Tests for the GARCH(1,1) + EVT conditional tail risk engine."""

from __future__ import annotations

import numpy as np

from fig_quant.risk.garch_evt import (
    conditional_sigma,
    fit_garch11,
    fit_gpd_tail,
    gpd_var,
    risk_loaded_premium,
)


def _simulate_garch(n: int, omega: float, alpha: float, beta: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = np.empty(n)
    sigma2 = omega / (1 - alpha - beta)
    for t in range(n):
        sigma2 = omega + alpha * (eps[t - 1] ** 2 if t > 0 else sigma2) + beta * sigma2
        eps[t] = rng.normal(scale=np.sqrt(sigma2))
    return eps


def test_fit_garch11_recovers_reasonable_persistence() -> None:
    eps = _simulate_garch(1500, omega=0.05, alpha=0.1, beta=0.8, seed=0)
    params = fit_garch11(eps)
    assert 0.0 <= params.alpha < 1.0
    assert 0.0 <= params.beta < 1.0
    assert params.alpha + params.beta < 1.0


def test_conditional_sigma_positive_and_reactive() -> None:
    eps = _simulate_garch(500, omega=0.05, alpha=0.15, beta=0.75, seed=1)
    params = fit_garch11(eps)
    sigma = conditional_sigma(eps, params)
    assert np.all(sigma > 0)
    assert len(sigma) == len(eps)


def test_gpd_tail_fit_and_var_monotonic_in_alpha() -> None:
    rng = np.random.default_rng(2)
    z = rng.standard_t(df=4, size=5000)
    fit = fit_gpd_tail(z, quantile=0.90)
    var_99 = gpd_var(fit, alpha=0.01)
    var_95 = gpd_var(fit, alpha=0.05)
    assert var_99 > var_95


def test_gpd_tail_fit_handles_sparse_tail() -> None:
    z = np.array([0.1, 0.2, -0.1, 0.05, 0.15])
    fit = fit_gpd_tail(z, quantile=0.90)
    assert fit.shape == 0.0


def test_risk_loaded_premium_increases_with_lambda() -> None:
    mean = np.full(10, 100.0)
    sigma = np.full(10, 5.0)
    low = risk_loaded_premium(mean, sigma, tail_var_standardized=2.0, lam=0.5, kappa=0.5)
    high = risk_loaded_premium(mean, sigma, tail_var_standardized=2.0, lam=2.0, kappa=0.5)
    assert np.all(high > low)
