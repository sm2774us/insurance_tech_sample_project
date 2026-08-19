"""Conditional tail risk: GARCH(1,1) volatility filter + EVT tail VaR.

Pipeline: conditional-mean estimator residuals -> GARCH(1,1) conditional
variance -> generalized Pareto tail fit on standardized residual excesses
above a high threshold -> risk-loaded premium
``Pi_t = E[L|X] + lambda*sigma_t + kappa*VaR_alpha``.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy import optimize, stats


@dataclasses.dataclass(frozen=True, slots=True)
class Garch11Params:
    """Fitted GARCH(1,1) parameters.

    Attributes:
      omega: Constant term (unconditional variance contribution).
      alpha: ARCH coefficient (reaction to the prior squared shock).
      beta: GARCH coefficient (persistence of prior conditional variance).
    """

    omega: float
    alpha: float
    beta: float


def fit_garch11(residuals: np.ndarray) -> Garch11Params:
    """Fits GARCH(1,1) by maximum likelihood on mean-zero residuals.

    ``sigma_t^2 = omega + alpha * eps_{t-1}^2 + beta * sigma_{t-1}^2``

    Args:
      residuals: Shape ``(n,)`` conditional-mean residuals ``eps_t``.

    Returns:
      Fitted :class:`Garch11Params`.
    """
    eps = np.asarray(residuals, dtype=np.float64)
    var_unconditional = float(np.var(eps)) + 1e-12

    def neg_log_likelihood(theta: np.ndarray) -> float:
        omega, alpha, beta = theta
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
            return 1e10
        sigma2 = np.empty_like(eps)
        sigma2[0] = var_unconditional
        for t in range(1, len(eps)):
            sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        sigma2 = np.clip(sigma2, 1e-12, None)
        ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + eps**2 / sigma2)
        return -float(ll)

    x0 = np.array([0.1 * var_unconditional, 0.05, 0.9])
    result = optimize.minimize(
        neg_log_likelihood,
        x0,
        method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 2000},
    )
    omega, alpha, beta = result.x
    return Garch11Params(omega=float(omega), alpha=float(alpha), beta=float(beta))


def conditional_sigma(residuals: np.ndarray, params: Garch11Params) -> np.ndarray:
    """Computes the filtered conditional-volatility path ``sigma_t``.

    Args:
      residuals: Shape ``(n,)`` conditional-mean residuals.
      params: Fitted GARCH(1,1) parameters.

    Returns:
      Shape ``(n,)`` conditional standard deviation path.
    """
    eps = np.asarray(residuals, dtype=np.float64)
    sigma2 = np.empty_like(eps)
    sigma2[0] = float(np.var(eps)) + 1e-12
    for t in range(1, len(eps)):
        sigma2[t] = params.omega + params.alpha * eps[t - 1] ** 2 + params.beta * sigma2[t - 1]
    return np.sqrt(np.clip(sigma2, 1e-12, None))


@dataclasses.dataclass(frozen=True, slots=True)
class GpdTailFit:
    """Generalized Pareto Distribution fit over a high threshold.

    Attributes:
      threshold: The excess-over-threshold cutoff ``u``.
      shape: GPD shape parameter ``xi``.
      scale: GPD scale parameter ``beta``.
      exceedance_rate: Empirical probability of exceeding ``threshold``.
    """

    threshold: float
    shape: float
    scale: float
    exceedance_rate: float


def fit_gpd_tail(standardized_residuals: np.ndarray, quantile: float = 0.90) -> GpdTailFit:
    """Fits a GPD to standardized-residual excesses above ``quantile``.

    Args:
      standardized_residuals: Shape ``(n,)`` residuals divided by their
        conditional GARCH volatility (i.e. approximately i.i.d.).
      quantile: Threshold quantile defining "extreme" excesses.

    Returns:
      A :class:`GpdTailFit` describing the fitted tail law.
    """
    z = np.asarray(standardized_residuals, dtype=np.float64)
    threshold = float(np.quantile(z, quantile))
    excesses = z[z > threshold] - threshold
    exceedance_rate = float(len(excesses) / len(z))
    if len(excesses) < 10:
        # Insufficient tail data: fall back to a conservative exponential
        # tail (shape=0) rather than an unstable GPD MLE.
        scale = float(np.mean(excesses)) if len(excesses) else 1e-6
        return GpdTailFit(threshold, 0.0, max(scale, 1e-6), exceedance_rate)
    shape, _loc, scale = stats.genpareto.fit(excesses, floc=0)
    return GpdTailFit(threshold, float(shape), float(max(scale, 1e-6)), exceedance_rate)


def gpd_var(fit: GpdTailFit, alpha: float) -> float:
    """Computes the EVT tail Value-at-Risk at level ``alpha``.

    Args:
      fit: A :class:`GpdTailFit`.
      alpha: Tail probability (e.g. ``0.01`` for the 99th percentile VaR).

    Returns:
      The standardized-residual VaR quantile.
    """
    xi, beta, u, zeta = fit.shape, fit.scale, fit.threshold, fit.exceedance_rate
    ratio = alpha / zeta
    if abs(xi) < 1e-8:
        return u - beta * np.log(ratio)
    return u + (beta / xi) * (ratio ** (-xi) - 1.0)


def risk_loaded_premium(
    conditional_mean: np.ndarray,
    sigma_t: np.ndarray,
    tail_var_standardized: float,
    lam: float = 1.0,
    kappa: float = 0.5,
) -> np.ndarray:
    """Computes ``Pi_t = E[L|X] + lambda*sigma_t + kappa*VaR_alpha*sigma_t``.

    Args:
      conditional_mean: Shape ``(n,)`` conditional expected loss ``E[L_t|X_t]``.
      sigma_t: Shape ``(n,)`` GARCH conditional volatility path.
      tail_var_standardized: Standardized-residual EVT VaR quantile.
      lam: Volatility risk-load coefficient.
      kappa: Tail risk-load coefficient.

    Returns:
      Shape ``(n,)`` risk-loaded premium path.
    """
    return conditional_mean + lam * sigma_t + kappa * tail_var_standardized * sigma_t
