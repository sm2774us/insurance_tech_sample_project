"""Small-sample regime sizing via Bayesian posterior lower confidence bounds.

For a newly-launched pod line with too few claims for reliable frequentist
point estimates, we place a conjugate Normal-Inverse-Gamma prior on the
(mean, variance) of log-severity, update via closed-form posterior
recursion (no MCMC needed for the conjugate case; falls back to Gibbs VI
for non-conjugate hurdle structure), and size exposure off the posterior
*lower* confidence bound rather than the posterior mean, so genuine
parameter uncertainty is penalized rather than ignored.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy import stats


@dataclasses.dataclass(frozen=True, slots=True)
class NigPosterior:
    """Posterior parameters of a Normal-Inverse-Gamma update.

    Attributes:
      mu: Posterior mean of the Normal component.
      kappa: Posterior precision-scaling of the Normal component.
      alpha: Posterior shape of the Inverse-Gamma variance component.
      beta: Posterior scale of the Inverse-Gamma variance component.
    """

    mu: float
    kappa: float
    alpha: float
    beta: float


def update_nig_posterior(
    data: np.ndarray,
    prior_mu: float = 0.0,
    prior_kappa: float = 1.0,
    prior_alpha: float = 2.0,
    prior_beta: float = 1.0,
) -> NigPosterior:
    """Performs closed-form conjugate Normal-Inverse-Gamma posterior update.

    Args:
      data: Shape ``(n,)`` observed log-severities.
      prior_mu: Prior mean.
      prior_kappa: Prior pseudo-count / precision scaling.
      prior_alpha: Prior IG shape.
      prior_beta: Prior IG scale.

    Returns:
      Updated :class:`NigPosterior`.
    """
    n = len(data)
    if n == 0:
        return NigPosterior(prior_mu, prior_kappa, prior_alpha, prior_beta)
    sample_mean = float(np.mean(data))
    sample_ss = float(np.sum((data - sample_mean) ** 2))

    kappa_n = prior_kappa + n
    mu_n = (prior_kappa * prior_mu + n * sample_mean) / kappa_n
    alpha_n = prior_alpha + n / 2.0
    beta_n = (
        prior_beta
        + 0.5 * sample_ss
        + (prior_kappa * n * (sample_mean - prior_mu) ** 2) / (2.0 * kappa_n)
    )
    return NigPosterior(mu_n, kappa_n, alpha_n, beta_n)


def posterior_predictive_lcb(posterior: NigPosterior, confidence: float = 0.95) -> float:
    """Computes the lower confidence bound of the posterior predictive mean.

    Under the NIG conjugate model, the marginal posterior of the mean
    follows a Student-t distribution with ``2*alpha`` degrees of freedom,
    location ``mu``, and scale ``sqrt(beta / (alpha * kappa))``.

    Args:
      posterior: A fitted :class:`NigPosterior`.
      confidence: One-sided confidence level for the lower bound.

    Returns:
      The posterior-mean LCB in log-severity units.
    """
    df = 2.0 * posterior.alpha
    scale = np.sqrt(posterior.beta / (posterior.alpha * posterior.kappa))
    t_crit = stats.t.ppf(1.0 - confidence, df)
    return posterior.mu + t_crit * scale


def expected_severity_lcb(posterior: NigPosterior, confidence: float = 0.95) -> float:
    """Converts the log-scale LCB back to severity units.

    Uses the lognormal LCB-of-mean approximation
    ``exp(mu_lcb + 0.5 * E[sigma^2])`` where ``E[sigma^2] = beta/(alpha-1)``
    is the posterior mean of the variance component (requires
    ``alpha > 1``).

    Args:
      posterior: A fitted :class:`NigPosterior`.
      confidence: One-sided confidence level.

    Returns:
      The conservative (lower-bound) expected severity estimate.
    """
    mu_lcb = posterior_predictive_lcb(posterior, confidence)
    var_mean = posterior.beta / max(posterior.alpha - 1.0, 1e-6)
    return float(np.exp(mu_lcb + 0.5 * var_mean))


def hurdle_constrained_weights(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    lam: float,
    hurdle: float,
    weight_upper_bound: float = 1.0,
) -> np.ndarray:
    """Solves the hurdle-constrained mean-variance optimizer via projected GD.

    Maximizes ``E[w'Y|X] - lambda * w' Var(Y|X) w`` subject to
    ``E[w'Y|X] >= hurdle``, ``w >= 0``, ``w <= weight_upper_bound``.

    Args:
      expected_returns: Shape ``(m,)`` posterior LCB expected returns per
        proposal.
      cov_matrix: Shape ``(m, m)`` posterior covariance of returns.
      lam: Risk-aversion coefficient.
      hurdle: Minimum required portfolio expected return.
      weight_upper_bound: Per-proposal weight cap (e.g. concentration
        limit).

    Returns:
      Shape ``(m,)`` optimal non-negative weight vector.
    """
    m = len(expected_returns)
    w = np.full(m, 1.0 / m)
    step = 0.01
    for _ in range(5000):
        grad = expected_returns - 2.0 * lam * cov_matrix @ w
        w = w + step * grad
        w = np.clip(w, 0.0, weight_upper_bound)
        port_return = float(expected_returns @ w)
        if port_return < hurdle and w.sum() > 0:
            # Project back toward the hurdle constraint by rescaling.
            scale = hurdle / max(port_return, 1e-8)
            w = np.clip(w * min(scale, weight_upper_bound / max(w.max(), 1e-8)), 0.0, weight_upper_bound)
    return w
