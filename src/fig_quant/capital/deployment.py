"""Capital deployment pipeline: falsification gate + quadratic allocator.

Underwriting proposal ``m`` is first subjected to an out-of-sample
falsification test producing a test statistic ``T_m``; only proposals
clearing the validation gate (``T_m >= tau_gate``) enter the quadratic
capital allocator, which maximizes ``Return - lambda * Risk`` subject to
``sum(K_m) <= K_total``. Rejected proposals receive zero capital by
construction, not by post-hoc override.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy import optimize


@dataclasses.dataclass(frozen=True, slots=True)
class UnderwritingProposal:
    """A single pod underwriting proposal awaiting a capital decision.

    Attributes:
      proposal_id: Unique identifier.
      test_statistic: Out-of-sample falsification test statistic ``T_m``.
      expected_return: Posterior/backtested expected return per unit
        capital.
      risk_contribution: Marginal variance contribution per unit capital
        (diagonal of the risk model, or a full covariance row is passed
        separately to :func:`quadratic_capital_allocation`).
    """

    proposal_id: str
    test_statistic: float
    expected_return: float
    risk_contribution: float


def validation_gate(
    proposals: list[UnderwritingProposal], tau_gate: float
) -> list[UnderwritingProposal]:
    """Filters proposals to those clearing the falsification test gate.

    Args:
      proposals: Candidate underwriting proposals.
      tau_gate: Minimum required test statistic.

    Returns:
      The subset of proposals with ``test_statistic >= tau_gate``.
    """
    return [p for p in proposals if p.test_statistic >= tau_gate]


def quadratic_capital_allocation(
    proposals: list[UnderwritingProposal],
    cov_matrix: np.ndarray,
    lam: float,
    k_total: float,
) -> dict[str, float]:
    """Solves ``max Return'K - lambda * K'Cov*K`` s.t. ``sum(K) <= K_total``.

    Proposals that failed :func:`validation_gate` should already have been
    excluded from ``proposals``; anything not passed in receives implicit
    zero capital.

    Args:
      proposals: Gate-passing underwriting proposals.
      cov_matrix: Shape ``(m, m)`` return covariance across the passed
        proposals, in the same order as ``proposals``.
      lam: Risk-aversion coefficient.
      k_total: Total capital available for deployment.

    Returns:
      Mapping of ``proposal_id`` to allocated capital ``K_m >= 0``.
    """
    if not proposals:
        return {}
    m = len(proposals)
    returns = np.array([p.expected_return for p in proposals])

    def neg_objective(k: np.ndarray) -> float:
        return -(float(returns @ k) - lam * float(k @ cov_matrix @ k))

    constraints = [{"type": "ineq", "fun": lambda k: k_total - np.sum(k)}]
    bounds = [(0.0, k_total) for _ in range(m)]
    x0 = np.full(m, k_total / (2 * m))
    result = optimize.minimize(
        neg_objective, x0, method="SLSQP", bounds=bounds, constraints=constraints
    )
    allocation = np.clip(result.x, 0.0, None)
    return {p.proposal_id: float(k) for p, k in zip(proposals, allocation)}


def deploy_capital(
    all_proposals: list[UnderwritingProposal],
    cov_matrix_by_id: dict[tuple[str, str], float],
    tau_gate: float,
    lam: float,
    k_total: float,
) -> dict[str, float]:
    """Runs the full gate -> quadratic-allocator capital deployment pipeline.

    Args:
      all_proposals: Every proposal under consideration this cycle.
      cov_matrix_by_id: Pairwise return covariance keyed by
        ``(proposal_id, proposal_id)``, including diagonal variances.
      tau_gate: Validation gate threshold.
      lam: Risk-aversion coefficient for the allocator.
      k_total: Total capital available.

    Returns:
      Mapping of every ``proposal_id`` in ``all_proposals`` to its
      allocated capital; rejected proposals map to ``0.0``.
    """
    passed = validation_gate(all_proposals, tau_gate)
    if not passed:
        return {p.proposal_id: 0.0 for p in all_proposals}
    ids = [p.proposal_id for p in passed]
    cov = np.array(
        [[cov_matrix_by_id[(i, j)] for j in ids] for i in ids]
    )
    allocated = quadratic_capital_allocation(passed, cov, lam, k_total)
    return {
        p.proposal_id: allocated.get(p.proposal_id, 0.0) for p in all_proposals
    }
