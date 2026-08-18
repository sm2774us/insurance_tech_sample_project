"""Tests for Bayesian small-sample sizing and the capital deployment pipeline."""

from __future__ import annotations

import numpy as np

from fig_quant.capital.bayesian_sizing import (
    expected_severity_lcb,
    hurdle_constrained_weights,
    posterior_predictive_lcb,
    update_nig_posterior,
)
from fig_quant.capital.deployment import (
    UnderwritingProposal,
    deploy_capital,
    quadratic_capital_allocation,
    validation_gate,
)


def test_nig_posterior_shrinks_toward_prior_with_no_data() -> None:
    posterior = update_nig_posterior(np.array([]), prior_mu=1.0, prior_kappa=2.0)
    assert posterior.mu == 1.0


def test_nig_posterior_lcb_below_mean() -> None:
    data = np.array([1.0, 1.2, 0.9, 1.1, 1.05])
    posterior = update_nig_posterior(data)
    lcb = posterior_predictive_lcb(posterior, confidence=0.95)
    assert lcb < posterior.mu


def test_expected_severity_lcb_positive() -> None:
    data = np.log(np.array([500.0, 620.0, 480.0, 700.0]))
    posterior = update_nig_posterior(data)
    lcb = expected_severity_lcb(posterior)
    assert lcb > 0


def test_hurdle_constrained_weights_nonnegative_and_bounded() -> None:
    returns = np.array([0.1, 0.05, 0.08])
    cov = np.eye(3) * 0.01
    weights = hurdle_constrained_weights(returns, cov, lam=1.0, hurdle=0.02, weight_upper_bound=1.0)
    assert np.all(weights >= 0.0)
    assert np.all(weights <= 1.0)


def test_validation_gate_filters_below_threshold() -> None:
    proposals = [
        UnderwritingProposal("A", test_statistic=3.0, expected_return=0.1, risk_contribution=0.01),
        UnderwritingProposal("B", test_statistic=1.0, expected_return=0.2, risk_contribution=0.02),
    ]
    passed = validation_gate(proposals, tau_gate=2.0)
    assert [p.proposal_id for p in passed] == ["A"]


def test_quadratic_allocation_respects_budget() -> None:
    proposals = [
        UnderwritingProposal("A", test_statistic=3.0, expected_return=0.12, risk_contribution=0.01),
        UnderwritingProposal("B", test_statistic=3.0, expected_return=0.08, risk_contribution=0.02),
    ]
    cov = np.array([[0.02, 0.005], [0.005, 0.03]])
    allocation = quadratic_capital_allocation(proposals, cov, lam=5.0, k_total=100.0)
    assert sum(allocation.values()) <= 100.0 + 1e-6
    assert all(v >= 0.0 for v in allocation.values())


def test_deploy_capital_zero_for_rejected_and_end_to_end() -> None:
    proposals = [
        UnderwritingProposal("A", test_statistic=5.0, expected_return=0.1, risk_contribution=0.01),
        UnderwritingProposal("B", test_statistic=0.5, expected_return=0.3, risk_contribution=0.01),
    ]
    cov_by_id = {("A", "A"): 0.01}
    result = deploy_capital(proposals, cov_by_id, tau_gate=2.0, lam=1.0, k_total=50.0)
    assert result["B"] == 0.0
    assert result["A"] >= 0.0
