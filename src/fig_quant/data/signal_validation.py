"""Incremental signal validation: residual IC, permutation nulls, BHY FDR.

Reusable across any alt-data feed the pod evaluates. A candidate feature
only earns production status if it (a) carries information incremental to
the incumbent control set and (b) survives a permutation-null test at a
false-discovery-rate-controlled threshold valid under arbitrary dependence
across simultaneously tested features.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy import stats


@dataclasses.dataclass(frozen=True, slots=True)
class ValidationResult:
    """Outcome of validating one candidate feature.

    Attributes:
        feature_name: Name of the candidate feature.
        incremental_ic: Spearman IC of the feature residual against the
          target residual, both orthogonalized to the control matrix.
        p_value: Permutation-null two-sided p-value for ``incremental_ic``.
        bhy_threshold: The BHY-adjusted critical p-value this feature was
          compared against.
        rejected: Whether the feature clears the BHY FDR screen (i.e. is
          judged to carry genuine incremental signal).
    """

    feature_name: str
    incremental_ic: float
    p_value: float
    bhy_threshold: float
    rejected: bool


def _residualize(target: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Returns OLS residuals of ``target`` regressed on ``controls``.

    Args:
      target: Shape ``(n,)`` response vector.
      controls: Shape ``(n, k)`` design matrix (should include an
        intercept column if desired).

    Returns:
      Shape ``(n,)`` residual vector ``target - controls @ beta_hat``.
    """
    beta_hat, *_ = np.linalg.lstsq(controls, target, rcond=None)
    return target - controls @ beta_hat


def incremental_ic(
    feature: np.ndarray, target: np.ndarray, controls: np.ndarray
) -> float:
    """Computes the incremental Spearman IC of ``feature`` on ``target``.

    Both series are first orthogonalized against ``controls`` via OLS; the
    Spearman correlation of the two residual series isolates the marginal
    predictive contribution of ``feature`` beyond the incumbent controls.

    Args:
      feature: Shape ``(n,)`` candidate feature values.
      target: Shape ``(n,)`` target values.
      controls: Shape ``(n, k)`` control design matrix.

    Returns:
      The incremental information coefficient in ``[-1, 1]``.
    """
    resid_feature = _residualize(feature, controls)
    resid_target = _residualize(target, controls)
    ic, _ = stats.spearmanr(resid_feature, resid_target)
    return float(ic)


def permutation_p_value(
    feature: np.ndarray,
    target: np.ndarray,
    controls: np.ndarray,
    groups: np.ndarray,
    n_permutations: int = 2000,
    random_state: int = 0,
) -> tuple[float, float]:
    """Computes the observed IC and a within-group permutation p-value.

    Args:
      feature: Shape ``(n,)`` candidate feature values.
      target: Shape ``(n,)`` target values.
      controls: Shape ``(n, k)`` control design matrix.
      groups: Shape ``(n,)`` grouping key (e.g. county) within which the
        feature is shuffled, preserving cross-sectional structure and
        avoiding spurious inflation from breaking group-level clustering.
      n_permutations: Number of null draws.
      random_state: Seed for reproducibility.

    Returns:
      A tuple ``(observed_ic, p_value)`` where ``p_value`` follows the
      add-one-smoothed formula
      ``(1 + #{|IC_null| >= |IC_obs|}) / (1 + n_permutations)``.
    """
    rng = np.random.default_rng(random_state)
    observed = incremental_ic(feature, target, controls)
    null_ics = np.empty(n_permutations)
    unique_groups = np.unique(groups)
    for i in range(n_permutations):
        shuffled = feature.copy()
        for grp in unique_groups:
            mask = groups == grp
            shuffled[mask] = rng.permutation(shuffled[mask])
        null_ics[i] = incremental_ic(shuffled, target, controls)
    exceed = int(np.sum(np.abs(null_ics) >= abs(observed)))
    p_value = (1 + exceed) / (1 + n_permutations)
    return observed, p_value


def benjamini_hochberg_yekutieli(p_values: dict[str, float], q: float = 0.10) -> dict[str, bool]:
    """Applies the BHY procedure, valid under arbitrary dependence.

    Args:
      p_values: Mapping of feature name to permutation p-value.
      q: Target false discovery rate.

    Returns:
      Mapping of feature name to whether its null hypothesis is rejected
      (i.e. the feature is judged to carry real signal).
    """
    names = list(p_values.keys())
    pvals = np.array([p_values[n] for n in names])
    m = len(pvals)
    order = np.argsort(pvals)
    c_m = np.sum(1.0 / np.arange(1, m + 1))
    thresholds = (np.arange(1, m + 1) / (m * c_m)) * q
    sorted_p = pvals[order]
    passed = sorted_p <= thresholds
    # Largest index i such that all j <= i pass determines the rejection set.
    if np.any(passed):
        max_i = np.max(np.nonzero(passed)[0])
        reject_mask = np.zeros(m, dtype=bool)
        reject_mask[: max_i + 1] = True
    else:
        reject_mask = np.zeros(m, dtype=bool)
    result = {names[order[i]]: bool(reject_mask[i]) for i in range(m)}
    return result


def validate_features(
    features: dict[str, np.ndarray],
    target: np.ndarray,
    controls: np.ndarray,
    groups: np.ndarray,
    q: float = 0.10,
    n_permutations: int = 2000,
    random_state: int = 0,
) -> list[ValidationResult]:
    """Runs the full incremental-IC -> permutation -> BHY pipeline.

    Args:
      features: Mapping of candidate feature name to its value array.
      target: Shape ``(n,)`` target values.
      controls: Shape ``(n, k)`` control design matrix.
      groups: Shape ``(n,)`` grouping key for permutation blocking.
      q: Target false discovery rate for the BHY screen.
      n_permutations: Number of permutation draws per feature.
      random_state: Base seed; offset per feature for independence.

    Returns:
      One :class:`ValidationResult` per candidate feature.
    """
    ics: dict[str, float] = {}
    p_values: dict[str, float] = {}
    for offset, (name, values) in enumerate(features.items()):
        ic, p = permutation_p_value(
            values, target, controls, groups, n_permutations, random_state + offset
        )
        ics[name] = ic
        p_values[name] = p
    rejections = benjamini_hochberg_yekutieli(p_values, q)
    names_sorted_by_p = sorted(p_values, key=lambda n: p_values[n])
    m = len(names_sorted_by_p)
    c_m = np.sum(1.0 / np.arange(1, m + 1)) if m else 1.0
    rank = {n: i + 1 for i, n in enumerate(names_sorted_by_p)}
    return [
        ValidationResult(
            feature_name=name,
            incremental_ic=ics[name],
            p_value=p_values[name],
            bhy_threshold=(rank[name] / (m * c_m)) * q if m else 0.0,
            rejected=rejections[name],
        )
        for name in features
    ]
