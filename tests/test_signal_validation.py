"""Tests for the signal-validation harness."""

from __future__ import annotations

import numpy as np

from fig_quant.data.signal_validation import (
    benjamini_hochberg_yekutieli,
    incremental_ic,
    permutation_p_value,
    validate_features,
)


def test_incremental_ic_detects_true_signal() -> None:
    rng = np.random.default_rng(1)
    n = 3000
    controls = np.column_stack([np.ones(n), rng.normal(size=n)])
    feature = rng.normal(size=n)
    target = 0.6 * feature + 0.2 * controls[:, 1] + rng.normal(scale=0.3, size=n)
    ic = incremental_ic(feature, target, controls)
    assert ic > 0.3


def test_incremental_ic_near_zero_for_pure_noise() -> None:
    rng = np.random.default_rng(2)
    n = 3000
    controls = np.column_stack([np.ones(n), rng.normal(size=n)])
    feature = rng.normal(size=n)
    target = 0.2 * controls[:, 1] + rng.normal(scale=0.5, size=n)
    ic = incremental_ic(feature, target, controls)
    assert abs(ic) < 0.1


def test_permutation_p_value_range() -> None:
    rng = np.random.default_rng(3)
    n = 500
    controls = np.column_stack([np.ones(n), rng.normal(size=n)])
    feature = rng.normal(size=n)
    target = rng.normal(size=n)
    groups = rng.integers(0, 20, size=n)
    _, p = permutation_p_value(feature, target, controls, groups, n_permutations=100)
    assert 0.0 <= p <= 1.0


def test_bhy_rejects_only_signal_feature() -> None:
    p_values = {"signal": 0.001, "noise_a": 0.8, "noise_b": 0.65}
    rejections = benjamini_hochberg_yekutieli(p_values, q=0.10)
    assert rejections["signal"] is True
    assert rejections["noise_a"] is False


def test_validate_features_end_to_end() -> None:
    rng = np.random.default_rng(4)
    n = 1500
    groups = rng.integers(0, 30, size=n)
    controls = np.column_stack([np.ones(n), rng.normal(size=n)])
    signal_feature = rng.normal(size=n)
    target = 0.5 * signal_feature + rng.normal(scale=0.4, size=n)
    noise_feature = rng.normal(size=n)
    results = validate_features(
        {"signal_feature": signal_feature, "noise_feature": noise_feature},
        target,
        controls,
        groups,
        n_permutations=200,
    )
    by_name = {r.feature_name: r for r in results}
    assert by_name["signal_feature"].rejected is True
    assert by_name["noise_feature"].rejected is False
