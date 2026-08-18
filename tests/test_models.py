"""Tests for the CANN pricing model and CPCV splitter."""

from __future__ import annotations

import numpy as np

from fig_quant.models.cann import CANNRegressor, TwoLayerResidualNet
from fig_quant.models.cpcv import cpcv_year_splits


def test_two_layer_net_loss_decreases() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 4))
    y = np.sum(x[:, :2] ** 2, axis=1) * 0.1 + rng.normal(scale=0.01, size=200)
    net = TwoLayerResidualNet(input_dim=4, hidden_dim=8, learning_rate=0.02, seed=1)
    history = net.fit(x, y, epochs=300)
    assert history[-1] < history[0]


def test_cann_predict_positive_and_beats_naive_mean() -> None:
    rng = np.random.default_rng(7)
    n = 400
    x = rng.uniform(0.1, 2.0, size=(n, 3))
    true_severity = np.exp(0.5 * x[:, 0] + 0.3 * x[:, 1]) + rng.gamma(2.0, 100.0, size=n)
    model = CANNRegressor(hidden_dim=8, seed=2).fit(x, true_severity, epochs=200)
    preds = model.predict(x)
    assert np.all(preds > 0)
    mae_model = np.mean(np.abs(preds - true_severity))
    mae_naive = np.mean(np.abs(np.mean(true_severity) - true_severity))
    assert mae_model < mae_naive


def test_cpcv_year_splits_purges_embargo() -> None:
    folds = cpcv_year_splits([2021, 2022, 2023, 2024], n_test_groups=2, embargo=1)
    assert len(folds) == 6
    for fold in folds:
        for test_year in fold.test_years:
            for train_year in fold.train_years:
                assert abs(train_year - test_year) > 1


def test_cpcv_no_embargo_includes_all_non_test_years() -> None:
    folds = cpcv_year_splits([2021, 2022, 2023], n_test_groups=1, embargo=0)
    assert len(folds) == 3
    for fold in folds:
        assert len(fold.train_years) == 2
