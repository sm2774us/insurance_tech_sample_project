"""Combined Actuarial Neural Network (CANN) severity pricing model.

Structure: an auditable GLM (Poisson/Gamma log-link) floor multiplied by a
bounded neural residual adjustment, trained on log-residuals of the GLM.
This mirrors an interpretable-core / nonlinear-edge philosophy: the GLM
remains regulator-facing and fully explainable, while the residual network
captures nonlinear interactions the GLM misses, without ever being able to
overturn the GLM's sign or dominate its magnitude unboundedly (residual is
exponentiated, so it acts as a multiplicative correction).
"""

from __future__ import annotations

import dataclasses

import numpy as np
from sklearn.linear_model import PoissonRegressor


@dataclasses.dataclass(slots=True)
class TwoLayerResidualNet:
    """A manually-implemented two-layer MLP trained via full-batch GD.

    Architecture: Linear -> ReLU -> Linear -> ReLU -> Linear(1). Weights
    use He initialization; training uses L2-regularized MSE loss on
    standardized inputs and log-residual targets.

    Attributes:
      input_dim: Number of input features.
      hidden_dim: Width of each hidden layer.
      l2: L2 weight-decay coefficient.
      learning_rate: Gradient-descent step size.
      seed: RNG seed for weight initialization.
    """

    input_dim: int
    hidden_dim: int = 16
    l2: float = 1e-3
    learning_rate: float = 0.05
    seed: int = 0
    w1: np.ndarray = dataclasses.field(init=False, default=None)
    b1: np.ndarray = dataclasses.field(init=False, default=None)
    w2: np.ndarray = dataclasses.field(init=False, default=None)
    b2: np.ndarray = dataclasses.field(init=False, default=None)
    w3: np.ndarray = dataclasses.field(init=False, default=None)
    b3: np.ndarray = dataclasses.field(init=False, default=None)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)

        def he_init(fan_in: int, fan_out: int) -> np.ndarray:
            return rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))

        self.w1 = he_init(self.input_dim, self.hidden_dim)
        self.b1 = np.zeros(self.hidden_dim)
        self.w2 = he_init(self.hidden_dim, self.hidden_dim)
        self.b2 = np.zeros(self.hidden_dim)
        self.w3 = he_init(self.hidden_dim, 1)
        self.b3 = np.zeros(1)

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, ...]:
        z1 = x @ self.w1 + self.b1
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ self.w2 + self.b2
        a2 = np.maximum(z2, 0.0)
        out = a2 @ self.w3 + self.b3
        return z1, a1, z2, a2, out

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Returns the raw (pre-exponential) residual output for ``x``."""
        *_, out = self._forward(x)
        return out.ravel()

    def fit(self, x: np.ndarray, residual: np.ndarray, epochs: int = 500) -> list[float]:
        """Trains the network via full-batch gradient descent.

        Args:
          x: Shape ``(n, input_dim)`` standardized feature matrix.
          residual: Shape ``(n,)`` log-residual targets
            (``log(y) - log(glm_pred)``).
          epochs: Number of full-batch gradient steps.

        Returns:
          Per-epoch MSE loss history.
        """
        n = x.shape[0]
        r = residual.reshape(-1, 1)
        history: list[float] = []
        for _ in range(epochs):
            z1, a1, z2, a2, out = self._forward(x)
            loss = float(np.mean((out - r) ** 2))
            history.append(loss)

            d_out = (2.0 * (out - r)) / n
            grad_w3 = a2.T @ d_out + self.l2 * self.w3
            grad_b3 = d_out.sum(axis=0)

            d_a2 = d_out @ self.w3.T
            d_z2 = d_a2 * (z2 > 0)
            grad_w2 = a1.T @ d_z2 + self.l2 * self.w2
            grad_b2 = d_z2.sum(axis=0)

            d_a1 = d_z2 @ self.w2.T
            d_z1 = d_a1 * (z1 > 0)
            grad_w1 = x.T @ d_z1 + self.l2 * self.w1
            grad_b1 = d_z1.sum(axis=0)

            self.w3 -= self.learning_rate * grad_w3
            self.b3 -= self.learning_rate * grad_b3
            self.w2 -= self.learning_rate * grad_w2
            self.b2 -= self.learning_rate * grad_b2
            self.w1 -= self.learning_rate * grad_w1
            self.b1 -= self.learning_rate * grad_b1
        return history


@dataclasses.dataclass(slots=True)
class CANNRegressor:
    """GLM floor + bounded neural residual adjustment for severity pricing.

    Attributes:
      hidden_dim: Hidden width of the residual network.
      l2: L2 weight decay for the residual network.
      learning_rate: Gradient-descent learning rate for the residual net.
      seed: RNG seed.
    """

    hidden_dim: int = 16
    l2: float = 1e-3
    learning_rate: float = 0.05
    seed: int = 0
    _glm: object = dataclasses.field(init=False, default=None)
    _mu: np.ndarray = dataclasses.field(init=False, default=None)
    _sigma: np.ndarray = dataclasses.field(init=False, default=None)
    _net: TwoLayerResidualNet = dataclasses.field(init=False, default=None)

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 500) -> CANNRegressor:
        """Fits the GLM floor, then the residual net on log-residuals.

        Args:
          x: Shape ``(n, k)`` raw policy feature matrix.
          y: Shape ``(n,)`` strictly positive severity target.
          epochs: Residual-net training epochs.

        Returns:
          ``self``, for chaining.
        """
        self._glm = PoissonRegressor(alpha=1e-4, max_iter=500)
        self._glm.fit(x, y)
        glm_pred = np.clip(self._glm.predict(x), 1e-6, None)

        self._mu = x.mean(axis=0)
        self._sigma = x.std(axis=0) + 1e-8
        x_std = (x - self._mu) / self._sigma

        log_residual = np.log(y) - np.log(glm_pred)
        self._net = TwoLayerResidualNet(
            input_dim=x.shape[1],
            hidden_dim=self.hidden_dim,
            l2=self.l2,
            learning_rate=self.learning_rate,
            seed=self.seed,
        )
        self._net.fit(x_std, log_residual, epochs=epochs)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Returns ``glm_pred * exp(net_residual)`` severity predictions.

        Args:
          x: Shape ``(n, k)`` raw policy feature matrix.

        Returns:
          Shape ``(n,)`` predicted severities.
        """
        glm_pred = np.clip(self._glm.predict(x), 1e-6, None)
        x_std = (x - self._mu) / self._sigma
        residual = self._net.predict(x_std)
        return glm_pred * np.exp(residual)
