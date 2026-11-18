"""Balanced data generation and the two NumPy models."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Split:
    x: np.ndarray
    y: np.ndarray


def sample(simulation, rng, size):
    y = rng.permutation(np.repeat([0.0, 1.0], size // 2))
    if simulation == "gaussian":
        x = (2 * y - 1)[:, None] + rng.normal(size=(size, 2))
    else:
        angle = rng.uniform(0, np.pi, size)
        x0 = np.column_stack((np.cos(angle), np.sin(angle)))
        x1 = np.array([1.0, 0.5]) - x0
        x = np.where(y[:, None] == 0, x0, x1) + rng.normal(0, 0.1, (size, 2))
    return Split(x, y)


class Logistic:
    def initialize(self, rng):
        return (rng.uniform(-np.sqrt(2), np.sqrt(2), 2), np.zeros(1))

    def forward(self, params, x):
        w, b = params
        return x @ w + b, None

    def backward(self, params, x, cache, residual):
        return x.T @ residual, np.array([residual.sum()])


class TwoLayer:
    def initialize(self, rng):
        bound1, bound2 = np.sqrt(6 / 10), np.sqrt(6 / 9)
        return (rng.uniform(-bound1, bound1, (2, 8)), np.zeros(8),
                rng.uniform(-bound2, bound2, 8), np.zeros(1))

    def forward(self, params, x):
        weights, bias, w, b = params
        hidden = np.tanh(x @ weights + bias)
        return hidden @ w + b, hidden

    def backward(self, params, x, hidden, residual):
        _, _, w, _ = params
        hidden_residual = residual[:, None] * w * (1 - hidden**2)
        return (x.T @ hidden_residual, hidden_residual.sum(axis=0),
                hidden.T @ residual, np.array([residual.sum()]))
