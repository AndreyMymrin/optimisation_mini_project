"""Stable binary losses and logit (sub)gradients for the homework."""

import numpy as np
from scipy.special import expit


def cross_entropy(logits, target):
    """Return unreduced BCE for class-1 target probabilities in [0, 1]."""
    return ((1.0 - target) * np.logaddexp(0.0, logits)
            + target * np.logaddexp(0.0, -logits))


def worst_target(logits, nominal, delta):
    """Select a maximizing target, using the nominal target at an exact tie."""
    if not 0 <= delta <= 1:
        raise ValueError("delta must be in [0, 1]")
    lower = np.maximum(0.0, nominal - delta)
    upper = np.minimum(1.0, nominal + delta)
    return np.where(logits > 0, lower, np.where(logits < 0, upper, nominal))


def robust_cross_entropy(logits, nominal, delta):
    return cross_entropy(logits, worst_target(logits, nominal, delta))


def smoothed_target(labels, epsilon):
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be in [0, 1]")
    return (1.0 - epsilon) * labels + epsilon / 2.0


def logit_gradient(logits, target):
    # Equivalent to sigmoid(logits) - target, with less cancellation in the tails.
    return (1.0 - target) * expit(logits) - target * expit(-logits)
