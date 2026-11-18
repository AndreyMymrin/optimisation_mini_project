"""Reproduce the numerical and derivative checks in Section 2."""

import numpy as np

from losses import (cross_entropy, logit_gradient, robust_cross_entropy,
                    smoothed_target, worst_target)


def main():
    p1 = np.array([0.1, 0.4, 0.6])
    logits = np.log(p1 / (1 - p1))
    ce = cross_entropy(logits, 0.0)
    robust = robust_cross_entropy(logits, 0.0, 0.2)
    np.testing.assert_allclose(ce, [0.1054, 0.5108, 0.9163], atol=5e-5)
    np.testing.assert_allclose(robust, [0.5448, 0.5919, 0.9163], atol=5e-5)
    np.testing.assert_array_equal(worst_target(logits, 0.0, 0.2), [0.2, 0.2, 0.0])
    print("p1, CE, RobustCE, q*:")
    print(np.column_stack((p1, ce, robust, worst_target(logits, 0.0, 0.2))))

    rng = np.random.default_rng(911)
    logits = np.r_[rng.normal(0, 10, 2048), -1000.0, 0.0, 1000.0]
    nominal = rng.choice([0.0, 0.2, 0.5, 0.9, 1.0], size=len(logits))
    ce = cross_entropy(logits, nominal)
    np.testing.assert_array_equal(robust_cross_entropy(logits, nominal, 0), ce)
    previous = ce
    for delta in (0.02, 0.2, 0.5, 1.0):
        robust = robust_cross_entropy(logits, nominal, delta)
        assert np.isfinite(robust).all()
        assert np.min(robust - previous) >= -1e-12
        previous = robust
    np.testing.assert_allclose(previous, np.logaddexp(0, np.abs(logits)))

    # Direct dense search using the probability-space formula independently.
    error = 0.0
    for p in (0.01, 0.1, 0.4, 0.5, 0.6, 0.9, 0.99):
        for m in (0.0, 0.2, 0.5, 0.9, 1.0):
            for delta in (0.0, 0.02, 0.2, 0.5, 1.0):
                targets = np.linspace(max(0, m - delta), min(1, m + delta), 2001)
                direct = np.max(-(1 - targets) * np.log1p(-p) - targets * np.log(p))
                analytic = robust_cross_entropy(np.log(p / (1 - p)), m, delta)
                error = max(error, abs(direct - analytic))
    assert error < 1e-12

    s = np.array([-4.0, -0.5, 0.5, 4.0])
    step = 1e-5
    for m in (0.0, 0.3, 1.0):
        delta = 0.2
        finite_difference = (robust_cross_entropy(s + step, m, delta)
                             - robust_cross_entropy(s - step, m, delta)) / (2 * step)
        np.testing.assert_allclose(logit_gradient(s, worst_target(s, m, delta)),
                                   finite_difference, atol=1e-9)
        left = 0.5 - min(1, m + delta)
        right = 0.5 - max(0, m - delta)
        at_tie = logit_gradient(0, worst_target(0, m, delta))
        assert left - 1e-12 <= at_tie <= right + 1e-12

    labels = np.array([0.0, 1.0])
    np.testing.assert_array_equal(smoothed_target(labels, 0), labels)
    np.testing.assert_array_equal(smoothed_target(labels, 1), [0.5, 0.5])
    np.testing.assert_allclose(smoothed_target(labels, 0.2), [0.1, 0.9])
    print(f"Maximum dense-search error over 175 cases: {error:.3e}")
    print("All loss and derivative checks passed.")


if __name__ == "__main__":
    main()
