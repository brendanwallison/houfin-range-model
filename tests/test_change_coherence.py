"""Independent Gram-matrix check of the change covariance used in the audit."""
import numpy as np

from research.exp.final_change_coherence import delta_cov


def test_change_covariance_matches_explicit_gram_and_endpoint_energy():
    rng = np.random.default_rng(4)
    a, b = rng.uniform(.1, 3, (2, 5, 2, 7))
    x = np.concatenate([a.reshape(-1, 7), b.reshape(-1, 7)])
    gram = (np.minimum(x[:, None], x[None, :]).sum(-1)
            / np.maximum(x[:, None], x[None, :]).sum(-1))
    D = np.zeros((5, 10))
    for k in range(5):
        D[k, 2*k:2*k+2] = [1, -1]
    expected = D @ gram[:10, 10:] @ D.T
    expected = (expected + expected.T) / 2
    i, j = np.indices((5, 5)).reshape(2, -1)
    np.testing.assert_allclose(delta_cov(a, b, i, j).reshape(5, 5), expected, atol=1e-14)
    # With independent halves equal, pair-matched V-C is half the squared
    # distance between the two changes, and is nonnegative for a PSD kernel.
    v = delta_cov(a, a, np.arange(5), np.arange(5))
    c = delta_cov(a, a, i, j)
    assert np.min((v[i] + v[j]) / 2 - c) > -1e-14


def test_identical_regional_change_has_unit_shared_ratio():
    state = np.array([[1., 2., 5.], [3., 4., 1.]])
    a = np.repeat(state[None], 4, axis=0)
    v = delta_cov(a, a, np.arange(4), np.arange(4))
    c = delta_cov(a, a, np.array([0, 1]), np.array([2, 3]))
    np.testing.assert_allclose(c / v[:2], 1)
    unchanged = a.copy()
    unchanged[:, 1] = unchanged[:, 0]
    np.testing.assert_allclose(delta_cov(unchanged, unchanged, np.arange(4), np.arange(4)), 0, atol=1e-14)
