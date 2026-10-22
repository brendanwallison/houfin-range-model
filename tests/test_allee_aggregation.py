"""allee_prior.aggregation_k: encounter averaged over negative-binomial aggregation."""
import jax
import jax.numpy as jnp
import numpy as np

import src.model.age_forward as af


def test_limits_and_monte_carlo():
    jax.config.update("jax_enable_x64", True)
    x = jnp.array([1e-4, 0.1, 1.0, 3.0, 10.0])
    np.testing.assert_allclose(af.allee_mated_fraction(x, 1e9), -np.expm1(-np.asarray(x)), rtol=1e-6)   # k -> inf
    np.testing.assert_allclose(af.allee_mated_fraction(x, 1.0), x / (1 + x), rtol=1e-12)               # k = 1
    # small-x slope identical for every k (still a strong Allee effect)
    for k in (0.5, 1.0, 3.0):
        np.testing.assert_allclose(af.allee_mated_fraction(jnp.array(1e-6), k), 1e-6, rtol=1e-5)
    # E[1 - exp(-X)], X ~ Gamma(k, mean x): Monte Carlo agrees
    rng = np.random.default_rng(0)
    k, m = 2.0, 1.5
    X = rng.gamma(k, m / k, 2_000_000)
    np.testing.assert_allclose(float(af.allee_mated_fraction(jnp.array(m), k)), np.mean(-np.expm1(-X)), rtol=2e-3)
    # softer: for x above the median the aggregated form is lower, below it higher? -> just monotone in k at x = 1
    vals = [float(af.allee_mated_fraction(jnp.array(1.0), k)) for k in (1.0, 3.0, 1e6)]
    assert vals[0] < vals[1] < vals[2]


def test_default_is_original_form():
    assert af.ALLEE_AGG_K is None or isinstance(af.ALLEE_AGG_K, float)
    old = af.ALLEE_AGG_K
    try:
        af.ALLEE_AGG_K = None
        x = jnp.array([0.3, 2.0])
        np.testing.assert_array_equal(af.allee_mated_fraction(x), 1.0 - jnp.exp(-x))
    finally:
        af.ALLEE_AGG_K = old
