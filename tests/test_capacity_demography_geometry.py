"""Nesting, covariance and active-coordinate checks for the structural field split."""
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.handlers import seed, trace
import pytest

from src.model import age_priors as ap
from src.model.habitat_geometry import (
    capacity_demography_coefficients, capacity_demography_loadings,
)


@pytest.mark.parametrize("r", [0., .6, 1.])
def test_prior_covariance_and_field_order(r):
    loadings = np.asarray(capacity_demography_loadings(r))
    expected = np.ones((4, 4))
    expected[2, :] = r
    expected[:, 2] = r
    expected[2, 2] = 1.
    np.testing.assert_allclose(loadings @ loadings.T, expected, atol=1e-7)


def test_exact_single_field_nesting_and_absence_of_dead_factor(monkeypatch):
    monkeypatch.setattr(ap, "_MANIFOLD_PRIOR", {"fixed_capacity_correlation": 1.})
    amplitudes = jnp.array([.2, .3, 2., .1])
    tr = trace(seed(ap._capacity_demography_w_env, 3)).get_trace(7, amplitudes, 1.)
    assert "capacity_factor" not in tr
    np.testing.assert_array_equal(tr["w_env"]["value"], tr["manifold_factor"]["value"] * amplitudes)


def test_independent_capacity_does_not_alter_demographic_coefficients():
    f, g = jnp.arange(7.)[:, None], -jnp.arange(7.)[:, None]
    a = jnp.array([.2, .3, 2., .1])
    first = capacity_demography_coefficients(f, g, a, 0.)
    second = capacity_demography_coefficients(f, g + 1., a, 0.)
    np.testing.assert_array_equal(first[:, [0, 1, 3]], second[:, [0, 1, 3]])
    np.testing.assert_allclose(second[:, 2] - first[:, 2], 2.)
    # The split contains the old fitted predictions at f_K=f_demographic.
    np.testing.assert_array_equal(capacity_demography_coefficients(f, f, a, 0.), f * a)


def test_finite_active_gradients_for_interior_correlation():
    f = jnp.arange(1., 8.)[:, None]
    objective = lambda g: jnp.sum(capacity_demography_coefficients(f, g, jnp.ones(4), .6) ** 2)
    grad = np.asarray(jax.grad(objective)(f))
    assert np.isfinite(grad).all() and np.all(grad != 0.)


@pytest.mark.parametrize("r", [-.1, 1.1])
def test_invalid_correlation_rejected(monkeypatch, r):
    monkeypatch.setattr(ap, "_MANIFOLD_PRIOR", {"fixed_capacity_correlation": r})
    with pytest.raises(ValueError):
        seed(ap._capacity_demography_w_env, 0)(3, jnp.ones(4), 1.)
