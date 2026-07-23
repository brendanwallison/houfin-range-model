"""Physical and invariance tests for juvenile movement/model regularization."""
import numpy as np
import jax.numpy as jnp

from src.model.age_priors import age_structure_log_prior, equilibrium_age_quantities
from src.model.build_kernels import (
    angular_weights_toroidal,
    make_juvenile_kernel_stack,
    toroidal_distance_grid,
)
from src.model.build_path_features import integrate_paths


SPLITS = [0.0, 155.36162529769288, 482.7446923028151, 1e9]


def test_directional_wedges_are_partition_of_unity_including_origin():
    weights = angular_weights_toroidal(31, 25)
    total = sum(weights.values())
    assert np.allclose(total, 1.0, atol=1e-6)
    assert np.allclose([float(w[0, 0]) for w in weights.values()], 0.25)


def test_juvenile_stack_conserves_mass_and_realizes_configured_mdd():
    # Production-sized padded lattice: finite-domain truncation makes the
    # discrete mean (~321 km) slightly below the continuous target (330 km).
    lx, ly, cell_km = 447, 265, 27.0
    stack, labels = make_juvenile_kernel_stack(
        lx, ly, cell_km, SPLITS, mean_dist=330.0, shape=0.468
    )
    distance = toroidal_distance_grid(lx, ly, cell_km)
    realized = float(jnp.sum(jnp.sum(stack, axis=0) * distance))
    assert stack.shape == (12, ly, lx)
    assert len(labels) == 12
    assert np.isclose(float(stack.sum()), 1.0, atol=3e-6)
    assert abs(realized - 330.0) / 330.0 < 0.05


def test_land_conditioned_operator_preserves_constant_fields():
    ny, nx = 7, 9
    land = jnp.ones((ny, nx), dtype=jnp.float32)
    z = jnp.full((1, ny, nx, 2), 3.25, dtype=jnp.float32)
    # A broad, positive cohort avoids degenerate zero-mass resized kernels.
    kernel = jnp.ones((1, 2 * ny - 1, 2 * nx - 1), dtype=jnp.float32)
    kernel /= kernel.sum()
    out = integrate_paths(z, kernel, land, steps=3)
    assert np.allclose(np.asarray(out), 3.25, rtol=2e-5, atol=2e-5)


def test_local_age_structure_prior_is_resolution_invariant():
    rho = jnp.array([[0.2, 0.5], [0.7, 0.9]], dtype=jnp.float32)
    base = float(age_structure_log_prior(rho, alpha=2, beta=3, effective_sites=50))
    tiled = float(
        age_structure_log_prior(
            jnp.tile(rho, (8, 11)), alpha=2, beta=3, effective_sites=50
        )
    )
    assert np.isclose(base, tiled, rtol=1e-6, atol=1e-6)


def test_equilibrium_algebra_matches_survival_then_reproduction_census():
    sa, sj, fmax = 0.8, 0.4, 4.0
    # Saturated Allee term makes F(K) equal the lambda=1 target.
    c, f_at_k, lam, rho = equilibrium_age_quantities(
        sa, sj, fmax, K=100.0, allee_gamma=10.0
    )
    expected_f = (1.0 - sa) / (sa * sj)
    expected_rho = (expected_f * sa) / (expected_f * sa + 1.0)
    assert np.isclose(float(f_at_k), expected_f, rtol=2e-5)
    assert np.isclose(float(lam), 1.0, rtol=2e-5)
    assert np.isclose(float(rho), expected_rho, rtol=2e-5)
