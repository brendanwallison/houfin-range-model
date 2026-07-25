"""Physical and invariance tests for juvenile movement/model regularization."""
import copy

import numpy as np
import jax.numpy as jnp
import pytest
from scipy.special import gamma, gammainc

from src.config_utils import load_age_model_config
from src.model.age_priors import age_structure_log_prior, equilibrium_age_quantities
from src.model.build_kernels import (
    angular_weights_toroidal,
    dispersal_spec,
    make_juvenile_kernel_stack,
    resolve_radial_splits,
    toroidal_distance_grid,
)
from src.model.build_path_features import integrate_paths


# The committed baseline's pinned splits. These exact literals are recorded in the
# path_feature_meta.json of the existing Z_disp cube, and ingest compares the
# resolved dispersal dict for EXACT equality -- so they must never be rounded or
# re-derived. See resolve_radial_splits.
PINNED_BASELINE_SPLITS = [0.0, 155.36162529769288, 482.7446923028151, 1e9]
SWEEP_MDDS = (200.0, 250.0, 300.0, 330.0, 350.0)


def _derived(mdd, shape=0.468, quantiles=None):
    return resolve_radial_splits(mdd, shape, "derive", quantiles)


def _band_mass_fractions(splits, mdd, shape):
    """Continuous kernel mass in each radial band (the property splits control)."""
    g2, g3 = gamma(2.0 / shape), gamma(3.0 / shape)
    scale = mdd / (g3 / g2)
    cdf = [0.0 if r <= 0 else (1.0 if r >= 1e8
           else float(gammainc(2.0 / shape, (r / scale) ** shape))) for r in splits]
    return [b - a for a, b in zip(cdf[:-1], cdf[1:])]


def test_directional_wedges_are_partition_of_unity_including_origin():
    weights = angular_weights_toroidal(31, 25)
    total = sum(weights.values())
    assert np.allclose(total, 1.0, atol=1e-6)
    assert np.allclose([float(w[0, 0]) for w in weights.values()], 0.25)


@pytest.mark.parametrize("mdd", SWEEP_MDDS)
def test_juvenile_stack_conserves_mass_and_realizes_configured_mdd(mdd):
    # Production-sized padded lattice. Finite-domain truncation drops mass beyond
    # the domain, so the discrete mean sits BELOW the continuous target (330 ->
    # ~321 km) and the shortfall grows with the target. Hence a one-sided bound
    # rather than a symmetric tolerance.
    lx, ly, cell_km = 447, 265, 27.0
    splits = _derived(mdd)
    stack, labels = make_juvenile_kernel_stack(
        lx, ly, cell_km, splits, mean_dist=mdd, shape=0.468
    )
    distance = toroidal_distance_grid(lx, ly, cell_km)
    realized = float(jnp.sum(jnp.sum(stack, axis=0) * distance))
    assert stack.shape == (12, ly, lx)
    assert len(labels) == 12
    assert np.isclose(float(stack.sum()), 1.0, atol=3e-6)
    assert realized <= mdd * 1.005          # truncation cannot inflate the mean
    assert realized >= 0.90 * mdd           # but must not lose an order of magnitude


def test_pinned_splits_round_trip_unchanged():
    # Rounding these would silently invalidate every existing path_feature_meta.
    assert resolve_radial_splits(330.0, 0.468, PINNED_BASELINE_SPLITS) == PINNED_BASELINE_SPLITS


def test_committed_config_still_resolves_to_the_pinned_baseline():
    spec = dispersal_spec(load_age_model_config())
    assert spec["juvenile_radial_splits_km"] == PINNED_BASELINE_SPLITS
    # The returned dict's KEYS are part of the ingest guard's compared value; a
    # new key invalidates all previously built path features.
    assert set(spec) == {
        "adult_mdd_km", "adult_shape", "juvenile_mdd_km", "juvenile_shape",
        "juvenile_radial_splits_km", "path_integration_steps", "path_operator",
    }


def test_dispersal_spec_is_deterministic_and_idempotent():
    # This is the unit-level guard for model_inputs.py's exact-equality check
    # between path-feature metadata and the freshly resolved config.
    cfg = load_age_model_config()
    first = dispersal_spec(copy.deepcopy(cfg))
    second = dispersal_spec(copy.deepcopy(cfg))
    assert first == second
    derive_cfg = copy.deepcopy(cfg)
    derive_cfg["dispersal"]["juvenile_radial_splits_km"] = "derive"
    assert dispersal_spec(derive_cfg) == dispersal_spec(copy.deepcopy(derive_cfg))


@pytest.mark.parametrize("mdd", SWEEP_MDDS)
def test_derived_splits_give_equal_mass_bands_at_every_mdd(mdd):
    # The whole point of deriving: the radial discretization stops co-varying
    # with mdd. The pinned log-spaced splits fail this badly (67/28/4% at 150 km).
    splits = _derived(mdd)
    assert splits[0] == 0.0 and splits[-1] >= 1e8
    assert all(b > a for a, b in zip(splits[:-1], splits[1:]))
    masses = _band_mass_fractions(splits, mdd, 0.468)
    assert len(masses) == 3
    assert np.allclose(masses, 1.0 / 3.0, atol=0.02)


def test_derived_splits_scale_linearly_with_mdd():
    # scale is proportional to mdd for fixed shape, so the interior boundaries
    # must be too -- a cheap invariant that catches a quantile/scale mix-up.
    a, b = _derived(200.0), _derived(400.0)
    assert np.allclose(np.array(b[1:-1]) / np.array(a[1:-1]), 2.0, rtol=1e-6)


def test_custom_quantiles_are_honored():
    splits = _derived(330.0, quantiles=[0.25, 0.5, 0.75])
    assert len(splits) == 5  # 4 bands -> K = 16
    masses = _band_mass_fractions(splits, 330.0, 0.468)
    assert np.allclose(masses, 0.25, atol=0.02)


@pytest.mark.parametrize("bad", ["terciles", 3, {}, ""])
def test_malformed_split_spec_raises(bad):
    with pytest.raises(ValueError, match="must be a list of km boundaries"):
        resolve_radial_splits(330.0, 0.468, bad)


@pytest.mark.parametrize("bad", [[0.5, 100.0, 1e9], [0.0, 100.0, 50.0, 1e9], [0.0, 100.0]])
def test_malformed_pinned_split_lists_raise(bad):
    with pytest.raises(ValueError):
        resolve_radial_splits(330.0, 0.468, bad)


def test_short_mdd_warns_that_inner_cohort_is_unresolved():
    # 27 km cells -> 54 km boundary smoothing. At mdd=100 the tercile inner split
    # is ~40 km, below that width, so the inner cohort is only partly resolved.
    # The sweep deliberately visits short distances, so this warns, not raises.
    splits = _derived(100.0)
    assert splits[1] < 54.0
    with pytest.warns(RuntimeWarning, match="partly resolved"):
        make_juvenile_kernel_stack(97, 81, 27.0, splits, mean_dist=100.0, shape=0.468)


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
