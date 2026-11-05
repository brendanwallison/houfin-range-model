"""dispersal.kernel_discretization: point stays the default and bit-identical; cell_integrated conserves
mass and makes coarse kernels match the fine kernel aggregated to coarse blocks."""
import numpy as np
import pytest


def test_resolver_defaults_to_point_and_keeps_dispersal_spec_unchanged():
    from src.model.dispersal_spec import POINT_DISCRETIZATION, dispersal_spec, kernel_discretization
    base = {"dispersal": {"adult_mdd_km": 100.0, "adult_shape": 0.859, "juvenile_mdd_km": 330.0,
                          "juvenile_shape": 0.468, "juvenile_radial_splits_km": [0.0, 155.0, 483.0, 1e9],
                          "path_integration_steps": 20, "path_operator": "land_conditioned_neighborhood"}}
    assert kernel_discretization(base) == POINT_DISCRETIZATION
    integ = {"dispersal": {**base["dispersal"],
                           "kernel_discretization": {"method": "cell_integrated", "subsamples": 9}}}
    assert kernel_discretization(integ) == {"method": "cell_integrated", "subsamples": 9}
    assert dispersal_spec(integ) == dispersal_spec(base)     # recorded spec (and its hashes) unchanged
    with pytest.raises(ValueError):
        kernel_discretization({"dispersal": {"kernel_discretization": {"method": "area"}}})


def test_point_is_historical_construction():
    import jax.numpy as jnp
    from src.model.build_kernels import get_gamma_scale, radial_kernel_grid, toroidal_distance_grid
    r = toroidal_distance_grid(61, 41, 27.0)
    k = jnp.exp(-(r / get_gamma_scale(100.0, 0.859)) ** 0.859)
    assert np.array_equal(np.asarray(k / jnp.sum(k)), np.asarray(radial_kernel_grid(61, 41, 27.0, 100.0, 0.859)))


def test_integrated_mass_and_resolution_consistency():
    from src.model.build_kernels import make_juvenile_kernel_stack, radial_kernel_grid
    dz = {"method": "cell_integrated", "subsamples": 9}
    stack, _ = make_juvenile_kernel_stack(121, 81, 54.0, [0.0, 155.0, 483.0, 1e9], 330.0, 0.468, dz)
    assert abs(float(stack.sum()) - 1.0) < 2e-5
    # 27 km kernel aggregated onto 54 km blocks vs native 54 km: integrated agrees, point does not.
    def self_agg(k27):
        k = np.asarray(k27, np.float64)
        m = {-1: 1.0, 0: 2.0, 1: 1.0}
        return sum(0.25 * m[a] * m[b] * k[a % k.shape[0], b % k.shape[1]] for a in (-1, 0, 1) for b in (-1, 0, 1))
    agg = self_agg(radial_kernel_grid(161, 121, 27.0, 100.0, 0.859, dz))
    nat_i = float(radial_kernel_grid(81, 61, 54.0, 100.0, 0.859, dz)[0, 0])
    nat_p = float(radial_kernel_grid(81, 61, 54.0, 100.0, 0.859)[0, 0])
    assert abs(nat_i - agg) / agg < 0.02
    assert nat_p > 1.5 * agg
