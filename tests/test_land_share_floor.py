import numpy as np
import jax.numpy as jnp

from src.model.build_path_features import integrate_paths, low_land_share


def _setup():
    land = np.zeros((9, 9), np.float32); land[4, 4] = 1; land[0:3, 0:3] = 1
    k = np.zeros((2, 17, 17), np.float32)
    k[0, 0, 0] = 1.0                                   # band 0: all mass on the self-cell
    k[1, 8, 8] = 1.0                                   # band 1: all mass far away (off-grid / water)
    Z = np.random.default_rng(0).normal(size=(1, 9, 9, 3)).astype(np.float32)
    return jnp.asarray(Z), jnp.asarray(k), jnp.asarray(land)


def test_floor_off_is_the_old_operator_and_on_replaces_only_low_share_pairs():
    Z, k, land = _setup()
    off = np.asarray(integrate_paths(Z, k, land, steps=2))
    on = np.asarray(integrate_paths(Z, k, land, steps=2, land_share_floor=1e-4))
    low = low_land_share(k, land, 2, 1e-4)
    assert low.shape == (2, 9, 9) and low[1].any() and not (low & (np.asarray(land)[None] < .5)).any()
    keep = ~np.broadcast_to(np.transpose(low, (1, 2, 0))[None, :, :, None, :], off.shape)
    np.testing.assert_array_equal(on[keep], off[keep])
    # every step is a convex average of land Z or the cell's own Z, so the convex bound holds on land
    lm = np.asarray(land) > .5
    assert np.abs(on[0][lm]).max() <= np.abs(np.asarray(Z)[0][lm]).max() + 1e-5
