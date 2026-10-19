"""subcell_reproduction's town survival tilt (sigma_s)."""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
from src.model.age_forward import subcell_reproduction  # noqa: E402

S, Ny, Nx = 6, 3, 4
a = jnp.array([0.15, 0.225, 0.25, 0.2, 0.12, 0.055])
u = jnp.broadcast_to(jnp.linspace(-1.5, 2.5, S)[:, None, None], (S, Ny, Nx))
pools = (jnp.full((Ny, Nx), 0.3), jnp.full((Ny, Nx), 0.1), jnp.full((Ny, Nx), 0.05))
args = (jnp.full((Ny, Nx), 0.55), jnp.full((Ny, Nx), 0.3), jnp.full((Ny, Nx), 5.0), jnp.full((Ny, Nx), 1.0), 30.0, u, a, 0.0, 0.0)


def test_zero_tilt_is_the_untilted_model():
    off = subcell_reproduction(pools, *args)
    on = subcell_reproduction(pools, *args, sigma_s=0.0)
    for x, y in zip(off, on):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-10)


def test_positive_tilt_helps_a_sparse_cell_and_has_finite_gradient():
    sparse = tuple(p * 1e-2 for p in pools)
    base = float(sum(subcell_reproduction(sparse, *args)).sum())
    town = float(sum(subcell_reproduction(sparse, *args, sigma_s=1.0)).sum())
    assert town > base            # birds sort into the towns, where survival is now higher
    g = jax.grad(lambda s: sum(subcell_reproduction(sparse, *args, sigma_s=s)).sum())(0.5)
    assert np.isfinite(float(g))
