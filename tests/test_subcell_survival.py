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


def test_smooth_allocation_gives_descent_gradients_near_thresholds():
    # Heterogeneous cells/slices put the equal-lambda level near slice thresholds (the hard max(n, 0) corners
    # that stalled L-BFGS in E058b/c). With the smoothed positive part, a small step along -grad must lower U.
    import src.model.age_forward as af
    rng = np.random.default_rng(0)
    u_h = jnp.asarray(rng.normal(0, 1.2, (S, Ny, Nx)))
    pl = tuple(jnp.asarray(rng.uniform(0.001, 0.4, (Ny, Nx))) for _ in range(3))
    old = af.SMOOTH_ALLOC
    af.SMOOTH_ALLOC = 1e-3
    try:
        def f(th):
            a2 = list(args); a2[5] = u_h; a2[7] = th[0]; a2[8] = th[1]
            return sum(subcell_reproduction(pl, *a2, sigma_s=th[2])).sum()
        th = jnp.array([-0.5, 0.2, 0.4])
        g = jax.grad(f)(th)
        for h in (1e-6, 1e-5):
            assert float(f(th - h * g / jnp.linalg.norm(g))) < float(f(th))
        fd = np.array([(float(f(th.at[i].add(1e-6))) - float(f(th.at[i].add(-1e-6)))) / 2e-6 for i in range(3)])
        np.testing.assert_allclose(np.asarray(g), fd, rtol=1e-4, atol=1e-8)
    finally:
        af.SMOOTH_ALLOC = old
