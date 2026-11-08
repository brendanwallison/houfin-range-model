"""population_model.kernel_ramp: the split juvenile dispersal reduces to the base stack at share 0 and to the alternate
stack at share 1 (ROBUST lever A, 2026-11-06)."""
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from src.model.age_forward import juvenile_dispersal_vectorized  # noqa: E402


def _stack(rng, K, Ny, Nx):
    k = rng.random((K, Ny, Nx)); k /= k.sum(axis=(1, 2), keepdims=True)
    return jnp.fft.fft2(jnp.asarray(k))


def test_share_limits_and_linearity():
    rng = np.random.default_rng(0); K, Ny, Nx = 3, 8, 10
    base, alt = _stack(rng, K, Ny, Nx), _stack(rng, K, Ny, Nx)
    edge_b = jnp.asarray(rng.uniform(.5, 1., (K, Ny, Nx))); edge_a = jnp.asarray(rng.uniform(.5, 1., (K, Ny, Nx)))
    disp = jnp.asarray(rng.random((Ny, Nx))); Q = jnp.asarray(rng.uniform(.5, 1., (K, Ny, Nx)))
    ref_base = juvenile_dispersal_vectorized(disp, base, Q, edge_b)
    ref_alt = juvenile_dispersal_vectorized(disp, alt, Q, edge_a)
    z = jnp.zeros((Ny, Nx)); o = jnp.ones((Ny, Nx))
    np.testing.assert_allclose(juvenile_dispersal_vectorized(disp, base, Q, edge_b, alt=(alt, edge_a, z)), ref_base, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(juvenile_dispersal_vectorized(disp, base, Q, edge_b, alt=(alt, edge_a, o)), ref_alt, rtol=1e-12, atol=1e-14)
    h = .3 * o
    np.testing.assert_allclose(juvenile_dispersal_vectorized(disp, base, Q, edge_b, alt=(alt, edge_a, h)),
                               .7 * ref_base + .3 * ref_alt, rtol=1e-12, atol=1e-14)
