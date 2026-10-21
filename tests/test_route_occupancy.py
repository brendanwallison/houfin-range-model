"""observation.route_occupancy (E072): mean-preserving zero inflation, finite as mu -> 0."""
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from src.model.age_priors import route_occupancy_parts


def test_mean_preserved_and_limits():
    jax.config.update("jax_enable_x64", True)
    mu = jnp.array([0.0, 1e-9, 1e-3, 0.1, 1.0, 10.0, 100.0])
    gate, mu_occ, logits = route_occupancy_parts(mu, 0.3)
    np.testing.assert_allclose(jax.nn.sigmoid(logits), gate, rtol=1e-9, atol=1e-15)   # gate itself rounds at large mu
    occ = jax.nn.sigmoid(-logits)                                  # 1 - gate, exact
    np.testing.assert_allclose(occ * mu_occ, jnp.maximum(mu, 1e-12), rtol=1e-12)
    assert float(gate[0]) > 0.999999 and float(gate[-1]) < 1e-100
    np.testing.assert_allclose(mu_occ[:2], 0.3, rtol=1e-6)        # occupied-route mean -> patch size m


def test_gradient_finite_at_zero_density():
    def lp(m):
        _, mu_occ, logits = route_occupancy_parts(jnp.array([0.0, 0.01, 2.0]), m)
        return dist.ZeroInflatedNegativeBinomial2(mean=mu_occ, concentration=2.0, gate_logits=logits).log_prob(
            jnp.array([0, 1, 3])).sum()
    g = jax.grad(lp)(0.3)
    assert np.isfinite(float(g))
