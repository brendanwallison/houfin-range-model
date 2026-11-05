"""Two-direction habitat coefficients with an explicitly shared demographic field.

Columns retain the model contract [Sa, Fmax, K, Sj]. Correlation here is a
coefficient-prior correlation, not the realized correlation of nonlinear rates.
"""
import jax.numpy as jnp


def capacity_demography_loadings(correlation):
    """Unit-variance loadings; Sa/Fmax/Sj coincide, K has correlation r with them."""
    r = jnp.asarray(correlation)
    demographic = jnp.stack([jnp.ones_like(r), jnp.zeros_like(r)])
    capacity = jnp.stack([r, jnp.sqrt(jnp.maximum(1.0 - r * r, 0.0))])
    return jnp.stack([demographic, demographic, capacity, demographic])


def capacity_demography_coefficients(shared, capacity, amplitudes, correlation):
    """Compose iid feature draws without normalizing or rotating exported Z."""
    factors = jnp.concatenate([shared, capacity], axis=-1)
    return (factors @ capacity_demography_loadings(correlation).T) * amplitudes
