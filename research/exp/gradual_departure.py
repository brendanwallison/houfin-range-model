"""Stage-specific departure; K remains the inherited equilibrium reference."""
import jax.numpy as jnp
import jax.nn as jnn


def departure_probabilities(adult, young, capacity, curve, adult_multiplier):
    if curve not in ('veit', 'rational'):
        raise ValueError(curve)
    if not 0 <= adult_multiplier <= 1:
        raise ValueError('adult_multiplier must lie in [0,1]')
    x = (jnp.maximum(adult, 0.) + jnp.maximum(young, 0.)) / jnp.maximum(capacity, 1e-12)
    if curve == 'rational':
        p = x / (1. + x)
    else:
        # Exact inherited smooth Veit control, including its numerical floor.
        p = jnp.clip(x - jnn.softplus(10. * (x - 1.)) / 10., 1e-6, 1.)
    return adult_multiplier * p, p
