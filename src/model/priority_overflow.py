"""Research prototype: bounded smooth overflow with resident-adult priority.

Capacity must be specified in the same pre-departure census as the stage pools.
Frozen experiment forks explicitly select these rules; legacy fits are unchanged.
"""
import jax.numpy as jnp
from jax.nn import softplus
from jax.nn import sigmoid


def smooth_retained(pool,capacity,width):
    """Smooth min(pool,capacity), zero on either axis, within [0,min(pool,capacity)].

    Requires nonnegative pools/capacity and strictly positive width. Evaluate the
    equivalent symmetric expression to avoid subtracting two huge pool terms.
    """
    return -width*jnp.logaddexp(-pool/width,-capacity/width)+width*softplus(-(pool+capacity)/width)


def priority_overflow(adults,young,capacity,width):
    """Allocate adults first, then young; return retained and exporting pools.

    Use the same positive smoothing width for both allocations, including when
    the remaining juvenile capacity is zero. Width is fixed for first tests.
    """
    retained_a=smooth_retained(adults,capacity,width)
    retained_j=smooth_retained(young,capacity-retained_a,width)
    return retained_a,retained_j,adults-retained_a,young-retained_j


def overflow_probability(pool,capacity,width):
    """Export/pool, with its analytic extension at an empty population.

    The symmetric divided difference avoids subtracting pool and retained pool
    at low density. Its Taylor branch also has finite first/second derivatives.
    """
    u=pool/width;v=capacity/width
    small=jnp.abs(u)<1e-3
    safe=jnp.where(small,1.,u)
    s=sigmoid(-v)
    ordinary=(softplus(u-v)-softplus(-u-v))/safe
    series=2*s+(u*u/3)*s*(1-s)*(1-2*s)
    return jnp.where(small,series,ordinary)


def departure_probabilities(adults,young,capacity,mode,width_fraction=.05):
    width=width_fraction*jnp.maximum(capacity,1e-12)
    if mode=='shared':
        p=overflow_probability(adults+young,capacity,width)
        return p,p
    if mode!='adult_priority':raise ValueError(mode)
    ra=smooth_retained(adults,capacity,width)
    return (overflow_probability(adults,capacity,width),
            overflow_probability(young,jnp.maximum(capacity-ra,0.),width))
