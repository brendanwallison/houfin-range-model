"""Differentiable port of the existing coarse adult dead-recovery likelihood.

Same monthly multinomial model as bbl_survival.py: constant annual survival per
origin region, era reporting conditional on death, incomplete first exposure,
administrative censoring and all never-recovered releases. This is a reusable
likelihood block, NOT the spatial simulator adapter or juvenile movement model.
"""
import jax
import jax.numpy as jnp


def adult_region_loglik(survival, reporting_by_era, era, row, col, recovered_counts,
                        never_recovered_counts, initial_fraction=.5):
    reporting=reporting_by_era[era]
    monthly=jnp.exp(jnp.log(survival)/12)
    first=jnp.exp(jnp.log(survival)*initial_fraction/12)
    def step(future,r):
        probability=(1-monthly)*r+monthly*future
        return probability,probability
    _,reverse=jax.lax.scan(step,jnp.asarray(0.,dtype=reporting.dtype),reporting[::-1])
    future=reverse[::-1]
    reported=(1-first)*reporting+first*jnp.concatenate([future[1:],jnp.zeros(1,dtype=future.dtype)])
    lag=col-row
    before=jnp.where(lag>0,lag-1+initial_fraction,0)/12
    duration=jnp.where(lag==0,initial_fraction,1)/12
    logprob=before*jnp.log(survival)+jnp.log(-jnp.expm1(duration*jnp.log(survival)))+jnp.log(reporting[col])
    return jnp.sum(recovered_counts*logprob)+jnp.sum(never_recovered_counts*jnp.log1p(-reported))
