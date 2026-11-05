"""Fixed-calibration historical range constraints; not a CBC count likelihood.

NB2 zero/censored-positive probabilities reuse the project's observation family.
Calibration is frozen from the reference fit; synthetic range evidence cannot
weaken itself by fitting a CBC scale or a new dispersion parameter.
"""
import jax.numpy as jnp
import numpyro


def zero_loss(density,concentration):
    return concentration*jnp.log1p(jnp.maximum(density,0.)/concentration)


def positive_loss(density,concentration):
    z=zero_loss(density+1e-12,concentration)
    return -jnp.log(-jnp.expm1(-z))


def range_observe(data,Na,Nj,pop_scalar,spec):
    rr,cc=data['land_rows'],data['land_cols']
    n=(Na+Nj)[:,rr,cc]*pop_scalar
    k=float(spec['fixed_concentration'])
    loss=zero_loss(n[data['hr_zero_t'],data['hr_zero_cell']],k)
    numpyro.factor('historical_range_absence',-jnp.sum(loss*data['hr_zero_weight']))
    # Union of local and potential seasonal source support: a detection requires
    # accessible population, but does not locate its permanent breeding cell.
    pt=data['hr_positive_t'][:,None];idx=data['hr_positive_idx']
    available=jnp.sum(n[pt,idx]*data['hr_positive_w'],axis=1)
    pl=positive_loss(available,k)
    numpyro.factor('historical_range_presence',-jnp.sum(pl*data['hr_positive_weight']))
    numpyro.deterministic('historical_range_absence_loss',jnp.sum(loss*data['hr_zero_weight']))
    numpyro.deterministic('historical_range_presence_loss',jnp.sum(pl*data['hr_positive_weight']))
