"""CBC NB2 observation likelihood; seasonal redistribution never feeds demography.

Effort response follows the Box-Cox family of Link & Sauer/Link et al.
Circle heterogeneity has a fixed prior SD to avoid joint-MAP scale collapse.
"""
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def boxcox_effort(log_relative_hours, power):
    # Stable at p=0, including derivatives. Series avoids a 0/0 JAX branch.
    p=jnp.asarray(power);x=log_relative_hours
    safe=jnp.where(jnp.abs(p)<1e-4,1.,p)
    ordinary=jnp.expm1(p*x)/safe
    series=x+.5*p*x*x+(p*p/6.)*x*x*x
    return jnp.where(jnp.abs(p)<1e-4,series,ordinary)


def winter_readout(adult,young,data,migrant_fraction,young_availability):
    """Positive linear observation operator; no changes to simulated pools."""
    ci=data['cbc_circle'];idx=data['cbc_source_idx'][ci]
    weights=(1-migrant_fraction)*data['cbc_w_local'][ci]+migrant_fraction*data['cbc_w_seasonal'][ci]
    t=data['cbc_t'][:,None]
    return jnp.sum(weights*(adult[t,idx]+young_availability*young[t,idx]),axis=1)


def cbc_observe(data,Na_grid,Nj_grid,pop_scalar,spec):
    if spec.get('family')=='early_detection':
        from src.model.cbc_early import early_observe
        return early_observe(data,Na_grid,Nj_grid,pop_scalar,spec)
    rows,cols=data['land_rows'],data['land_cols']
    adult,young=Na_grid[:,rows,cols],Nj_grid[:,rows,cols]
    fraction=(jnp.asarray(float(spec['fixed_migrant_fraction'])) if 'fixed_migrant_fraction' in spec
              else numpyro.sample('cbc_migrant_fraction',dist.Beta(2.,2.)))
    young_q=(jnp.asarray(float(spec['fixed_young_availability'])) if 'fixed_young_availability' in spec
             else numpyro.sample('cbc_young_availability',dist.Beta(2.,2.)))
    density=winter_readout(adult,young,data,fraction,young_q)*pop_scalar
    intercept=numpyro.sample('cbc_log_scale',dist.Normal(0.,2.5))
    raw=numpyro.sample('cbc_circle_raw',dist.Normal(0.,1.).expand([int(data['cbc_n_circles'])]).to_event(1))
    effects=float(spec.get('circle_sd',1.))* (raw-jnp.mean(raw))
    slope=numpyro.sample('cbc_effort_slope',dist.LogNormal(jnp.log(.6),.5))
    power=numpyro.sample('cbc_effort_power',dist.Uniform(-.5,.5))
    feeder=numpyro.sample('cbc_feeder_effect',dist.Normal(0.,.5))
    feeder_missing=numpyro.sample('cbc_feeder_missing_effect',dist.Normal(0.,.5))
    log_mu=jnp.log(jnp.maximum(density,1e-8))+intercept+effects[data['cbc_circle']]
    log_mu+=slope*boxcox_effort(data['cbc_log_effort'],power)+feeder*data['cbc_feeder']
    log_mu+=feeder_missing*data['cbc_feeder_missing']
    # Clipping only outside the representable float32 likelihood range.
    mu=jnp.exp(jnp.clip(log_mu,-30.,30.))
    conc=numpyro.sample('cbc_concentration',dist.LogNormal(jnp.log(.5),1.5))
    ll=dist.NegativeBinomial2(mean=mu,concentration=conc).log_prob(data['cbc_count'])
    train=data['cbc_train']>0
    numpyro.factor('cbc_loglik',jnp.sum(jnp.where(train,ll,0.)))
    numpyro.deterministic('cbc_expected',mu)
    numpyro.deterministic('cbc_loglik_train',jnp.sum(jnp.where(train,ll,0.)))
    numpyro.deterministic('cbc_loglik_test',jnp.sum(jnp.where(~train,ll,0.)))
    return mu
