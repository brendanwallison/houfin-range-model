"""Historical CBC detection model, conditional on the annual population process.

Bernoulli with complementary log-log link: detection is at least one encounter
in a Poisson encounter process. This is not a separately identified occupancy /
detection hierarchy. Effort adjustment follows the existing CBC Box-Cox model.
"""
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from src.model.cbc_joint import winter_readout,boxcox_effort


def detection_loglik(log_intensity,present):
    rate=jnp.exp(jnp.clip(log_intensity,-30.,25.))
    return jnp.where(present>0,jnp.log(-jnp.expm1(-rate)),-rate)


def early_observe(data,Na,Nj,pop_scalar,spec):
    rr,cc=data['land_rows'],data['land_cols']
    a,j=Na[:,rr,cc],Nj[:,rr,cc]
    fraction=float(spec.get('fixed_migrant_fraction',.35))
    density=winter_readout(a,j,data,fraction,1.)*pop_scalar
    scale=numpyro.sample('cbc_log_scale',dist.Normal(0.,2.5))
    raw=numpyro.sample('cbc_circle_raw',dist.Normal(0.,1.).expand([int(data['cbc_n_circles'])]).to_event(1))
    effect=float(spec.get('circle_sd',1.))*(raw-jnp.mean(raw))
    slope=numpyro.sample('cbc_effort_slope',dist.LogNormal(jnp.log(.6),.5))
    power=numpyro.sample('cbc_effort_power',dist.Uniform(-.5,.5))
    missing=numpyro.sample('cbc_effort_missing_effect',dist.Normal(0.,1.))
    feeder=numpyro.sample('cbc_feeder_effect',dist.Normal(0.,.5))
    log_rate=jnp.log(density+1e-8)+scale+effect[data['cbc_circle']]
    log_rate+=slope*boxcox_effort(data['cbc_log_effort'],power)
    log_rate+=missing*data['cbc_effort_missing']+feeder*data['cbc_feeder']
    ll=detection_loglik(log_rate,data['cbc_count']>0)
    numpyro.factor('cbc_early_loglik',jnp.sum(jnp.where(data['cbc_train']>0,ll,0.)))
    numpyro.deterministic('cbc_early_test_loglik',jnp.sum(jnp.where(data['cbc_train']>0,0.,ll)))
    if spec.get('envelope_enabled'):
        # Synthetic exterior range constraints: separate from surveyed nondetections,
        # with their own fixed weight and no circle effect or count magnitudes.
        n=(a+j)[data['cbc_envelope_t'],data['cbc_envelope_cell']]*pop_scalar
        ell=detection_loglik(jnp.log(n+1e-8)+scale,jnp.zeros_like(n))
        numpyro.factor('cbc_envelope_loglik',float(spec.get('envelope_weight',.1))*jnp.sum(ell))
