"""Separate CBC and remaining likelihood/prior gradients at a completed joint MAP."""
import json
from pathlib import Path
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from numpyro.infer.util import log_density
from src.config_utils import load_age_model_config
from src.model.age_priors import build_model_2d
from src.model.data_loading import load_data
from src.model.checkpoints import load_map_params,auto_delta_params_to_latents

cfg=load_age_model_config();data=load_data(cfg['input_dir'],precision='float64',verbose=False)
p,_=load_map_params('/work/07980/bwa386/ls6/houfin/processed/model_results/refined_float64__age_map_float32_grid54_step38_cbc1951_free_s41_quick90')
lat={k:jnp.asarray(v) for k,v in auto_delta_params_to_latents(p).items()}
keys=['alpha_a','alpha_j','alpha_f','log_inv_total'];z=jnp.array([lat[k] for k in keys])
dk=[k for k,v in data.items() if isinstance(v,jax.Array)];static={k:v for k,v in data.items() if k not in dk}
lk=list(lat);dv=tuple(data[k] for k in dk);lv=tuple(lat[k] for k in lk)
def losses(v,arrays,parameters):
    d=dict(static);d.update(zip(dk,arrays));point=dict(zip(lk,parameters));point.update(zip(keys,v))
    lp,tr=log_density(build_model_2d,(),{'data':d,'prior_scale':1.},point)
    c=-tr['cbc_loglik_train']['value']
    return jnp.stack([c,-lp-c])
values=np.asarray(jax.jit(losses)(z,dv,lv))
g=np.asarray(jax.jit(jax.jacrev(losses,argnums=0))(z,dv,lv))
if not np.isfinite(g).all():raise ValueError('nonfinite likelihood decomposition')
out={'point':'step38_cbc1951_free_s41','parameter_order':keys,'CBC_negative_loglik':float(values[0]),
 'remaining_negative_logdensity':float(values[1]),'CBC_gradient':g[0].tolist(),'remaining_gradient':g[1].tolist(),
 'total_gradient':g.sum(0).tolist(),
 'interpretation':'Positive loss derivative means that likelihood alone favors lowering the named raw parameter, holding all other parameters fixed. Local directional pressure is not a global causal explanation or a profile likelihood.'}
path=Path('/mnt/c/Dev/houfin-range-model/research/experiments/CBC_LIKELIHOOD_PRESSURE_20261102.json')
path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)
