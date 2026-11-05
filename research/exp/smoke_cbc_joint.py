"""Check the full joint likelihood and demographic gradient at a real frozen point."""
import json,math
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
p,_=load_map_params('/work/07980/bwa386/ls6/houfin/processed/model_results/refined_float64__age_map_float32_grid54_step35_est1951_ctl_mid_s41_quick90')
lat={k:jnp.asarray(v) for k,v in auto_delta_params_to_latents(p).items()}
lat.update(log_inv_total=jnp.asarray(math.log(560/102.8)),cbc_migrant_fraction=jnp.asarray(.3),
 cbc_young_availability=jnp.asarray(.7),cbc_log_scale=jnp.asarray(3.),
 cbc_circle_raw=jnp.zeros(int(data['cbc_n_circles'])),cbc_effort_slope=jnp.asarray(.6),
 cbc_effort_power=jnp.asarray(0.),cbc_feeder_effect=jnp.asarray(0.),cbc_feeder_missing_effect=jnp.asarray(0.),cbc_concentration=jnp.asarray(.5))
keys=['alpha_a','log_inv_total','cbc_log_scale'];z=jnp.array([lat[k] for k in keys])
def objective(v,mask):
    point=dict(lat);point.update(zip(keys,v));d=dict(data);d['cbc_train']=mask
    return -log_density(build_model_2d,(),{'data':d,'prior_scale':1.},point)[0]
value_grad=jax.jit(jax.value_and_grad(objective))
v,g=value_grad(z,data['cbc_train']);v0,g0=value_grad(z,jnp.zeros_like(data['cbc_train']))
v,g,v0,g0=jax.device_get((v,g,v0,g0))
if not(np.isfinite(v) and np.isfinite(g).all() and np.isfinite(v0) and np.isfinite(g0).all()):raise ValueError('nonfinite joint objective or gradient')
if not abs(g[1]-g0[1])>1e-6:raise ValueError('CBC does not reach the invasion abundance gradient')
out={'finite':True,'parameters_checked':keys,'joint_negative_log_density':float(v),
 'without_CBC_data_factor_negative_log_density':float(v0),'joint_gradient':g.tolist(),
 'without_CBC_data_factor_gradient':g0.tolist(),'CBC_gradient_difference':(g-g0).tolist(),
 'cbc_training_rows':int(np.asarray(data['cbc_train']).sum()),
 'scope':'Full model/code/data integration and nonzero ecological/initialization gradient, not an optimizer or scientific-validation result'}
Path('/mnt/c/Dev/houfin-range-model/.research_runtime/cbc_joint_inputs_v2/FULL_MODEL_SMOKE.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2),flush=True)
