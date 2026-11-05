"""Full-data likelihood/gradient and frozen-movement accounting integration check."""
import inspect,json,math,os
from pathlib import Path
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from numpyro import handlers
from numpyro.infer.util import log_density
from src.config_utils import load_age_model_config
from src.model import age_priors as ap,age_forward as af
from src.model.data_loading import load_data
from src.model.checkpoints import load_map_params,auto_delta_params_to_latents
from contribution_budget import audit_forward

cfg=load_age_model_config();data=load_data(cfg['input_dir'],precision='float64',verbose=False)
parent='/work/07980/bwa386/ls6/houfin/processed/model_results/refined_float64__age_map_float32_grid54_step38_cbc1951_noncbc_free_s41_quick90'
p,_=load_map_params(parent);lat={k:jnp.asarray(v) for k,v in auto_delta_params_to_latents(p).items()}
lat.update({k:jnp.asarray(v) for k,v in cfg['population_model']['pinned_sites'].items()})
if cfg['population_model']['cbc_likelihood']['enabled']:
    lat.update(cbc_log_scale=jnp.asarray(0.),cbc_circle_raw=jnp.zeros(int(data['cbc_n_circles'])),
       cbc_effort_slope=jnp.asarray(.6),cbc_effort_power=jnp.asarray(0.),
       cbc_effort_missing_effect=jnp.asarray(0.),cbc_feeder_effect=jnp.asarray(0.))
keys=['alpha_a','alpha_j','alpha_f','alpha_k','log_inv_total']
z=jnp.array([lat[k] for k in keys])
def objective(v):
    point=dict(lat);point.update(zip(keys,v))
    return -log_density(ap.build_model_2d,(),dict(data=data,prior_scale=1.),point)[0]
value,grad=jax.jit(jax.value_and_grad(objective))(z)
value,grad=jax.device_get((value,grad))
if not(np.isfinite(value) and np.isfinite(grad).all()):raise ValueError('nonfinite full-model objective/gradient')
captured=[];original=ap.forward_sim_age_structured
def record(*args,**kw):
    bound=inspect.signature(original).bind(*args,**kw);bound.apply_defaults()
    states=original(*args,**kw);captured.append((bound.arguments,states));return states
ap.forward_sim_age_structured=record
try:trace=handlers.trace(handlers.seed(handlers.condition(ap.build_model_2d,data=lat),rng_seed=104)).get_trace(data=data,prior_scale=1.)
finally:ap.forward_sim_age_structured=original
if len(captured)!=1:raise ValueError('ambiguous forward')
x,states=captured[0];arrays,_=audit_forward(af,x,states)
errors={n:float(np.max(np.abs(arrays[n]))) for n in ['BIDE_error','replay_adult_error','replay_juvenile_error']}
scale=max(float(np.max(np.asarray(states[0]))),1.)
if max(errors.values())>1e-9*scale:raise ValueError('frozen movement accounting mismatch '+str(errors))
floor=cfg['population_model'].get('capacity_floor_internal',0.)
if float(np.min(np.asarray(x['K_flat'])))+1e-12<floor:raise ValueError('capacity floor violated')
out=dict(config=os.environ['AGE_MODEL_CONFIG'],finite=True,negative_log_density=float(value),
    parameters=keys,gradient=grad.tolist(),accounting_errors=errors,capacity_floor=floor,
    minimum_K=float(np.min(np.asarray(x['K_flat']))))
if cfg['population_model'].get('historical_range',{}).get('enabled'):
    from src.model.historical_range import zero_loss
    oldloss=jnp.sum(data['hr_old_zero_weight']*zero_loss(trace['expected_obs']['value'],trace['obs_concentration']['value']))
    out['matched_reference_trajectory']=dict(original_pseudozero_loss=float(oldloss),
        derived_range_absence_loss=float(trace['historical_range_absence_loss']['value']),
        derived_range_presence_loss=float(trace['historical_range_presence_loss']['value']),
        old_pseudozero_weight_remaining=float(jnp.sum(data['obs_weight']*data['hr_old_zero_mask'])))
    if out['matched_reference_trajectory']['old_pseudozero_weight_remaining']!=0:raise ValueError('old pseudozeros double counted')
    np.savez_compressed(Path(os.environ['SMOKE_OUT']).with_suffix('.npz'),
        population=(arrays['next_adult']+arrays['next_juvenile'])*float(data['pop_scalar']),
        young=arrays['next_juvenile']*float(data['pop_scalar']),
        years=np.asarray(data['years']),land_rows=np.asarray(data['land_rows']),land_cols=np.asarray(data['land_cols']),
        old_loss_by_observation=np.asarray(data['hr_old_zero_weight']*zero_loss(trace['expected_obs']['value'],trace['obs_concentration']['value'])),
        obs_rows=np.asarray(data['obs_rows']),obs_cols=np.asarray(data['obs_cols']))
Path(os.environ['SMOKE_OUT']).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
