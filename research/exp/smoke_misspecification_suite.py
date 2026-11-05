"""Allocated-GPU full-model likelihood, gradient and accounting validation."""
import os
if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Full-data smoke requires TACC allocation')
import json
from pathlib import Path
import sys
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
from src.model.misspecification_suite import audit_arguments
if not any(d.platform=='gpu' for d in jax.devices()):raise RuntimeError('GPU required')
cfg=load_age_model_config();spec=cfg['_misspecification']
code=Path(ap.__file__).parents[2];sys.path.insert(0,str(code/'research/misspecification_audit'))
from contribution_budget import audit_forward
R=Path('/work/07980/bwa386/ls6/houfin/research/grid54');run=R/'runs'/spec['step']
data=load_data(cfg['input_dir'],precision='float64',verbose=False)
base=R.parents[1]/('processed/model_results/refined_float64__age_map_float32_grid54_'+spec['reference']+'_quick90')
p,_=load_map_params(str(base));lat={k:jnp.asarray(v) for k,v in auto_delta_params_to_latents(p).items()}
assets=np.load(code/'src/model/misspecification_assets.npz');T=len(assets['years']);N=assets['basis'].shape[0]
pattern=None
if spec['kind'] in ['abundance','establishment','arrival']:
 shape=(N,) if spec['mode']=='cell_static' else ((spec['n_blocks'],N) if spec['mode']=='cell_blocks' else (T,8))
 pattern=.2*jnp.cos(jnp.arange(np.prod(shape)).reshape(shape)/7.)
 lat['mm_u']=pattern
elif spec['kind']=='shooting':lat['mm_nodes_log']=jnp.asarray(assets['node_center'])+.01
keys=['alpha_a','alpha_j','alpha_f','alpha_k'];v0=jnp.asarray([lat[k] for k in keys]+[1.])
def point(v):
 p=dict(lat);p.update(zip(keys,v[:4]))
 if pattern is not None:p['mm_u']=pattern*v[-1]
 if spec['kind']=='shooting':p['mm_nodes_log']=jnp.asarray(assets['node_center'])+.01*v[-1]
 return p
def obj(v):return -log_density(ap.build_model_2d,(),dict(data=data,prior_scale=1.),point(v))[0]
value,gradient=jax.jit(jax.value_and_grad(obj))(v0)
value=float(value);gradient=np.asarray(gradient)
assert np.isfinite(value) and np.isfinite(gradient).all(),(value,gradient)
if pattern is not None or spec['kind']=='shooting':assert abs(gradient[-1])>1e-10,'disconnected discrepancy'
handlers.seed(handlers.condition(ap.build_model_2d,data=point(v0)),rng_seed=105)(data=data,prior_scale=1.)
last=ap._MM_LAST;x=audit_arguments(last);states=last['states']
arrays,_=audit_forward(af,x,states)
scale=max(float(np.max(np.asarray(states[0]))),1.)
errors={k:float(np.max(np.abs(arrays[k]))) for k in ['BIDE_error','replay_adult_error','replay_juvenile_error']}
assert max(errors.values())<1e-9*scale,errors
result=dict(passed=True,spec=spec,full_data=True,gpu=str(jax.devices()[0]),
 negative_log_density=value,gradient=gradient.tolist(),accounting_errors=errors,
 validation_scope='Nonzero discrepancy, full joint observation likelihood, derivative and replay parity. Not fit convergence.')
(run/'out/misspec_smoke.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
