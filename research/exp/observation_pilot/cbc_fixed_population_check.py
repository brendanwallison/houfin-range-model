"""Numerical/calibration check of CBC source parameters at a frozen population."""
import json,time
from pathlib import Path
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
import numpyro
from numpyro import handlers
from numpyro.infer.util import initialize_model
from numpyro.infer.initialization import init_to_value
from scipy.optimize import minimize
from scipy.stats import nbinom
from cbc_joint import cbc_observe,winter_readout

repo=Path('/mnt/c/Dev/houfin-range-model');out=repo/'.research_runtime/cbc_joint_inputs_v2'
with np.load(out/'cbc_joint_300km.npz') as z:d={k:jnp.asarray(z[k]) for k in z.files}
d['cbc_n_circles']=int(d['cbc_w_local'].shape[0])
with np.load('/mnt/e/Datasets/houfin/contribution_20261101_v3/step35_est1951_ctl_mid_s41/annual_budget.npz') as z:
    rr,cc=z['land_rows'],z['land_cols'];a=z['next_adult'];j=z['next_juvenile'];years=z['years']
d.update(land_rows=jnp.asarray(rr),land_cols=jnp.asarray(cc))
shape=(len(years),int(rr.max()+1),int(cc.max()+1));A=jnp.zeros(shape).at[:,rr,cc].set(a);J=jnp.zeros(shape).at[:,rr,cc].set(j)
import pickle
m=pickle.load(open('/work/07980/bwa386/ls6/houfin/research/grid54/inputs/established1951/metadata.pkl','rb'))
scale=float(m['pop_scalar'])
def model():cbc_observe(d,A,J,scale,{'circle_sd':1.})
density=np.asarray(winter_readout(jnp.asarray(a),jnp.asarray(j),d,.3,.7))*scale
keep=np.asarray(d['cbc_train'])>0;y=np.asarray(d['cbc_count'])
initial={'cbc_log_scale':float(np.log(max(y[keep].mean(),1e-8)/max(density[keep].mean(),1e-8))),
 'cbc_circle_raw':jnp.zeros(d['cbc_n_circles']),'cbc_migrant_fraction':.3,'cbc_young_availability':.7,
 'cbc_effort_slope':.6,'cbc_effort_power':0.,'cbc_feeder_effect':0.,'cbc_feeder_missing_effect':0.,'cbc_concentration':.5}
info=initialize_model(jax.random.PRNGKey(41),model,init_strategy=init_to_value(values=initial))
z0,unravel=ravel_pytree(info.param_info.z)
fun=jax.jit(jax.value_and_grad(lambda z:info.potential_fn(unravel(z))))
def objective(z):
    v,g=fun(jnp.asarray(z));return float(v),np.asarray(g,float)
start=time.time();v0,g0=objective(np.asarray(z0));print('FIXED CBC INITIAL',v0,'gradmax',np.max(np.abs(g0)),flush=True)
if not(np.isfinite(v0) and np.isfinite(g0).all()):raise ValueError('nonfinite initial CBC objective')
res=minimize(objective,np.asarray(z0),jac=True,method='L-BFGS-B',options={'maxiter':2400,'ftol':1e-11,'gtol':1e-4,'maxls':40})
values=info.postprocess_fn(unravel(jnp.asarray(res.x)))
trace=handlers.trace(handlers.substitute(model,data=values)).get_trace()
mu=np.asarray(trace['cbc_expected']['value']);k=float(values['cbc_concentration']);ll=nbinom.logpmf(y,k,k/(k+mu))
report={'scope':'Source-parameter calibration conditional on one frozen population, not a demographic refit or acceptance gate',
 'success':bool(res.success),'message':str(res.message),'iterations':int(res.nit),'elapsed_seconds':time.time()-start,
 'initial_potential':v0,'final_potential':float(res.fun),'gradient_max':float(np.abs(res.jac).max()),
 'finite':bool(np.isfinite(res.fun) and np.isfinite(res.jac).all() and np.isfinite(mu).all()),
 'parameters':{k:np.asarray(v).tolist() for k,v in values.items() if k.startswith('cbc_') and np.asarray(v).ndim==0 and not k.startswith('cbc_loglik')},'scores':{}}
for name,mask in [('train',keep),('spatial',np.asarray(d['cbc_spatial_holdout'])>0),('temporal',np.asarray(d['cbc_temporal_holdout'])>0)]:
    report['scores'][name]={'n':int(mask.sum()),'mean_log_score':float(ll[mask].mean()),'observed_mean':float(y[mask].mean()),'predicted_mean':float(mu[mask].mean())}
(out/'FIXED_POPULATION_CHECK_REFINED.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(report,indent=2),flush=True)
