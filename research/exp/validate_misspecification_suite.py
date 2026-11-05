"""Small-grid numerical checks only; full-data GPU smoke is a separate queue gate."""
import os
os.environ.update(JAX_PLATFORMS='cpu',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
from pathlib import Path
import json
import sys
import inspect
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from src.model import age_forward as af
from misspecification_suite import build_forward,bounded_log,ar1,audit_arguments
from stage_overflow_early import sha
R=Path('/work/07980/bwa386/ls6/houfin/research/grid54')
af.FFT_PAD_VALUE=0.;af.FFT_FLOOR_ULPS=None
N=6;T=4;rr=jnp.repeat(jnp.arange(2),3);cc=jnp.tile(jnp.arange(3),2)
adult=jnp.ones((4,6),dtype=jnp.complex128)
juv=jnp.tile(adult[None],(4,1,1))/4.
x=dict(Sa_flat=jnp.full((T,N),.6),Sj_flat=jnp.full((T,N),.3),Fmax_flat=jnp.full((T,N),3.),
 K_flat=jnp.full((T,N),10.),c_flat=jnp.full((T,N),.2),Q_flat=jnp.ones((T,N,4)),
 land_rows=rr,land_cols=cc,land_mask=jnp.ones((2,3)),adult_fft_kernel=adult,juvenile_fft_kernel_stack=juv,
 adult_edge_correction=jnp.ones((2,3)),juvenile_edge_correction_stack=jnp.ones((4,2,3)),
 initpop_latent=jnp.arange(1,7,dtype=float).reshape(2,3),dispersal_random=jnp.zeros(T),inv_pop=jnp.zeros((1,1)),
 time=T,inv_locations=jnp.array([[0,0]]),inv_timestep=1,dispersal_logit_intercept=jnp.array(-1.),
 dispersal_logit_slope=jnp.array(1.),allee_gamma=jnp.array(1.),yearling_breeding=True,veit_departure=True)
base=af.forward_sim_age_structured(**x)
checks={}
sys.path.insert(0,str(Path(af.__file__).parents[2]/'research/misspecification_audit'))
from contribution_budget import audit_forward
def accounting(mm,out):
 bound=inspect.signature(af.forward_sim_age_structured).bind(**x);bound.apply_defaults()
 args=audit_arguments(dict(args=bound.arguments,diagnostics=mm['diagnostics']))
 arrays,_=audit_forward(af,args,out)
 errors={k:float(np.max(np.abs(arrays[k]))) for k in ['BIDE_error','replay_adult_error','replay_juvenile_error']}
 assert max(errors.values())<1e-10,errors
 return errors
for kind in ['control','abundance','establishment','arrival']:
 mm=dict(kind=kind,established_only=True,clock=kind=='establishment')
 if kind!='control':mm['field']=jnp.zeros((T,N))
 fn=build_forward(af,mm);out=fn(**x)
 for a,b in zip(base,out):np.testing.assert_allclose(a,b,atol=1e-12,rtol=1e-12)
 checks[kind+'_zero_identity']=True
 if kind!='control':
  def loss(v):
   mm['field']=jnp.ones((T,N))*v
   return jnp.sum(fn(**x)[0])
  value,g=jax.value_and_grad(loss)(jnp.array(.2))
  assert np.isfinite(value) and np.isfinite(g)
  if kind!='arrival':assert abs(float(g))>1e-8
  checks[kind+'_finite_nonzero_probe']=dict(value=float(value),gradient=float(g))
  if kind=='arrival':
   mm['field']=jnp.arange(T*N,dtype=float).reshape(T,N)/30
   fn(**x);w=np.asarray(mm['diagnostics'][:,4]);assert np.isfinite(w).all()
   checks['arrival_weights_finite']=True
  mm['field']=jnp.arange(T*N,dtype=float).reshape(T,N)/30
  out=fn(**x);checks[kind+'_accounting']=accounting(mm,out)

mm=dict(kind='shooting',node_index=jnp.array([-1,-1,0,-1]),
 nodes=jnp.log(jnp.stack([base[1][1][rr,cc],base[2][1][rr,cc]])[None]))
fn=build_forward(af,mm);out=fn(**x)
for a,b in zip(base,out):np.testing.assert_allclose(a,b,atol=1e-12,rtol=1e-12)
checks['shooting_exact_continuity_identity']=True
def after(init):
 xx=dict(x,initpop_latent=init)
 return fn(**xx)[0][2:].sum()
g=jax.grad(after)(x['initpop_latent']);np.testing.assert_allclose(g,0.,atol=1e-12)
checks['shooting_cuts_trajectory_gradient']=True
mm['nodes']=mm['nodes']+.1
out=fn(**x);checks['shooting_accounting']=accounting(mm,out)
v=np.array([-1e5,0,1e5]);b=bounded_log(v,.2)
assert b[1]==0 and np.max(np.abs(b))<=2
np.testing.assert_allclose(ar1(np.array([[1.],[0.],[0.]]),.9).ravel(),[1,.9,.81])
checks['bounded_identity_and_ar1']=True

# Likelihood decomposition and replacement, including zeros.
import numpyro.distributions as dist
y=jnp.array([0.,1.,5.]);nb=dist.NegativeBinomial2(mean=jnp.array([.2,2.,4.]),concentration=2.)
lp=nb.log_prob(y);lp0=nb.log_prob(jnp.zeros_like(y));occ=jnp.where(y>0,jnp.log(-jnp.expm1(lp0)),lp0)
count=jnp.where(y>0,lp-occ,0.)
np.testing.assert_allclose(occ+count,lp,atol=1e-12)
np.testing.assert_allclose(lp-.75*count,occ+.25*count,atol=1e-12)
checks['hurdle_exact_decomposition_and_replacement']=True

# Exercise the NumPyro wrapper itself, including external capture hooks and
# cancellation of the initialization-only shooting proposal density.
import tempfile
import numpyro
from numpyro import handlers
from numpyro.infer.util import log_density
from misspecification_suite import install
with tempfile.TemporaryDirectory(prefix='houfin_mm_validation_') as temp:
 asset=Path(temp)/'assets.npz'
 node_center=np.log(np.stack([np.asarray(base[1][1][rr,cc]),np.asarray(base[2][1][rr,cc])])[None])
 np.savez(asset,years=np.arange(T),basis=np.ones((N,2))/2**.5,active=np.ones(T),
   node_center=node_center,node_times=np.array([2]),node_index=np.array([-1,-1,0,-1]))
 for kind in ['control','abundance','establishment','arrival','shooting']:
  ns={'forward_sim_age_structured':af.forward_sim_age_structured}
  def simple(data,prior_scale=1.):
   a=numpyro.sample('a',dist.Normal(0.,1.))
   xx=dict(x,Fmax_flat=x['Fmax_flat']*jnp.exp(a))
   result=ns['forward_sim_age_structured'](**xx)
   mu=jnp.maximum(result[0].reshape(-1),1e-6)
   numpyro.deterministic('expected_obs',mu)
   numpyro.deterministic('obs_concentration',jnp.full_like(mu,2.))
   with handlers.mask(mask=data['obs_weight']>0),handlers.scale(scale=jnp.where(data['obs_weight']>0,data['obs_weight'],1.)):
    numpyro.sample('obs',dist.NegativeBinomial2(mean=mu,concentration=2.),obs=data['observed_results'])
  spec=dict(kind=kind,mode='cell_static',sigma=.15,clock=kind=='establishment',positive_count_weight=.25,established_only=True)
  model=install(simple,ns,spec,asset)
  corrected=ns['forward_sim_age_structured'];seen=[]
  def external(*args,**kw):
   result=corrected(*args,**kw);seen.append(result);return result
  ns['forward_sim_age_structured']=external
  small=dict(observed_results=jnp.tile(jnp.array([0.,1.,2.]),8),obs_weight=jnp.ones(T*N).at[-1].set(0.),obs_obsroute_idx=jnp.arange(T*N).at[0].set(-1))
  point={'a':jnp.array(.1)}
  if kind in ['abundance','establishment','arrival']:point['mm_u']=jnp.full(N,.2)
  if kind=='shooting':point['mm_nodes_log']=jnp.asarray(node_center)+.01
  val,grad=jax.value_and_grad(lambda a:-log_density(model,(),dict(data=small),dict(point,a=a))[0])(jnp.array(.1))
  assert np.isfinite(val) and np.isfinite(grad) and seen
  # A fresh trace must expose materialized diagnostics and the external hook.
  seen.clear();tr=handlers.trace(handlers.seed(handlers.condition(model,data=point),rng_seed=0)).get_trace(data=small)
  assert len(seen)==1 and np.isfinite(np.asarray(tr['mm_diagnostics']['value'])).all()
  if kind=='shooting':assert np.isfinite(np.asarray(tr['mm_node_log_residual']['value'])).all()
  checks[kind+'_numpyro_wrapper_and_capture']=dict(value=float(val),gradient=float(grad))
specs=json.loads((R/'monitor/misspecification_20261104_specs.json').read_text())
manifests={}
for spec in specs:
 run=R/'runs'/spec['step'];manifest=run/'CODE_MANIFEST.sha256'
 for line in manifest.read_text().splitlines():
  h,rel=line.split('  ',1);assert sha(run/'code'/rel)==h
 manifests[spec['step']]=sha(manifest)
result=dict(passed=True,scope='Small-grid identities and gradients; full-data smoke deferred to each allocated GPU task.',checks=checks,manifests=manifests)
(R/'monitor/misspecification_20261104_VALIDATED.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
