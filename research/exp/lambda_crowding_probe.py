"""Read-only density-response diagnostic of frozen yearling fits; no refitting."""
import inspect,json,os
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
from numpyro import handlers
import rasterio
from rasterio.warp import transform
from src.config_utils import load_age_model_config
from src.model import age_priors as ap,age_forward as af
from src.model.data_loading import load_data
from src.model.checkpoints import load_map_params,auto_delta_params_to_latents

step=os.environ['STEP'];cfg=load_age_model_config();data=load_data(cfg['input_dir'],precision='float64',verbose=False)
p,_=load_map_params('/work/07980/bwa386/ls6/houfin/processed/model_results/refined_float64__age_map_float32_grid54_'+step+'_quick90')
lat=auto_delta_params_to_latents(p);capture=[];original=ap.forward_sim_age_structured
def record(*a,**kw):
 b=inspect.signature(original).bind(*a,**kw);b.apply_defaults();states=original(*a,**kw);capture.append((b.arguments,states));return states
ap.forward_sim_age_structured=record
try:handlers.seed(handlers.condition(ap.build_model_2d,data=lat),rng_seed=104)(data=data,prior_scale=1.)
finally:ap.forward_sim_age_structured=original
x,states=capture[0]
assert x['yearling_breeding'] and not x.get('veit_reproduction') and not x.get('allee_relative')
rr,cc=map(np.asarray,[data['land_rows'],data['land_cols']]);years=np.asarray(data['years'])
with rasterio.open('/work/07980/bwa386/ls6/houfin/processed/grid54/ref_grid_54km.tif') as grid:
 xx,yy=rasterio.transform.xy(grid.transform,rr,cc);lon,la=map(np.asarray,transform(grid.crs,'EPSG:4326',xx,yy))
regions=dict(east=(lon>=-80)&(la>=25)&(la<49),plains=(lon>=-100)&(lon<-95)&(la>=25)&(la<49),west=(lon<-100)&(la>=25)&(la<49))
oy=years[np.asarray(data['obs_time_indices'],int)];positive=(np.asarray(data['observed_results'])>0)&(oy>=2010)&(oy<=2025)
observed=np.zeros(states[0].shape[1:],bool);observed[np.asarray(data['obs_rows'])[positive],np.asarray(data['obs_cols'])[positive]]=True
observed=observed[rr,cc]
budget=Path('/mnt/e/Datasets/houfin/contribution_20261101_v3')/step/'annual_budget.npz'
with np.load(budget) as z:
 apool,jstay,jarr,localSa=[z[k] for k in ['adult_post','juvenile_stay','juvenile_arrive','local_adult_survival']]
 actualF=z['realized_offspring_per_breeder'];weights=z['next_breeders'];saved_potential=z['potential_lambda']
sa,sj,f,k,c=[np.asarray(x[n]) for n in ['Sa_flat','Sj_flat','Fmax_flat','K_flat','c_flat']]
np.testing.assert_allclose(sa+sj*f,saved_potential,rtol=1e-10,atol=1e-12)
dm=np.divide(localSa,sa,out=np.ones_like(sa),where=sa>0)
density=apool+jstay+jarr*dm
ratio=density/np.maximum(k,1e-12)
mate=np.asarray(af.allee_mated_fraction(x['allee_gamma']*density))
replayed=f/(1+c*ratio)*mate
valid=weights>1e-12
np.testing.assert_allclose(replayed[valid],actualF[valid],rtol=1e-6,atol=1e-8)
elastic=np.where(c>0,(1-ratio)/(1+c*ratio),1.)
def summary(v,w):
 keep=np.isfinite(v)&np.isfinite(w)&(w>0)
 if not keep.any():return None
 v,w=v[keep],w[keep];order=np.argsort(v);v,w=v[order],w[order]
 return dict(mean=float(np.average(v,weights=w)),q10_50_90=np.interp([.1,.5,.9],np.cumsum(w)/w.sum(),v).tolist())
result=dict(step=step,scope='Frozen fitted-density fecundity sensitivity, not a refitted alternative or posterior',regions={})
for region,mask in regions.items():
 ix=np.ix_((years>=2010)&(years<=2025),mask)
 result['regions'][region]=dict(reproductive_N_over_K=summary(ratio[ix],weights[ix]),
  partial_log_fecundity_elasticity=summary(elastic[ix],weights[ix]),
  offspring_remaining_fraction=summary(actualF[ix]/np.maximum(f[ix],1e-12),weights[ix]),
  breeder_share_elasticity_below_point2=float(weights[ix][elastic[ix]<.2].sum()/weights[ix].sum()))
# At common observed cells, compare unconstrained potential with the largest frozen-density
# stage-projection multiplier attainable when BOTH mate finding and crowding are applied.
# This is not the density-feedback Jacobian and excludes movement. Disease retained below.
t=int(np.flatnonzero(years==2020)[0]);gamma=float(x['allee_gamma']);cap=np.maximum(k[t],1e-12)
low=np.log(cap)-np.log(1e8);high=np.log(np.maximum(cap,1/gamma))+np.log(100.)
amp=np.zeros_like(k[t]) if x.get('disease_m_amp') is None else np.asarray(x['disease_m_amp'])[t]
best_off=np.zeros_like(k[t]);best_on=np.zeros_like(k[t])
for q in np.linspace(0,1,2500):
 n=np.exp(low+(high-low)*q)
 agg=af.ALLEE_AGG_K
 m=-np.expm1(-gamma*n) if agg is None else -np.expm1(-float(agg)*np.log1p(gamma*n/float(agg)))
 g=sa[t]+sj[t]*f[t]*m/(1+c[t]*n/cap)
 hill=(n+1e-12)**float(x.get('disease_hill_n',1.));d=1-amp*hill/(hill+float(x.get('disease_k_half',1.))**float(x.get('disease_hill_n',1.)))
 best_off=np.maximum(best_off,g);best_on=np.maximum(best_on,g*d)
for region,mask in regions.items():
 keep=mask&observed
 result['regions'][region]['common_positive_cells_2020']=dict(n=int(keep.sum()),
  potential=summary((sa[t]+sj[t]*f[t])[keep],np.ones(keep.sum())),
  maximum_with_mates_and_crowding=summary(best_off[keep],np.ones(keep.sum())),
  maximum_with_mates_crowding_disease=summary(best_on[keep],np.ones(keep.sum())))
out=Path('/mnt/c/Dev/houfin-range-model/research/experiments/lambda_history_review_20261102')/f'CROWDING_PROBE_{step}.json'
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
