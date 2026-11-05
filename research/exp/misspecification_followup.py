"""Matched-support full/skeleton, raw-count and correction-dependence readouts."""
import sys,json
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from numpyro import handlers
import numpyro.distributions as dist
import rasterio
from rasterio.warp import transform
from src.config_utils import load_age_model_config,load_data_config
from src.model import age_priors as ap
from src.model.data_loading import load_data
from src.model.checkpoints import load_map_params,auto_delta_params_to_latents

checkpoint=Path(sys.argv[1]);out=Path(sys.argv[2]);cfg=load_age_model_config();spec=cfg['_misspecification']
data=load_data(cfg['input_dir'],target_device=jax.devices('cpu')[0],precision='float64',verbose=False)
p,_=load_map_params(str(checkpoint));lat=auto_delta_params_to_latents(p)
def replay(disable):
 ap._MM_DISABLE=disable
 try:return handlers.trace(handlers.seed(handlers.condition(ap.build_model_2d,data=lat),rng_seed=104)).get_trace(data=data,prior_scale=1.)
 finally:ap._MM_DISABLE=False
full=replay(False);skeleton=replay(True)
lr,lc=np.asarray(data['land_rows']),np.asarray(data['land_cols']);years=np.asarray(data['years'])
with rasterio.open(load_data_config()['grid']['ref_raster']) as src:
 xx,yy=rasterio.transform.xy(src.transform,lr,lc);lon,latitude=map(np.asarray,transform(src.crs,'EPSG:4326',xx,yy))
belt=(latitude>=25)&(latitude<49)
regions=dict(east=belt&(lon>=-95),plains=belt&(lon>=-100)&(lon<-95),west=belt&(lon<-100))
lookup=np.full((data['Ny'],data['Nx']),-1,int);lookup[lr,lc]=np.arange(len(lr))
cells=lookup[np.asarray(data['obs_rows']),np.asarray(data['obs_cols'])];real=np.asarray(data['obs_obsroute_idx'])>=0
weights=np.asarray(data['obs_weight']);oy=years[np.asarray(data['obs_time_indices'])];y=jnp.asarray(data['observed_results'])
def values(tr,key):return np.asarray(tr[key]['value'])
def get_population(tr):
 for name in ['simulated_density','densities','simulated_densities']:
  if name in tr:return values(tr,name)[:,lr,lc]
 # The model wrapper captures states; deterministic names can vary by snapshot.
 raise KeyError('population deterministic absent: '+str([k for k in tr if 'dens' in k]))
records={};density={}
for label,tr in [('full',full),('skeleton',skeleton)]:
 # Use freshly captured forward outputs so names cannot silently select observations.
 fresh=replay(label=='skeleton');population=np.asarray(ap._MM_LAST['states'][0])[:,lr,lc];density[label]=population
 mu=jnp.asarray(values(fresh,'expected_obs'));conc=jnp.asarray(values(fresh,'obs_concentration'))
 nb=dist.NegativeBinomial2(mean=mu,concentration=conc);lp=np.asarray(nb.log_prob(y));lp0=np.asarray(nb.log_prob(jnp.zeros_like(y)))
 occurrence=np.where(np.asarray(y)>0,np.log(-np.expm1(np.minimum(lp0,-1e-15))),lp0)
 growth=values(fresh,'Sa_flat')+values(fresh,'Sj_flat')*values(fresh,'Fmax_flat')
 records[label]={}
 for region,mask in regions.items():
  obs=real&(cells>=0)&mask[np.maximum(cells,0)]
  early=growth[(years>=1902)&(years<=1915)][:,mask].mean();modern=growth[(years>=2010)&(years<=2025)][:,mask].mean()
  r=dict(historical_growth_change=float(modern-early),early_growth=float(early),modern_growth=float(modern),
    post1966_growth_change=float(modern-growth[(years>=1966)&(years<=1985)][:,mask].mean()),
    raw_real_record_loglik=float(lp[obs&(weights>0)].sum()),
    raw_real_record_occurrence_loglik=float(occurrence[obs&(weights>0)].sum()),
    heldout_raw_loglik=float(lp[obs&(weights==0)].sum()),n_train=int((obs&(weights>0)).sum()),n_holdout=int((obs&(weights==0)).sum()),arrivals={})
  for threshold in [.001,.01,.1]:
   hit=population[:,mask]>=threshold;first=years[np.argmax(hit,axis=0)].astype(float);first[~hit.any(0)]=np.nan
   r['arrivals'][str(threshold)]=dict(never_fraction=float(np.mean(~hit.any(0))),median_year=float(np.nanmedian(first)) if np.isfinite(first).any() else None)
  records[label][region]=r
d=values(full,'mm_diagnostics');dependence={}
for region,mask in regions.items():
 f=density['full'][:,mask];s=density['skeleton'][:,mask];active=(years>=1951)&(years<=2020)
 dependence[region]=dict(log_population_difference_rms=float(np.sqrt(np.mean((np.log1p(f)-np.log1p(s))**2))),
  annual_abundance_log_multiplier_rms=float(np.sqrt(np.mean(d[active,2][:,mask]**2))),
  establishment_log_multiplier_rms=float(np.sqrt(np.mean(d[active,3][:,mask]**2))),
  node_net_adjustment=float(d[:,5][:,mask].sum()),node_gross_adjustment=float(np.abs(d[:,5][:,mask]).sum()))
np.savez_compressed(out/'misspecification_population.npz',full=density['full'],skeleton=density['skeleton'],years=years,land_rows=lr,land_cols=lc,diagnostics=d)
result=dict(spec=spec,regions=records,dependence=dependence,
 thresholds='Fixed internal density units, identical across all fits; not observed first-arrival dates.',
 scope='Point estimates. Raw count scores undo hurdle reweighting for comparison. Skeleton has every discrepancy removed without refitting.',
 missing=['Paired predictive uncertainty','Dynamic-gate isolation','Finite-propagule establishment','HMC geometry'])
(out/'misspecification_followup.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
