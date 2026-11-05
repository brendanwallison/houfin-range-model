"""Measure the actual fitted departure/crowding mismatch, including disease."""
import inspect,json,os
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from numpyro import handlers
import rasterio
from rasterio.warp import transform
from src.config_utils import load_age_model_config
from src.model import age_priors as ap,age_forward as af
from src.model.data_loading import load_data
from src.model.checkpoints import load_map_params,auto_delta_params_to_latents

step=os.environ['STEP'];cfg=load_age_model_config();data=load_data(cfg['input_dir'],precision='float64',verbose=False)
params,_=load_map_params('/work/07980/bwa386/ls6/houfin/processed/model_results/refined_float64__age_map_float32_grid54_'+step+'_quick90')
lat=auto_delta_params_to_latents(params);captured=[];old=ap.forward_sim_age_structured
def record(*a,**kw):
    bound=inspect.signature(old).bind(*a,**kw);bound.apply_defaults();out=old(*a,**kw);captured.append((bound.arguments,out));return out
ap.forward_sim_age_structured=record
try:handlers.seed(handlers.condition(ap.build_model_2d,data=lat),rng_seed=104)(data=data,prior_scale=1.)
finally:ap.forward_sim_age_structured=old
x,states=captured[0];rr,cc=map(np.asarray,[data['land_rows'],data['land_cols']]);years=np.asarray(data['years'])
if not x.get('yearling_breeding') or x.get('veit_reproduction') or x.get('allee_relative'):
    raise NotImplementedError('Closed-cell calculation supports standard yearling breeding with absolute Allee only')
with rasterio.open('/work/07980/bwa386/ls6/houfin/processed/grid54/ref_grid_54km.tif') as grid:
    xx,yy=rasterio.transform.xy(grid.transform,rr,cc);lon,latgrid=map(np.asarray,transform(grid.crs,'EPSG:4326',xx,yy))
regions=dict(east=(lon>=-80)&(latgrid>=25)&(latgrid<49),plains=(lon>=-100)&(lon<-95)&(latgrid>=25)&(latgrid<49),west=(lon<-100)&(latgrid>=25)&(latgrid<49))
na,nj=np.asarray(states[1])[:,rr,cc],np.asarray(states[2])[:,rr,cc]
initial=np.asarray(x['initpop_latent'])[rr,cc]
aa=np.concatenate([initial[None]*.5,na[:-1]]);jj=np.concatenate([initial[None]*.5,nj[:-1]])
for i,loc in enumerate(np.asarray(x['inv_locations'])):
    cell=np.flatnonzero((rr==loc[0])&(cc==loc[1]))[0]
    start=int(x['inv_timestep']);pulse=np.asarray(x['inv_pop'])[i]
    aa[start:start+len(pulse),cell]+=pulse*.5;jj[start:start+len(pulse),cell]+=pulse*.5
k=np.asarray(x['K_flat']);ratio=(aa+jj)/np.maximum(k,1e-6)
if getattr(af,'OVERFLOW_ENABLED',False):pa,pj=map(np.asarray,af.overflow_probabilities(jnp.asarray(aa),jnp.asarray(jj),jnp.maximum(jnp.asarray(k),1e-6)))
else:
    pv=np.clip(ratio-np.logaddexp(0,10*(ratio-1))/10,1e-6,1.);pa=pj=pv
depart=aa*pa+jj*pj;n=aa+jj
result=dict(step=step,scope='Actual pre-departure pools; population-weighted census and attempt rates, not successful off-cell movement',eras={})
def weighted(v,w):
    use=np.isfinite(v)&np.isfinite(w)&(w>0)
    if not use.any():return None
    order=np.argsort(v[use]);a=v[use][order];b=w[use][order];c=np.cumsum(b)/b.sum()
    return dict(mean=float(np.average(a,weights=b)),q10_50_90=np.interp([.1,.5,.9],c,a).tolist())
for era,(lo,hi) in dict(pre_disease=(1970,1989),modern=(2010,2025)).items():
    t=(years>=lo)&(years<=hi);result['eras'][era]={}
    for reg,mask in regions.items():
        ix=np.ix_(t,mask);result['eras'][era][reg]=dict(N_over_K=weighted(ratio[ix],n[ix]),
          attempted_departure_fraction=float(depart[ix].sum()/n[ix].sum()))
# At2020, evaluate closed-cell growth at K with and without the fitted additional disease loss.
t=int(np.flatnonzero(years==2020)[0]);sa=np.asarray(x['Sa_flat'])[t];sj=np.asarray(x['Sj_flat'])[t];f=np.asarray(x['Fmax_flat'])[t];cap=k[t];crowd=np.asarray(x['c_flat'])[t]
amp=np.zeros_like(cap) if x.get('disease_m_amp') is None else np.asarray(x['disease_m_amp'])[t]
hill=float(x.get('disease_hill_n',1.));half=float(x.get('disease_k_half',1.));gamma=float(x['allee_gamma'])
agg=af.ALLEE_AGG_K
def local_growth(density,disease):
    m=1-np.exp(-gamma*density) if agg is None else -np.expm1(-float(agg)*np.log1p(gamma*density/float(agg)))
    loss=amp*(density+1e-12)**hill/((density+1e-12)**hill+half**hill) if disease else 0.
    return (1-loss)*(sa+sj*f*m/(1+crowd*density/np.maximum(cap,1e-12)))
curves=np.geomspace(1e-6,4,600);equilibria={}
for disease in [False,True]:
    growth=local_growth(curves[:,None]*cap[None],disease)
    crossings=(growth[:-1]>1)&(growth[1:]<=1)
    roots=np.full(len(cap),np.nan)
    for cell in np.flatnonzero(crossings.any(0)):
        q=np.flatnonzero(crossings[:,cell])[-1];a,b=curves[q:q+2];ga,gb=growth[q:q+2,cell]
        roots[cell]=a+(b-a)*(ga-1)/(ga-gb)
    equilibria[str(disease)]=roots
result['closed_cell_2020']={reg:dict(growth_at_K_without_disease=weighted(local_growth(cap,False)[mask],n[t,mask]),
    growth_at_K_with_disease=weighted(local_growth(cap,True)[mask],n[t,mask]),
    equilibrium_over_K_without_disease=weighted(equilibria['False'][mask],n[t,mask]),
    equilibrium_over_K_with_disease=weighted(equilibria['True'][mask],n[t,mask])) for reg,mask in regions.items()}
out=Path('/mnt/c/Dev/houfin-range-model/research/experiments')/f'RETENTION_MECHANISM_{step}_20261102.json';out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
