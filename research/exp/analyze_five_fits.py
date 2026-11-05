"""Read-only matched comparison of the five completed flexibility fits on TACC."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import json,sys,time,hashlib
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import transform
from scipy.ndimage import label,binary_dilation,binary_propagation,generate_binary_structure
from scipy.stats import spearmanr
R=Path('/work/07980/bwa386/ls6/houfin/research/grid54');sys.path.insert(0,str(R/'monitor'))
from range_watch import HostUnpickler
STEPS=['step39_kfloor4p0_s41','step39_kfloor4p0_s42','step44_shared_s43','step44_kprivate10_s41','step44_kprivate25_s41','step44_allrho99_s41','step45_regional_s010_s41']
ERAS={'early':(1902,1915),'invasion':(1951,1965),'bbs_early':(1966,1985),'middle':(1986,2009),'modern':(2010,2025)}
ratio=lambda n,d:np.divide(n,d,out=np.full_like(n,np.nan,dtype=float),where=d>1e-20)
def finite_mean(v):
 v=np.asarray(v);v=v[np.isfinite(v)];return float(v.mean()) if v.size else None

def classify(v,w):
 keep=w&np.isfinite(v);v=v[keep]
 return dict(cells=int(len(v)),mean=finite_mean(v),q10_50_90=np.quantile(v,[.1,.5,.9]).tolist() if len(v) else None,
             sink_share_strict=float(np.mean(v<1)) if len(v) else None,sink_share_005=float(np.mean(v<.95)) if len(v) else None,source_share_005=float(np.mean(v>1.05)) if len(v) else None)

def connected(v,threshold,structure,allowed,west,east):
 active=np.isfinite(v)&(v>threshold)&allowed;grid=np.zeros(shape,bool);grid[rr,cc]=active
 labs,_=label(grid,structure=structure);labs=labs[rr,cc]
 left=np.unique(labs[active&west]);right=np.unique(labs[active&east]);return bool(np.intersect1d(left[left>0],right[right>0]).size)

metadata=R/'inputs/established1951/metadata.pkl'
with metadata.open('rb') as f:meta=HostUnpickler(f).load()
records=[];cache={};gridref=None
for step in STEPS:
 run=R/'runs'/step;out=run/'out';assert (out/'FIT_DONE').exists() and (out/'contribution/READY.json').exists()
 cfg=json.loads((run/f'code/config/overlays/grid54_{step}.json').read_text());pop=cfg['population_model']
 assert pop['reproduction']['yearling_breeding'] and not pop['cbc_likelihood']['enabled']
 pathology=json.loads((out/'pathology.json').read_text());metrics=json.loads((out/'viz_ref/metrics.json').read_text());budget=json.loads((out/'contribution/READY.json').read_text());front=json.loads((out/'front_speed.json').read_text())
 with np.load(out/'contribution/annual_budget.npz') as z:
  years,rr,cc=z['years'],z['land_rows'],z['land_cols'];a,j=z['input_adult'],z['input_juvenile'];state=a+j
  names=['potential_lambda','local_adult_survival','local_juvenile_stayer_survival','breeder_contribution','local_total_replacement','successful_breeder_export','immigrant_breeders','next_breeders','retained_breeders','realized_offspring_per_breeder','origin_adult_survival','origin_juvenile_recruitment']
  arrays={n:z[n] for n in names}
  arrays['biological_C']=z['biological_breeder_contribution'] if 'biological_breeder_contribution' in z else arrays['breeder_contribution']
  arrays['full_lambda']=z['potential_lambda_with_process'] if 'potential_lambda_with_process' in z else arrays['potential_lambda']
  arrays['process_multiplier']=z['process_multiplier'] if 'process_multiplier' in z else np.ones_like(a)
  arrays['process_adjustment']=z['process_adjustment'] if 'process_adjustment' in z else np.zeros_like(a)
 if gridref is None:
  gridref=(years.copy(),rr.copy(),cc.copy())
  with rasterio.open(R.parents[1]/'processed/grid54/ref_grid_54km.tif') as src:
   shape=src.shape;xx,yy=rasterio.transform.xy(src.transform,rr,cc);lon,lat=map(np.asarray,transform(src.crs,'EPSG:4326',xx,yy))
  belt=(lat>=25)&(lat<49);regions=dict(east=belt&(lon>=-95),plains=belt&(lon>=-100)&(lon<-95),west=belt&(lon<-100))
  assert np.array_equal(sum(m.astype(int) for m in regions.values()),belt.astype(int))
  index={(int(r),int(c)):i for i,(r,c) in enumerate(zip(rr,cc))};oi=np.array([index.get((int(r),int(c)),-1) for r,c in zip(meta['obs_rows'],meta['obs_cols'])])
  oy=np.asarray(meta['years'])[np.asarray(meta['obs_time_indices'],int)]
  real=np.asarray(meta.get('obs_obsroute_idx',meta['obs_route_idx']))>=0
  keep=real&(oy>=2010)&(oy<=2025)&(np.asarray(meta['observed_results'])>0)&(oi>=0)
  observed=np.bincount(oi[keep],minlength=len(rr))>0
  land=np.zeros(shape,bool);land[rr,cc]=True;water=~land;seed=np.zeros(shape,bool);seed[[0,-1],:]=water[[0,-1],:];seed[:,[0,-1]]=water[:,[0,-1]]
  ocean=binary_propagation(seed,mask=water);coast=(land&binary_dilation(ocean,structure=generate_binary_structure(2,1)))[rr,cc]
  endpoints=dict(plains=(regions['plains'],lon<-99.2,lon>=-95.8),coast_to_coast=(belt,coast&(lon<-120),coast&(lon>-81)),interior=(belt,lon<-115,lon>=-80))
 else:
  assert all(np.array_equal(x,y) for x,y in zip(gridref,(years,rr,cc)))
 modern=(years>=2010)&(years<=2025);early=(years>=1902)&(years<=1915)
 np.testing.assert_allclose(arrays['potential_lambda'][modern].mean(),pathology['regions']['all_land']['modern']['lambda0']['land']['mean'],atol=1e-10)
 with np.load(out/'viz_ref/07_source_sink_fields.npz') as z:
  dead=z['allee_dead_mask'][rr,cc].astype(bool);kap=z['K_modern'][rr,cc];cached_lam=z['lam_fundamental_modern'][rr,cc]
 adult=a[modern].sum(0);total=state[modern].sum(0)
 maps=dict(lambda0=arrays['potential_lambda'][modern].mean(0),full_lambda=arrays['full_lambda'][modern].mean(0),C=ratio(arrays['breeder_contribution'][modern].sum(0),adult),
  biological_C=ratio(arrays['biological_C'][modern].sum(0),adult),local_replacement=ratio((np.nan_to_num(arrays['local_total_replacement'][modern])*state[modern]).sum(0),total),
  export=ratio(arrays['successful_breeder_export'][modern].sum(0),adult),retention=ratio(arrays['retained_breeders'][modern].sum(0),adult),
  immigrant_share=ratio(arrays['immigrant_breeders'][modern].sum(0),arrays['next_breeders'][modern].sum(0)),K=kap,adult_survival=arrays['local_adult_survival'][modern].mean(0))
 rec=dict(step=step,regions={},isolation_legacy=budget.get('isolation'),front={k:{thr:v[thr] for thr in ['1pct_of_median_occupied','10pct']} for k,v in front.items()},
  demography_all_population=pathology['regions']['all_land']['modern'],nests=pathology.get('nests'),holdout=metrics.get('holdout'),manifold_prior_report=metrics.get('manifold'),
  process_noise=metrics.get('process_noise'),founders_bird_conversion=pathology.get('founders_birds_conversion_102_8'),definition=metrics.get('fundamental_niche_definition'),checks=budget['checks'],missing=['paired real-record predictive uncertainty','new-fit HMC geometry/sampling','permanent-movement moments','finite-propagule followup'])
 for reg,mask in regions.items():
  rec['regions'][reg]={}
  for support,w in [('all_land',mask),('fixed_observed_cells',mask&observed)]:
   era={era:float(arrays['potential_lambda'][(years>=lo)&(years<=hi)][:,w].mean()) for era,(lo,hi) in ERAS.items()}
   fullera={era:float(arrays['full_lambda'][(years>=lo)&(years<=hi)][:,w].mean()) for era,(lo,hi) in ERAS.items()}
   rrout=dict(cells=int(w.sum()),potential_growth=era,historical_change=era['modern']-era['early'],post1966_change=era['modern']-era['bbs_early'],
    full_process_growth=fullera,full_historical_change=fullera['modern']-fullera['early'],
    modern_maps={n:classify(v,w) for n,v in maps.items() if n not in ['K','adult_survival']},
    K_q10_50_90=np.quantile(kap[w],[.1,.5,.9]).tolist(),K_within10pct_floor=float(np.mean(kap[w]<=1.1*pop['capacity_floor_internal'])),
    K_Allee_nonviable_positive_growth_share=float(dead[w].mean()),potential_sink_share_cached=float(np.mean(cached_lam[w]<1)),
    effective_adult_survival=float((arrays['origin_adult_survival'][modern][:,w]*a[modern][:,w]).sum()/a[modern][:,w].sum()),
    effective_juvenile_recruitment=float((arrays['origin_juvenile_recruitment'][modern][:,w]*j[modern][:,w]).sum()/j[modern][:,w].sum()),
    realized_offspring_per_breeder=float((arrays['realized_offspring_per_breeder'][modern][:,w]*a[modern][:,w]).sum()/a[modern][:,w].sum()),
    fitted_C_exposure_weighted=float(arrays['breeder_contribution'][modern][:,w].sum()/a[modern][:,w].sum()),
    K_Sa_rank_correlation=float(spearmanr(kap[w],maps['adult_survival'][w]).statistic))
   if 'process_correction' in budget:
    adj=arrays['process_adjustment'][:,w];mult=arrays['process_multiplier'][:,w];eps=np.log(mult)
    rrout['noise']=dict(gross_adjustment_over_starting_state=float(np.abs(adj).sum()/((a+j)[:,w].sum())),net_adjustment_over_starting_state=float(adj.sum()/(a+j)[:,w].sum()),
     cumulative_log_innovation_q10_50_90=np.quantile(eps.sum(0),[.1,.5,.9]).tolist(),annual_rms=float(np.sqrt(np.mean(eps**2))),active_year_rms=float(np.sqrt(np.mean(eps[(years>=1966)&(years<=2020)]**2))))
   rec['regions'][reg][support]=rrout
 rec['bridges']={}
 for name in ['lambda0','C','biological_C','local_replacement']:
  rec['bridges'][name]={str(threshold):{str(neigh):{ep:connected(maps[name],threshold,generate_binary_structure(2,1 if neigh==4 else 2),*args) for ep,args in endpoints.items()} for neigh in [4,8]} for threshold in [1.,1.01,1.05]}
 # Fitted standardized coefficient directions; these are not field correlations over geography.
 paramdir=R.parents[1]/f'processed/model_results/refined_float64__age_map_float32_grid54_{step}_quick90'
 with (paramdir/'map_params.pkl').open('rb') as f:params=HostUnpickler(f).load()
 lat={k.removesuffix('_auto_loc'):np.asarray(v) for k,v in params.items() if k.endswith('_auto_loc')}
 h=lat['manifold_factor'].reshape(-1);coef=np.repeat(h[:,None],4,axis=1)
 tau=pop.get('capacity_private_relative_sd',0.)
 if tau:coef[:,2]+=tau*lat['capacity_private'].reshape(-1)
 rho=pop['manifold_prior']['fixed_coupling']
 if rho<1.:coef=np.sqrt(rho)*coef+np.sqrt(1-rho)*lat['manifold_idio']
 norms=np.linalg.norm(coef,axis=0);cos=coef.T@coef/(norms[:,None]*norms[None,:])
 rec['coefficients']=dict(n_features=len(h),parameter_count=sum(np.asarray(v).size for v in params.values()),standardized_shared_rms=float(np.sqrt(np.mean(h*h))),
  cosine=cos.tolist(),private_K_relative_norm=float(tau*np.linalg.norm(lat['capacity_private'])/np.linalg.norm(h)) if tau else None)
 rec['evidence_sha256']={k:hashlib.sha256((out/v).read_bytes()).hexdigest() for k,v in [('pathology','pathology.json'),('metrics','viz_ref/metrics.json'),('front','front_speed.json'),('budget_receipt','contribution/READY.json')]}
 cache[step]=maps;records.append(rec);print('SUMMARIZED',step,flush=True)
comparisons=[]
for rec in records:
 for reference in ['step39_kfloor4p0_s41','step39_kfloor4p0_s42','step44_shared_s43']:
  if rec['step']==reference:continue
  for reg,mask in regions.items():
   for support,w in [('all_land',mask),('fixed_observed_cells',mask&observed)]:
    fields={}
    for n in cache[reference]:
     x,y=cache[reference][n],cache[rec['step']][n];valid=w&np.isfinite(x)&np.isfinite(y);d=y[valid]-x[valid]
     fields[n]=dict(cells=int(valid.sum()),left_mean=finite_mean(x[valid]),right_mean=finite_mean(y[valid]),signed_change=finite_mean(d),mean_absolute_change=finite_mean(np.abs(d)),rms_change=float(np.sqrt(np.mean(d*d))),
       strict_reversal_cells=int(np.sum((x[valid]-1)*(y[valid]-1)<0)) if n in ['lambda0','C','biological_C','local_replacement'] else None,
       neutral005_reversal_cells=int(np.sum(((x[valid]<.95)&(y[valid]>1.05))|((y[valid]<.95)&(x[valid]>1.05)))) if n in ['lambda0','C','biological_C','local_replacement'] else None)
    comparisons.append(dict(reference=reference,step=rec['step'],region=reg,support=support,metrics=fields))
result=dict(epoch=time.time(),definition='Potential lambda=Sa+Sj*Fmax under yearling breeding. Equal fixed cells in each era; actual-state contributions use fitted exposures within each cell. No posterior identification claim.',
 regional_support='25<=latitude<49: East longitude >= -95; Plains [-100,-95); West longitude < -100. Isolation summaries retain explicitly labeled legacy masks.',
 region_cells={k:int(v.sum()) for k,v in regions.items()},fixed_observed_cells={k:int((v&observed).sum()) for k,v in regions.items()},eras=ERAS,records=records,comparisons=comparisons)
p=R/'monitor/FIVE_FIT_ANALYSIS_20261103.json';p.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print('ANALYSIS_READY',p,flush=True)