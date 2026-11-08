"""Read saved results only: common geographic support and fixed arrival threshold."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
from pathlib import Path
import json,sys
import numpy as np
import rasterio
from rasterio.warp import transform
from scipy.ndimage import label as component_label,binary_dilation,binary_propagation,generate_binary_structure
R=Path('/work/07980/bwa386/ls6/houfin/research/grid54');sys.path.insert(0,str(R/'monitor'))
from range_watch import HostUnpickler
with (R/'inputs/established1951/metadata.pkl').open('rb') as f:m=HostUnpickler(f).load()
rr,cc=np.asarray(m['land_rows']),np.asarray(m['land_cols']);years=np.asarray(m['years'])
with rasterio.open(R.parents[1]/'processed/grid54/ref_grid_54km.tif') as src:
 shape=src.shape
 xx,yy=rasterio.transform.xy(src.transform,rr,cc);lon,lat=map(np.asarray,transform(src.crs,'EPSG:4326',xx,yy))
belt=(lat>=25)&(lat<49);regions=dict(east=belt&(lon>=-95),plains=belt&(lon>=-100)&(lon<-95),west=belt&(lon<-100))
land=np.zeros(shape,bool);land[rr,cc]=True;water=~land;seed=np.zeros(shape,bool)
seed[[0,-1],:]=water[[0,-1],:];seed[:,[0,-1]]=water[:,[0,-1]]
ocean=binary_propagation(seed,mask=water);coast=(land&binary_dilation(ocean,structure=generate_binary_structure(2,1)))[rr,cc]
west_coast=coast&(lon<-120);east_coast=coast&(lon>-81)
def bridges(values):
 result={}
 for threshold in [1.,1.05]:
  result[str(threshold)]={}
  active=np.isfinite(values)&(values>threshold)&belt;grid=np.zeros(shape,bool);grid[rr,cc]=active
  for connectivity in [1,2]:
   labs,_=component_label(grid,structure=generate_binary_structure(2,connectivity));labs=labs[rr,cc]
   west=np.unique(labs[active&west_coast]);east=np.unique(labs[active&east_coast])
   result[str(threshold)][str(4 if connectivity==1 else 8)]=bool(np.intersect1d(west[west>0],east[east>0]).size)
 return result
lookup=np.full((m['Ny'],m['Nx']),-1,int);lookup[rr,cc]=np.arange(len(rr));cell=lookup[m['obs_rows'],m['obs_cols']]
oy=years[np.asarray(m['obs_time_indices'])];real=np.asarray(m['obs_obsroute_idx'] if 'obs_obsroute_idx' in m else m['obs_route_idx'])>=0
survey=np.full(len(rr),np.inf);first=np.full(len(rr),np.inf);ok=real&(cell>=0)
np.minimum.at(survey,cell[ok],oy[ok]);positive=ok&(np.asarray(m['observed_results'])>0);np.minimum.at(first,cell[positive],oy[positive])
eligible=regions['east']&np.isfinite(first)&(first>=survey+3)
threshold=.008027860278758616 # The completed full control's 10% threshold, fixed for ALL fits.
modern=(years>=2010)&(years<=2025);records=[]
for spec in json.loads((R/'monitor/misspecification_20261104_specs.json').read_text()):
 step=spec['step'];out=R/'runs'/step/'out';p=out/'misspecification_followup.json'
 if not p.exists() or not (out/'contribution/READY.json').exists():continue
 d=json.loads(p.read_text());path=json.loads((out/'pathology.json').read_text());v=path['regions']['all_land']['modern']
 record=dict(step=step,growth_change={r:d['regions']['full'][r]['historical_growth_change'] for r in regions},
  adult_survival_median=v['Sa']['population']['q10_50_90'][1],juvenile_adult_ratio_median=v['Sj_over_Sa']['population']['q10_50_90'][1],
  juvenile_ratio_ceiling_population_share=v['juvenile_ratio_above_095_population_share'],Fmax_median=v['Fmax']['population']['q10_50_90'][1],
  realized_reproduction_at_nests=path['nests']['nest_F_real']['mean'],regions={})
 with np.load(out/'misspecification_population.npz') as z:
  for label in ['full','skeleton']:
   n=z[label];hit=n>=threshold;arrival=years[np.argmax(hit,axis=0)].astype(float);arrival[~hit.any(0)]=np.nan
   lag=arrival[eligible]-first[eligible]
   record[label+'_fixed_threshold_front']=dict(cells=int(eligible.sum()),median_offset=float(np.nanmedian(lag)),never_fraction=float(np.mean(~np.isfinite(lag))),share_earlier_5=float(np.mean(lag<=-5)))
 with np.load(out/'viz_ref/07_source_sink_fields.npz') as z:dead=z['allee_dead_mask'][rr,cc].astype(bool)
 with np.load(out/'contribution/annual_budget.npz') as z:
  a=z['input_adult'][modern];state=a+z['input_juvenile'][modern]
  C=z['breeder_contribution'][modern].sum(0)/np.maximum(a.sum(0),1e-20)
  local=(np.nan_to_num(z['local_total_replacement'][modern])*state).sum(0)/np.maximum(state.sum(0),1e-20)
  immigrant=z['immigrant_breeders'][modern].sum(0)/np.maximum(z['next_breeders'][modern].sum(0),1e-20)
  potential=z['potential_lambda'][modern].mean(0)
  record['coast_to_coast_bridges']={k:bridges(v) for k,v in [('potential_growth',potential),('contribution',C),('local_replacement',local)]}
  for region,mask in regions.items():
   q=d['regions']['full'][region]
   record['regions'][region]=dict(cells=int(mask.sum()),C_mean=float(C[mask].mean()),C_sink_share=float((C[mask]<1).mean()),
    local_replacement_mean=float(local[mask].mean()),local_sink_share=float((local[mask]<1).mean()),immigrant_share_mean=float(immigrant[mask].mean()),
    K_Allee_nonviability_share=float(dead[mask].mean()),potential_sink_share=float((potential[mask]<1).mean()),
    heldout_LPD=q['heldout_raw_loglik']/q['n_holdout'],training_LPD=q['raw_real_record_loglik']/q['n_train'],
    annual_abundance_log_adjustment_RMS=d['dependence'][region]['annual_abundance_log_multiplier_rms'],
    establishment_log_adjustment_RMS=d['dependence'][region]['establishment_log_multiplier_rms'])
 records.append(record)
result=dict(support='Equal land cells 25–49N; East >=95W, Plains 100–95W, West <100W.',
 fixed_arrival_threshold=threshold,records=records,scope='One optimizer seed completed; no posterior-identifiability or overall winner claim. Raw heldout scores lack paired uncertainty.')
(R/'monitor/MISSPECIFICATION_MATCHED_SUMMARY_20261104.json').write_text(json.dumps(result,indent=2)+'\n')
for rec in records:print(rec['step'],'FIXED_FRONT',rec['full_fixed_threshold_front'],'PLAINS',rec['regions']['plains'])
