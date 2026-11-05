"""Inspect normal-valued components of saved refinement states without refitting.

Validate flat-state decoding and exact potential-growth mapping against exported
completed fits before evaluating any incomplete checkpoint. No population replay.
"""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,json,time
from pathlib import Path
import numpy as np
from scipy.special import expit
import rasterio
from rasterio.warp import transform
R=Path('/work/07980/bwa386/ls6/houfin/research/grid54');sys.path.insert(0,str(R/'monitor'))
from range_watch import HostUnpickler

def load(p):
 with p.open('rb') as f:return HostUnpickler(f).load()

def params(directory):return {k.removesuffix('_auto_loc'):np.asarray(v) for k,v in load(directory/'map_params.pkl').items() if k.endswith('_auto_loc')}

def decode(step):
 base=R.parents[1]/f'processed/model_results/age_map_float32_grid54_{step}_quick90';initial=params(base)
 refined=base.with_name('refined_float64__'+base.name);state=load(refined/'refine_state.pkl');x=np.asarray(state['x'])
 # Frozen model uses ZeroSumNormal for observer-route effects: its support
 # transform removes exactly one coordinate. We do not decode those effects.
 names=sorted(initial);assert len(x)==sum(v.size for v in initial.values())-1,'unexpected support-transform dimension'
 offset=0;raw={}
 for name in names:
  n=initial[name].size-(1 if name=='observer_route_eps' else 0)
  if name!='observer_route_eps':raw[name]=x[offset:offset+n].reshape(initial[name].shape)
  offset+=n
 assert np.isfinite(x).all()
 return initial,raw,state,refined

meta=load(R/'inputs/established1951/metadata.pkl');years=np.asarray(meta['years']);rr,cc=meta['land_rows'],meta['land_cols']
Z=np.asarray(np.memmap(R/'inputs/established1951'/meta['z_gathered_path'],dtype='float32',mode='r',shape=(meta['time'],meta['N_land'],meta['M'])),dtype=np.float64)
lookup=np.full((meta['Ny'],meta['Nx']),-1,int);lookup[rr,cc]=np.arange(len(rr));cell=lookup[meta['obs_rows'],meta['obs_cols']]
keep=(np.asarray(meta['observed_results'])>0)&(cell>=0);pairs=np.unique(np.stack([meta['obs_time_indices'][keep],cell[keep]],axis=1),axis=0)
center=Z[pairs[:,0],pairs[:,1]].mean(0)
with rasterio.open(R.parents[1]/'processed/grid54/ref_grid_54km.tif') as src:
 xx,yy=rasterio.transform.xy(src.transform,rr,cc);lon,lat=map(np.asarray,transform(src.crs,'EPSG:4326',xx,yy))
belt=(lat>=25)&(lat<49);regions=dict(east=belt&(lon>=-95),plains=belt&(lon>=-100)&(lon<-95),west=belt&(lon<-100))

def potential(raw):
 h=raw['manifold_factor'].reshape(-1);H=Z@h-center@h;scale=np.logaddexp(0,raw['w_scale_raw'])
 sa=expit(raw['alpha_a']+scale[0]*H);sj=sa*expit(raw['alpha_j']+scale[3]*H);f=np.logaddexp(0,raw['alpha_f']+scale[1]*H)
 return sa+sj*f

validated=[]
for step in ['step39_kfloor4p0_s42','step45_regional_s010_s41']:
 initial,raw,state,ref=decode(step);exported=params(ref)
 keys=['alpha_a','alpha_j','alpha_f','manifold_factor','w_scale_raw']
 if 'process_noise_u' in raw:keys.append('process_noise_u')
 for key in keys:np.testing.assert_allclose(raw[key],exported[key],atol=1e-12,rtol=1e-12)
 pred=potential(raw)
 with np.load(R/f'runs/{step}/out/contribution/annual_budget.npz') as z:gold=z['potential_lambda']
 err=float(np.max(np.abs(pred-gold)));np.testing.assert_allclose(pred,gold,atol=2e-10,rtol=2e-10)
 validated.append(dict(step=step,flat_normal_components_match_export=True,max_potential_growth_error=err))
 print('VALIDATED',step,'ERROR',err,flush=True)

records=[]
for step in ['step45_cell_iid_s005_s41','step45_cell_iid_s010_s41']:
 initial,raw,state,ref=decode(step);cfg=json.loads((R/f'runs/{step}/code/config/overlays/grid54_{step}.json').read_text());spec=cfg['population_model']['process_noise']
 assert cfg['population_model']['manifold_prior']['fixed_coupling']==1. and not cfg['population_model']['readout']['split']
 growth=potential(raw);active=(years>=spec['window_start'])&(years<=spec['window_end']);u=raw['process_noise_u'];assert u.shape==(int(active.sum()),len(rr))
 eps=float(spec['sigma'])*u-.5*float(spec['sigma'])**2
 r=dict(step=step,checkpoint_iteration=len(state['history']),terminal=state.get('terminal'),stop_reason=state.get('reason'),accepted_export_present=(ref/'refine_report.json').exists(),
  scope='Saved unconstrained Normal components; exact potential growth independently validated against two exported fits. No population replay, source/sink or prediction claim.',
  u_rms=float(np.sqrt(np.mean(u*u))),active_log_innovation_rms=float(np.sqrt(np.mean(eps*eps))),max_abs_log_innovation=float(np.max(np.abs(eps))),
  cumulative_log_innovation_q10_50_90=np.quantile(eps.sum(0),[.1,.5,.9]).tolist(),regions={})
 for reg,mask in regions.items():
  early=float(growth[(years>=1902)&(years<=1915)][:,mask].mean());bbs=float(growth[(years>=1966)&(years<=1985)][:,mask].mean());modern=float(growth[(years>=2010)&(years<=2025)][:,mask].mean())
  r['regions'][reg]=dict(early=early,modern=modern,historical_change=modern-early,post1966_change=modern-bbs,active_log_innovation_rms=float(np.sqrt(np.mean(eps[:,mask]**2))),cumulative_log_innovation_q10_50_90=np.quantile(eps[:,mask].sum(0),[.1,.5,.9]).tolist())
 records.append(r);print('PARTIAL_POINT',json.dumps(r),flush=True)
result=dict(epoch=time.time(),validation=validated,records=records)
(R/'monitor/PARTIAL_INDEPENDENT_ANALYSIS_20261103.json').write_text(json.dumps(result,indent=2)+'\n')
