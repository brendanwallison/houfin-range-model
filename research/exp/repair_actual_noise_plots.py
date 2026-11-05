"""Recheck noise plots at actual checkpoints using CPU within a TACC allocation."""
import os,sys,json,runpy,shutil,hashlib
from pathlib import Path
step=sys.argv[1];assert step in ['step45_cell_iid_s010_s41','step45_regional_s010_s41']
R=Path('/work/07980/bwa386/ls6/houfin/research/grid54');run=R/'runs'/step;code=run/'code';root=R.parents[1]
os.environ.update(AGE_MODEL_CONFIG=str(code/f'config/overlays/grid54_{step}.json'),DATA_CONFIG=str(code/'config/overlays/grid54_data.json'),
 JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',MPLBACKEND='Agg',HOUFIN_MODEL_PRECISION='float64',HOUFIN_Z_FLOAT32='1',
 HOUFIN_PROCESSED=str(root/'processed'),OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
sys.path[:0]=[str(code),str(code/'src'),str(code/'scripts/viz'),str(R/'monitor')]
import numpy as np
import rasterio
from src.config_utils import load_age_model_config
from src.model.data_loading import process_noise_basis
from src.model.minor_flexibility import normalize_cell_variance
from src.model.checkpoints import auto_delta_params_to_latents
from range_watch import HostUnpickler
cfg=load_age_model_config();spec=cfg['population_model']['process_noise']
with np.load(R/'runs/step44_shared_s43/out/contribution/annual_budget.npz') as z:
 years,rr,cc=z['years'],z['land_rows'],z['land_cols']
with rasterio.open(root/'processed/grid54/ref_grid_54km.tif') as grid:shape=grid.shape
meta=dict(years=years,land_rows=rr,land_cols=cc,Ny=shape[0],Nx=shape[1])
basis,active=process_noise_basis(meta,spec)
if spec.get('form')!='cell_iid' and spec.get('unit_cell_variance'):basis=normalize_cell_variance(basis)
data=dict(pn_S=basis,pn_active=active,pn_time_indices=np.flatnonzero(active))
ns=runpy.run_path(str(code/'scripts/viz/map_diagnostics.py'),run_name='noise_plot_repair')
receipts=[]
points=['map'] if spec['form']=='cell_iid' else ['map','refined']
for point in points:
 dirname=('refined_float64__' if point=='refined' else '')+f'age_map_float32_grid54_{step}_quick90'
 checkpoint=root/'processed/model_results'/dirname/'map_params.pkl'
 with checkpoint.open('rb') as stream:params=HostUnpickler(stream).load()
 lat=auto_delta_params_to_latents(params);sim=dict(process_noise_u=np.asarray(lat['process_noise_u']))
 amend=run/'diagnostic_amendments/noise_v1';out=amend/f'actual_{point}_process_noise.png'
 metrics=ns['plot_process_noise'](data,sim,years,rr,cc,shape,out)
 eps=ns['_process_noise_eps'](data,sim)
 assert eps.shape==(len(years),len(rr)) and np.isfinite(eps).all() and np.all(eps[active==0]==0)
 # For the completed regional fit update only this plot/metric; retain originals.
 if spec['form']=='regional':
  folder=run/'out'/('viz_ref' if point=='refined' else 'viz_map');old=folder/'18_process_noise.png';mfile=folder/'metrics.json'
  if old.exists():shutil.copy2(old,amend/f'original_{point}_process_noise.png')
  shutil.copy2(out,old)
  png=run/'out'/('viz_ref_png' if point=='refined' else 'viz_map_png')/'18_process_noise.png'
  if png.parent.exists():shutil.copy2(out,png)
  if mfile.exists():
   shutil.copy2(mfile,amend/f'original_{point}_metrics.json');m=json.loads(mfile.read_text());m['process_noise']=metrics
   tmp=mfile.with_suffix('.json.next');tmp.write_text(json.dumps(m,indent=2)+'\n');tmp.replace(mfile)
 receipt=dict(step=step,point=point,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),raw_shape=list(sim['process_noise_u'].shape),eps_shape=list(eps.shape),finite=True,inactive_years_exactly_zero=True,metrics=metrics)
 receipts.append(receipt);print('ACTUAL_NOISE_PLOT_VALIDATED',step,point,receipt['raw_shape'],receipt['eps_shape'],flush=True)
(amend/'actual_checkpoint_validation.json').write_text(json.dumps(receipts,indent=2)+'\n')