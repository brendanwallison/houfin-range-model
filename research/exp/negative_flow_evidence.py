"""Collect failed-flow QC evidence on an allocated node; never writes READY."""
import os
if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('An allocation is required')
import sys,json
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
from numpyro import handlers
from src.config_utils import load_age_model_config
from src.model import age_priors as ap,age_forward as af
from src.model.data_loading import load_data
from src.model.checkpoints import load_map_params,auto_delta_params_to_latents
from src.model.misspecification_suite import audit_arguments
code=Path(ap.__file__).parents[2];sys.path.insert(0,str(code/'research/misspecification_audit'))
from contribution_budget import audit_forward
ref=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(exist_ok=True)
cfg=load_age_model_config();data=load_data(cfg['input_dir'],target_device=jax.devices('cpu')[0],precision='float64',verbose=False)
params,_=load_map_params(str(ref));lat=auto_delta_params_to_latents(params)
handlers.seed(handlers.condition(ap.build_model_2d,data=lat),rng_seed=104)(data=data,prior_scale=1.)
last=ap._MM_LAST;x=audit_arguments(last);arrays,_=audit_forward(af,x,last['states'])
names=['immigrant_breeders','immigration','successful_emigration','journey_mortality_origin']
result=dict(accepted_contribution_audit=False,scope='QC evidence only; original QC gate unchanged.',flows={})
for key in names:
 a=arrays[key];positive=float(np.maximum(a,0).sum());negative=float(-np.minimum(a,0).sum())
 result['flows'][key]=dict(minimum=float(a.min()),maximum=float(a.max()),negative_cells=int((a<0).sum()),
  negative_mass=negative,positive_mass=positive,negative_over_positive=negative/max(positive,1e-30))
result['replay_errors']={k:float(np.max(np.abs(arrays[k]))) for k in ['BIDE_error','replay_adult_error','replay_juvenile_error']}
result['kernel_negative_mass']={}
for key in ['adult_fft_kernel','juvenile_fft_kernel_stack']:
 kernel=np.fft.ifft2(np.asarray(x[key],dtype=np.complex128),axes=(-2,-1)).real
 result['kernel_negative_mass'][key]=dict(minimum=float(kernel.min()),negative_mass=float(-np.minimum(kernel,0).sum()))
np.savez_compressed(out/'negative_flow_fields.npz',**{k:arrays[k] for k in names})
(out/'negative_flow_evidence.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
