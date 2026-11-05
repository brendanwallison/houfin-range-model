"""Replay one accepted frozen MAP point on CPU and validate readout parity.

Run with cwd/PYTHONPATH and AGE_MODEL_CONFIG of the point's own frozen source.
No fitting or new observation parameters. Requires the current simple readout
without extra route/regional/year offsets; observer effects are checked before
their application and are never copied to eBird.
"""
import argparse
import hashlib
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer import Predictive
from src.config_utils import load_age_model_config
from src.model.age_priors import build_model_2d
from src.model.checkpoints import load_map_params,auto_delta_params_to_latents
from src.model.data_loading import load_data
from census import breeding_grid
from prepare import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'READY.json').exists():raise FileExistsError('completed census export retained')
    report=json.load(open(a.run/'refine_report.json'))
    if not report.get('valid'):raise ValueError('unaccepted point')
    cfg=load_age_model_config();obs=cfg['population_model']['observation']
    if obs.get('form')!='breeding' or obs.get('route_effects') or not obs.get('observer_effect',{}).get('enabled'):
        raise ValueError('this parity export is registered for the current breeding/observer configuration only')
    jax.config.update('jax_enable_x64',True)
    data=load_data(cfg['input_dir'],target_device=jax.devices('cpu')[0],precision='float64',verbose=False)
    params,_=load_map_params(str(a.run));lat=auto_delta_params_to_latents(params)
    names=['simulated_density','Na_grid','Nj_grid','initpop_seeded','obs_fledgling_weight','expected_obs_before_observer']
    prediction=Predictive(build_model_2d,posterior_samples={k:jnp.asarray(v)[None] for k,v in lat.items()},return_sites=names)
    sim={k:np.asarray(v[0]) for k,v in jax.block_until_ready(prediction(jax.random.PRNGKey(104),data=data,prior_scale=1.)).items()}
    t=np.asarray(data['obs_time_indices']);rows=np.asarray(data['obs_rows']);cols=np.asarray(data['obs_cols'])
    phi=float(sim['obs_fledgling_weight']);scale=float(data['pop_scalar'])
    grid=breeding_grid(sim['simulated_density'],sim['Nj_grid'],phi,scale,initial=sim['initpop_seeded'])
    predicted=grid[t,rows,cols];reference=sim['expected_obs_before_observer']
    max_abs=float(np.max(np.abs(predicted-reference)))
    if not np.allclose(predicted,reference,rtol=1e-10,atol=1e-10):raise ValueError(f'BBS census parity failed: max error {max_abs}')
    lr,lc=np.asarray(data['land_rows']),np.asarray(data['land_cols'])
    years=np.asarray(data['years'])
    np.savez_compressed(a.out/'census.npz',years=years,land_rows=lr,land_cols=lc,
      total=sim['simulated_density'][:,lr,lc],adult=sim['Na_grid'][:,lr,lc],young=sim['Nj_grid'][:,lr,lc],
      bbs_reference_readout=grid[:,lr,lc],initial=sim['initpop_seeded'][lr,lc],fledgling_weight=phi,pop_scalar=scale)
    with (a.run/'map_checkpoint.pkl').open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    write_json(a.out/'READY.json',{'status':'complete','run':str(a.run),'checkpoint_sha256':sha,'input_dir':cfg['input_dir'],
      'precision':'float64','device':'cpu','bbs_before_observer_parity_max_abs':max_abs,'n_bbs_records_checked':int(len(t)),
      'bbs_fledgling_weight':phi,'first_year_readout':'uses the frozen simulator seeded initial state, including pseudozero observation parity',
      'scope':'Frozen-point census export and parity, not eBird likelihood acceptance or a demographic refit',
      'remaining':'eBird young visibility and census-date correspondence require explicit comparison; BBS observer offsets do not transfer'})
    print('CENSUS READY',len(t),'BBS records, max absolute error',max_abs,flush=True)


if __name__=='__main__':main()
