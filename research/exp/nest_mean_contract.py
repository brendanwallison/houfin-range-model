"""Check saved generating means with fit-derived versus original centering."""
import argparse
import json
import os
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer import Predictive
from audit_occupied_center import centers
from src.config_utils import load_age_model_config
from src.model.age_priors import build_model_2d
from src.model.checkpoints import load_map_params,auto_delta_params_to_latents
from src.model.data_loading import load_data
from src.model.runtime_diagnostics import require_gpu


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--truth',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();jax.config.update('jax_enable_x64',True)
    cfg=load_age_model_config(); syn=Path(os.environ['HOUFIN_SYNTHETIC_OBS'])
    c=centers(Path(cfg['input_dir']),syn)
    device=require_gpu('synthetic generating-mean contract audit')
    data=load_data(cfg['input_dir'],target_device=device,precision='float64',verbose=False)
    info=json.loads(a.truth.read_text());params,_=load_map_params(info['run_dir'])
    point=auto_delta_params_to_latents(params);point.update(info['set']);point.update(info['nuisance_truth'])
    pred=Predictive(build_model_2d,posterior_samples={k:jnp.asarray(v)[None] for k,v in point.items()},return_sites=['expected_obs'])
    with np.load(syn,allow_pickle=False) as z: saved=np.asarray(z['mu'],dtype=float)
    out={'status':'complete','centering':c,'comparisons':{}}
    for name in ['as_fitted','original_center']:
        d=dict(data)
        if name=='original_center':d['habitat_center_occupied']=jnp.asarray(c['original']['vector'])
        mu=np.asarray(jax.block_until_ready(pred(jax.random.PRNGKey(0),data=d,prior_scale=1.)['expected_obs'][0]))
        delta=mu-saved
        out['comparisons'][name]={'max_abs':float(abs(delta).max()),'rms':float(np.sqrt(np.mean(delta**2))),
                                 'mean_predicted':float(mu.mean()),'mean_saved':float(saved.mean()),
                                 'matches_saved_mu':bool(np.allclose(mu,saved,rtol=1e-5,atol=1e-6))}
        print(name,out['comparisons'][name],flush=True)
    a.out.parent.mkdir(parents=True,exist_ok=True);tmp=a.out.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');tmp.replace(a.out)


if __name__=='__main__':main()
