"""Check the realized release-cell filling/departure trajectory under frozen fits."""
import inspect,json
from pathlib import Path
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from numpyro import handlers
from src.config_utils import load_age_model_config
from src.model import age_priors as ap
from src.model.data_loading import load_data
from src.model.checkpoints import load_map_params,auto_delta_params_to_latents

cfg=load_age_model_config();data=load_data(cfg['input_dir'],precision='float64',verbose=False)
root=Path('/mnt/e/Datasets/houfin/grid54/runs');base=Path('/work/07980/bwa386/ls6/houfin/processed/model_results')
lr,lc=np.asarray(data['land_rows']),np.asarray(data['land_cols']);r,c=np.asarray(data['inv_locations'])[0]
cell=int(np.flatnonzero((lr==r)&(lc==c))[0]);records=[];original=ap.forward_sim_age_structured
for s in ['step38_cbc1951_free_s41','step38_cbc1951_noncbc_free_s41']:
    params,_=load_map_params(str(base/f'refined_float64__age_map_float32_grid54_{s}_quick90'))
    lat={k:jnp.asarray(v) for k,v in auto_delta_params_to_latents(params).items()};captured=[]
    def record(*args,**kw):
        b=inspect.signature(original).bind(*args,**kw);b.apply_defaults();value=original(*args,**kw)
        captured.append((b.arguments,value));return value
    ap.forward_sim_age_structured=record
    try:handlers.seed(handlers.substitute(ap.build_model_2d,data=lat),rng_seed=104)(data=data,prior_scale=1.)
    finally:ap.forward_sim_age_structured=original
    x,states=captured[0];K=np.asarray(x['K_flat'])[:,cell]
    with np.load(root/s/'out/contribution/annual_budget.npz') as z:
        years=z['years'];N=z['input_adult'][:,cell]+z['input_juvenile'][:,cell]
        expected=z['next_adult']
        replay=np.asarray(states[1])[:,lr,lc]
        if not np.allclose(expected,replay,rtol=1e-10,atol=1e-10):raise ValueError('frozen trajectory mismatch')
    ratio=N/np.maximum(K,1e-6)
    p=np.clip(ratio-np.logaddexp(0,10*(ratio-1))/10,1e-6,1.)
    take=np.isin(years,[1951,1952,1955,1960,1965,1970])
    records.append({'step':s,'veit_active':bool(x.get('veit_departure')),'native_multiplier_present':x.get('departure_scale') is not None,
      'release_cell':{'row':int(r),'col':int(c)},'selected_years':[{'year':int(y),'N_before_dispersal':float(n),'K':float(k),'N_over_K':float(v),'departure_fraction':float(q)} for y,n,k,v,q in zip(years[take],N[take],K[take],ratio[take],p[take])],
      'first_year_at_80pct_capacity':int(years[np.flatnonzero((years>=1951)&(ratio>=.8))[0]]) if ((years>=1951)&(ratio>=.8)).any() else None})
out=Path('/mnt/c/Dev/houfin-range-model/research/experiments/CBC_FOUNDER_DEPARTURE_20261102.json')
out.write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(records,indent=2),flush=True)
