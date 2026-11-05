"""Score source-specific CBC predictions from an accepted frozen fit/budget."""
import argparse,json
from pathlib import Path
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from numpyro import handlers
from scipy.stats import nbinom
from scipy.special import logsumexp
from src.config_utils import load_age_model_config
from src.model.checkpoints import load_map_params,auto_delta_params_to_latents
from src.model.cbc_joint import cbc_observe

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--budget',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
cfg=load_age_model_config();spec=cfg['population_model']['cbc_likelihood']
params,_=load_map_params(str(a.run));lat=auto_delta_params_to_latents(params)
if not spec['enabled']:
    out={'status':'matched no-CBC demographic fit','log_inv_total':float(lat.get('log_inv_total',np.nan)),
      'scope':'No CBC observation parameters fitted; not a CBC prediction comparison'}
else:
    with np.load(spec['file']) as z:data={k:jnp.asarray(z[k]) for k in z.files}
    data['cbc_n_circles']=int(data['cbc_w_local'].shape[0])
    with np.load(a.budget) as z:
        rr,cc=z['land_rows'],z['land_cols'];adult=z['next_adult'];young=z['next_juvenile'];years=z['years']
    shape=(len(years),int(rr.max()+1),int(cc.max()+1));A=jnp.zeros(shape).at[:,rr,cc].set(adult);J=jnp.zeros(shape).at[:,rr,cc].set(young)
    data.update(land_rows=jnp.asarray(rr),land_cols=jnp.asarray(cc))
    # Use the exact source-specific scale from metadata, not a guessed bird conversion.
    import pickle
    m=pickle.load(open(Path(cfg['input_dir'])/'metadata.pkl','rb'))
    trace=handlers.trace(handlers.substitute(cbc_observe,data=lat)).get_trace(data,A,J,float(m['pop_scalar']),spec)
    mu=np.asarray(trace['cbc_expected']['value']).copy();y=np.asarray(data['cbc_count']);k=float(lat['cbc_concentration'])
    ll=nbinom.logpmf(y,k,k/(k+mu))
    if not(np.isfinite(mu).all() and np.isfinite(ll).all()):raise ValueError('nonfinite CBC score')
    t=np.asarray(data['cbc_t']);summer=years[t]
    circle=np.asarray(data['cbc_circle']);train=np.asarray(data['cbc_train'])>0
    ntrain=np.bincount(circle[train],minlength=data['cbc_n_circles'])
    unseen=ntrain[circle]==0
    mu_point=mu.copy();detection=1-(k/(k+mu))**k
    # A held-out circle has an unobserved random effect. Integrate its prior,
    # rather than reporting a falsely precise prediction at raw effect zero.
    if unseen.any():
        nodes,weights=np.polynomial.hermite.hermgauss(20)
        sd=float(spec.get('circle_sd',1.))*np.sqrt(1-1/data['cbc_n_circles'])
        mixed_mu=mu[unseen,None]*np.exp(np.sqrt(2)*sd*nodes[None,:])
        ll[unseen]=logsumexp(nbinom.logpmf(y[unseen,None],k,k/(k+mixed_mu))+np.log(weights/np.sqrt(np.pi)),axis=1)
        detection[unseen]=np.sum((1-(k/(k+mixed_mu))**k)*(weights/np.sqrt(np.pi)),axis=1)
        mu[unseen]*=np.exp(.5*sd*sd)
    scopes={'train':np.asarray(data['cbc_train'])>0,'spatial':np.asarray(data['cbc_spatial_holdout'])>0,
      'temporal':np.asarray(data['cbc_temporal_holdout'])>0,
      'historical_train':(np.asarray(data['cbc_train'])>0)&(summer<=1965)}
    out={'status':'complete','scores':{},'parameters':{s:np.asarray(v).tolist() for s,v in lat.items() if s.startswith('cbc_') and s!='cbc_circle_raw'},
      'log_inv_total':float(lat.get('log_inv_total',cfg['population_model']['pinned_sites'].get('log_inv_total',np.nan))),
      'scope':'Conditional on joint MAP; unseen-circle effects integrated by 20-point Gauss-Hermite quadrature. Other parameter uncertainty is not integrated. Not independent BBS transfer.',
      'unseen_circle_rows':int(unseen.sum())}
    for name,keep in scopes.items():
        out['scores'][name]={'n':int(keep.sum()),'mean_log_score':float(ll[keep].mean()),'observed_mean':float(y[keep].mean()),
          'predicted_mean':float(mu[keep].mean()),'observed_detection':float((y[keep]>0).mean()),
          'predicted_detection':float(detection[keep].mean())}
    np.savez_compressed(a.out.with_suffix('.npz'),expected=mu,expected_point=mu_point,count=y,loglik=ll,summer_year=summer,
      train=np.asarray(data['cbc_train']),spatial=np.asarray(data['cbc_spatial_holdout']),temporal=np.asarray(data['cbc_temporal_holdout']))
a.out.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps(out,indent=2))
