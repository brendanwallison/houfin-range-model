"""Conditional binary CBC validation; unseen circle effects integrated by quadrature."""
import argparse,json,pickle
from pathlib import Path
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp
from src.config_utils import load_age_model_config
from src.model.checkpoints import load_map_params,auto_delta_params_to_latents
from src.model.cbc_joint import winter_readout,boxcox_effort

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--budget',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    cfg=load_age_model_config();s=cfg['population_model']['cbc_likelihood']
    if not s['enabled'] or s.get('family')!='early_detection':return
    params,_=load_map_params(str(a.run));lat=auto_delta_params_to_latents(params)
    with np.load(s['file']) as z:d={k:jnp.asarray(z[k]) for k in z.files}
    with np.load(a.budget) as z:adult,young,years=z['next_adult'],z['next_juvenile'],z['years']
    m=pickle.load(open(Path(cfg['input_dir'])/'metadata.pkl','rb'))
    density=np.asarray(winter_readout(jnp.asarray(adult),jnp.asarray(young),d,float(s['fixed_migrant_fraction']),1.))*float(m['pop_scalar'])
    raw=np.asarray(lat['cbc_circle_raw']);ci=np.asarray(d['cbc_circle']);sd=float(s.get('circle_sd',1.))
    lr=np.log(density+1e-8)+float(lat['cbc_log_scale'])+sd*(raw-raw.mean())[ci]
    lr+=float(lat['cbc_effort_slope'])*np.asarray(boxcox_effort(d['cbc_log_effort'],lat['cbc_effort_power']))
    lr+=float(lat['cbc_effort_missing_effect'])*np.asarray(d['cbc_effort_missing'])+float(lat['cbc_feeder_effect'])*np.asarray(d['cbc_feeder'])
    rate=np.exp(np.clip(lr,-30,25));prob=-np.expm1(-rate)
    train=np.asarray(d['cbc_train'])>0;y=np.asarray(d['cbc_count'])>0
    logscore=np.where(y,np.log(-np.expm1(-rate)),-rate)
    unseen=np.bincount(ci[train],minlength=len(raw))[ci]==0
    if unseen.any():
        nodes,weights=np.polynomial.hermite.hermgauss(20)
        rates=np.exp(np.clip(lr[unseen,None]+np.sqrt(2)*sd*np.sqrt(1-1/len(raw))*nodes,-30,25))
        prob[unseen]=(-np.expm1(-rates))@(weights/np.sqrt(np.pi))
        ll=np.where(y[unseen,None],np.log(-np.expm1(-rates)),-rates)
        logscore[unseen]=logsumexp(ll+np.log(weights/np.sqrt(np.pi)),axis=1)
    summer=years[np.asarray(d['cbc_t'])]
    scopes=dict(train=train,spatial=np.asarray(d['cbc_spatial_holdout'])>0,
        temporal=np.asarray(d['cbc_temporal_holdout'])>0,establishment_train=train&(summer>=1951))
    result=dict(status='complete',unseen_circle_rows=int(unseen.sum()),scores={},
        scope='Conditional MAP predictions, unseen circle effects integrated; not posterior validation or an absolute founder census',
        parameters={k:np.asarray(v).tolist() for k,v in lat.items() if k.startswith('cbc_') and k!='cbc_circle_raw'})
    for name,keep in scopes.items():
        result['scores'][name]=dict(n=int(keep.sum()),observed_detection=float(y[keep].mean()),
            predicted_detection=float(prob[keep].mean()),mean_log_score=float(logscore[keep].mean()),
            brier_score=float(np.mean((prob[keep]-y[keep])**2)))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.out.with_suffix('.npz'),probability=prob,present=y,summer=summer,train=train,
        spatial=np.asarray(d['cbc_spatial_holdout']),temporal=np.asarray(d['cbc_temporal_holdout']))
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
