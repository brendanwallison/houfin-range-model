"""Bounded two-chain NUTS performance screen, never an inference certificate."""
import argparse,hashlib,json,time
from pathlib import Path
from src.model import hmc_common as hc
import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer import MCMC,NUTS
from numpyro.diagnostics import summary
from src.config_utils import load_age_model_config
from src.model.age_priors import build_model_2d
from src.model.checkpoints import load_map_params,auto_delta_params_to_latents
from src.model.data_loading import load_data
from checkpointed_nuts import run_checkpointed


def finite(v):
    if isinstance(v,dict):return {k:finite(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [finite(x) for x in v]
    if isinstance(v,np.ndarray):return finite(v.tolist())
    if isinstance(v,np.generic):return finite(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    return v


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--geometry',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--warmup',type=int,default=128);p.add_argument('--samples',type=int,default=128);p.add_argument('--max-depth',type=int,default=6)
    p.add_argument('--metric',choices=['diag','block'],default='diag');a=p.parse_args()
    if hc.PRECISION!='float64':raise ValueError('float64 required')
    a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'READY.json').exists():raise FileExistsError('screen retained')
    geometry=json.load(open(a.geometry/'READY.json'))
    if geometry.get('screen_kind')!='comparative_geometry_screen':raise ValueError('full geometry screen required')
    sha=hashlib.sha256((a.run/'map_checkpoint.pkl').read_bytes()).hexdigest()
    if geometry['checkpoint_sha256']!=sha:raise ValueError('different checkpoint')
    data=load_data(load_age_model_config()['input_dir'],target_device=jax.devices('gpu')[0],precision='float64',verbose=False)
    params,_=load_map_params(str(a.run));lat=auto_delta_params_to_latents(params)
    arrays,static=hc.split_data(data);kw={'arrays':arrays};model=hc.hide_deterministics(hc.array_model(build_model_2d,static))
    z,potential,post=hc.potential_with_args(model,kw,lat);names,x0,unravel=hc.flatten_sorted(z)
    if tuple(geometry['site_sizes'])!=names:raise ValueError('site layout mismatch')
    with np.load(a.geometry/'geometry.npz') as f:scale=jnp.asarray(f['conditional_prior_sd'])
    largest=max(abs(geometry['conditional_prior_scaled']['ritz_min']),abs(geometry['conditional_prior_scaled']['ritz_max']),1.)
    dense=False;mass=scale**2
    if a.metric=='block':
        core=tuple(n for n in names if n!='observer_route_eps');nuisance=tuple(n for n in names if n=='observer_route_eps')
        if not core or not nuisance:raise ValueError('expected ecological and observer blocks')
        index=np.cumsum([0]+[int(np.size(z[n])) for n in names])
        indices=lambda group:np.concatenate([np.arange(index[i],index[i+1]) for i,n in enumerate(names) if n in group])
        dense=[core];mass={core:jnp.diag(scale[indices(core)]**2),nuisance:scale[indices(nuisance)]**2}
    samples=[];extras=[];wall=[]
    for chain in range(2):
        start=x0
        if chain:
            direction=jax.random.normal(jax.random.PRNGKey(741),x0.shape)
            for attempt in range(12):
                trial=x0+(.25*.5**attempt/np.sqrt(largest))*scale*direction
                value=float(potential(unravel(trial),kw))
                if np.isfinite(value):start=trial;break
            else:raise ValueError('no finite second-chain start')
        kernel=NUTS(model,target_accept_prob=.9,max_tree_depth=a.max_depth,
            inverse_mass_matrix=mass,dense_mass=dense,step_size=1e-3,
            adapt_mass_matrix=True,adapt_step_size=True)
        identity={'checkpoint':sha,'geometry':hashlib.sha256((a.geometry/'READY.json').read_bytes()).hexdigest(),
            'implementation':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'checkpoint_engine':hashlib.sha256(Path(__file__).with_name('checkpointed_nuts.py').read_bytes()).hexdigest(),
            'metric':a.metric,'warmup':a.warmup,'samples':a.samples,'depth':a.max_depth,'chain':chain}
        fp=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        ss,ee,seconds=run_checkpointed(kernel,unravel(start),kw,jax.random.PRNGKey(20261101+chain),
            a.warmup,a.samples,a.out/f'chain{chain}',fp,post,chunk=4)
        wall.append(seconds);samples.append(ss);extras.append(ee)
        np.savez_compressed(a.out/f'chain{chain}_samples.npz',**ss)
        np.savez_compressed(a.out/f'chain{chain}_extra.npz',**ee)
        (a.out/f'chain{chain}_DONE.json').write_text(json.dumps({'wall_seconds':wall[-1],'checkpoint_sha256':sha})+'\n')
    combined={n:np.stack([s[n] for s in samples]) for n in samples[0]}
    stats=summary(combined,group_by_chain=True)
    steps=sum(float(e['num_steps'].sum()) for e in extras)
    per_site={}
    for n,s in stats.items():
        ess=np.asarray(s['n_eff']);rh=np.asarray(s['r_hat'])
        per_site[n]={'ess_min':float(np.nanmin(ess)),'ess_median':float(np.nanmedian(ess)),
            'split_rhat_max':float(np.nanmax(rh)),'split_rhat_median':float(np.nanmedian(rh)),
            'ess_min_per_postwarmup_leapfrog':float(np.nanmin(ess)/steps),
            'ess_min_per_total_wall_second':float(np.nanmin(ess)/sum(wall))}
    chains=[]
    for e,w in zip(extras,wall):
        energy=e['energy'];var=float(np.var(energy))
        chains.append({'divergences':int(e['diverging'].sum()),'draws':a.samples,
            'mean_acceptance':float(e['accept_prob'].mean()),'step_size_median':float(np.median(e['adapt_state.step_size'])),
            'mean_leapfrog_steps':float(e['num_steps'].mean()),
            'depth_cap_fraction':float(np.mean(e['num_steps']>=2**a.max_depth-1)),
            'E_BFMI':float(np.mean(np.diff(energy)**2)/var) if var else None,'wall_seconds_including_warmup':w})
    result={'run':str(a.run),'checkpoint_sha256':sha,'metric':a.metric,'warmup':a.warmup,'samples_per_chain':a.samples,'max_tree_depth':a.max_depth,
        'chains':chains,'site_diagnostics':per_site,
        'scope':'Bounded local HMC screening; near-MAP starts and capped trajectories do not certify global exploration. NumPyro n_eff and split Rhat, not rank-normalized bulk/tail ESS. Full inference needs longer, dispersed-chain checks and derived biological estimands.',
        'efficiency_denominator':'postwarmup num_steps sum for leapfrog efficiency; total wall time includes compile and warmup'}
    (a.out/'READY.json').write_text(json.dumps(finite(result),indent=2,allow_nan=False)+'\n')
    print('NUTS SCREEN READY',chains,flush=True)


if __name__=='__main__':main()
