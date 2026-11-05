"""Exact-target HMC geometry screening using frozen posterior machinery.

No dense 19k-parameter Hessian, no removal/fixing of observer nuisance parameters,
and no substitution of constrained MAP curvature for HMC target curvature.
"""
import argparse,hashlib,json,time
from pathlib import Path
from src.model import hmc_common as hc
import jax
import jax.numpy as jnp
import numpy as np
from numpyro import handlers
from numpyro.distributions.transforms import biject_to
from src.config_utils import load_age_model_config
from src.model.age_priors import build_model_2d
from src.model.checkpoints import load_map_params,auto_delta_params_to_latents
from src.model.data_loading import load_data
from hmc_geometry_math import lanczos,spectrum_report


def write(path,value):
    tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=48);p.add_argument('--prior-draws',type=int,default=256)
    p.add_argument('--device',choices=['cpu','gpu'],default='gpu');p.add_argument('--rough-directions',type=int,default=4)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'READY.json').exists():raise FileExistsError('completed geometry audit retained')
    if hc.PRECISION!='float64':raise ValueError('HMC screening requires float64 target')
    if not json.load(open(a.run/'refine_report.json')).get('valid'):raise ValueError('unaccepted point')
    device=jax.devices(a.device)[0];cfg=load_age_model_config()
    data=load_data(cfg['input_dir'],target_device=device,precision='float64',verbose=False)
    params,checkpoint=load_map_params(str(a.run));lat=auto_delta_params_to_latents(params)
    arrays,static=hc.split_data(data);kw={'arrays':arrays}
    model=hc.hide_deterministics(hc.array_model(build_model_2d,static))
    z,potential,_=hc.potential_with_args(model,kw,lat)
    names,x0,unravel=hc.flatten_sorted(z);dimension=int(x0.size)
    U=lambda x,k:potential(unravel(x),k)
    vg=jax.jit(jax.value_and_grad(U))
    hv=jax.jit(lambda x,v,k:jax.jvp(lambda xx:jax.grad(U)(xx,k),(x,),(v,))[1])
    t=time.time();u,g=jax.block_until_ready(vg(x0,kw));compile_seconds=time.time()-t
    u=float(u);g=np.asarray(g,float)
    if not(np.isfinite(u) and np.isfinite(g).all()):raise ValueError('nonfinite posterior potential or gradient')
    times=[]
    for _ in range(3):
        t=time.time();jax.block_until_ready(vg(x0,kw));times.append(time.time()-t)
    # Standardize by conditional-prior scale IN UNCONSTRAINED coordinates.
    # Sampling and inverse transforms reuse NumPyro; constrained sd is not used.
    tr=handlers.trace(handlers.seed(handlers.condition(hc.hide_deterministics(build_model_2d),data=lat),rng_seed=20261101)).get_trace(data=data,prior_scale=1.)
    scales={};scale_method={}
    for i,name in enumerate(names):
        site=tr[name];dist=site['fn'];samples=dist.sample(jax.random.PRNGKey(i+729),sample_shape=(a.prior_draws,))
        unconstrained=biject_to(dist.support).inv(samples)
        sd=np.asarray(jnp.std(unconstrained,axis=0,ddof=1),float)
        if sd.shape!=np.asarray(z[name]).shape:raise ValueError('prior scale layout mismatch '+name)
        if not(np.isfinite(sd).all() and np.all(sd>0)):raise ValueError('invalid prior scales '+name)
        scales[name]=jnp.asarray(sd);scale_method[name]={'prior_class':type(dist).__name__,'sd_min':float(sd.min()),'sd_max':float(sd.max())}
    _,scale,_=hc.flatten_sorted(scales);scale=np.asarray(scale,float)
    del tr
    sizes={n:int(np.size(z[n])) for n in names};offsets=np.cumsum([0]+[sizes[n] for n in names])
    result={'run':str(a.run),'dimension':dimension,'site_sizes':sizes,'precision':'float64',
        'target':'NumPyro unconstrained negative log posterior, nominal priors, transformation Jacobians included',
        'potential_at_start':u,'gradient_norm':float(np.linalg.norm(g)),
        'gradient_max_abs':float(np.max(np.abs(g))),'gradient_norm_by_site':{n:float(np.linalg.norm(g[offsets[i]:offsets[i+1]])) for i,n in enumerate(names)},
        'grad_seconds_median':float(np.median(times)),'compile_seconds':compile_seconds,
        'scaling':'diagonal conditional-prior standard deviations after exact NumPyro inverse transforms; not a posterior whitening or adopted mass matrix',
        'scale_draws':a.prior_draws,'scale_by_site':scale_method,
        'scope':'Geometry screen at a fitted point. Not a posterior sample or proof of HMC suitability. Negative curvature at a nonstationary point does not by itself prove a saddle-point MAP.'}
    result['screen_kind']='smoke_only' if a.steps<24 or a.prior_draws<128 else 'comparative_geometry_screen'
    write(a.out/'PARTIAL.json',result)
    def operator(v):return np.asarray(jax.block_until_ready(hv(x0,jnp.asarray(v),kw)),float)
    t=time.time();raw=lanczos(operator,dimension,a.steps)
    result['raw_unconstrained']=spectrum_report(raw)
    normalized=lanczos(lambda v:scale*operator(scale*v),dimension,a.steps)
    result['conditional_prior_scaled']=spectrum_report(normalized)
    result['spectrum_seconds']=time.time()-t
    result['directions']={}
    vectors=[('largest_curvature',normalized['vectors'][:,-1]),('smallest_ritz',normalized['vectors'][:,0])]
    rng=np.random.default_rng(20261101)
    for i in range(max(0,a.rough_directions-2)):
        q=rng.normal(size=dimension);q/=np.linalg.norm(q);vectors.append((f'random_{i}',q))
    for label,q in vectors[:a.rough_directions]:
        v=scale*q;h=operator(v);curv=float(v@h)
        # Avoid arbitrarily huge perturbations when sampled curvature is near zero.
        v=v/np.sqrt(max(abs(curv),1.));h=operator(v);curv=float(v@h)
        rows=[];dot=float(g@v)
        for delta in [1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8]:
            up,gp=jax.block_until_ready(vg(x0+delta*v,kw));um,gm=jax.block_until_ready(vg(x0-delta*v,kw))
            up,um=float(up),float(um);gp,gm=np.asarray(gp),np.asarray(gm)
            finite=bool(np.isfinite(up) and np.isfinite(um) and np.isfinite(gp).all() and np.isfinite(gm).all())
            row={'delta':delta,'finite':finite}
            if finite:
                fd=(gp-gm)/(2*delta)
                row.update(potential_linear_remainder_plus=up-u-delta*dot,
                    potential_linear_remainder_minus=um-u+delta*dot,
                    quadratic_prediction=.5*delta**2*curv,
                    directional_derivative_error=(up-um)/(2*delta)-dot,
                    hvp_finite_difference_relative_error=float(np.linalg.norm(fd-h)/max(np.linalg.norm(fd),np.linalg.norm(h),1e-20)))
            rows.append(row)
        loading={n:float(np.sum(q[offsets[i]:offsets[i+1]]**2)) for i,n in enumerate(names)}
        neighborhood=[]
        for radius in [-1.,-.25,.25,1.]:
            point=x0+radius*v;uu,gg=jax.block_until_ready(vg(point,kw))
            finite=bool(np.isfinite(float(uu)) and np.isfinite(np.asarray(gg)).all())
            entry={'local_width_units':radius,'finite':finite}
            if finite:
                hh=np.asarray(jax.block_until_ready(hv(point,jnp.asarray(v),kw)),float)
                entry['directional_curvature']=float(v@hh) if np.isfinite(hh).all() else None
                entry['potential_change']=float(uu)-u
            neighborhood.append(entry)
        result['directions'][label]={'directional_curvature':curv,'site_loadings':dict(sorted(loading.items(),key=lambda x:-x[1])[:8]),'probes':rows,'nearby_curvature':neighborhood}
        write(a.out/'PARTIAL.json',result)
    result['inactive_structure_notes']={'veit_departure':cfg['population_model'].get('dispersal_departure',{}).get('form')=='veit',
        'note':'Prior-only nuisance directions remain in the actual target. Their presence is not evidence of ecological information. Compare block loadings and active structural switches.'}
    np.savez_compressed(a.out/'geometry.npz',unconstrained_point=np.asarray(x0),conditional_prior_sd=scale,
        raw_ritz_values=raw['values'],scaled_ritz_values=normalized['values'],scaled_extreme_vectors=normalized['vectors'][:,[0,-1]])
    result['checkpoint_sha256']=hashlib.sha256((a.run/'map_checkpoint.pkl').read_bytes()).hexdigest()
    result['implementation_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write(a.out/'READY.json',result)
    print('GEOMETRY READY',dimension,result['raw_unconstrained']['ritz_min'],result['raw_unconstrained']['ritz_max'],flush=True)


if __name__=='__main__':main()
