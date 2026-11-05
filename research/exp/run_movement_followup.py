"""Successful permanent movement distances and local finite-propagule checks.

Reuses each fit's frozen forward primitives and actual ambient populations.
This is not a stochastic colonization model or seasonal movement estimate.
"""
import argparse,inspect,json,hashlib
from pathlib import Path
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from numpyro import handlers
from src.config_utils import load_age_model_config,load_data_config
from src.model import age_priors as ap,age_forward as af
from src.model.data_loading import load_data
from src.model.checkpoints import load_map_params,auto_delta_params_to_latents
from contribution_budget import audit_forward


def moment_origins(move,stay_adult,sa,arriver_survival,a,j):
    def total(aa,jj):
        aa_post,_,ja=move(aa,jj)
        return jnp.sum((aa_post-aa*stay_adult)*sa+ja*arriver_survival)
    ca,cj=jax.grad(total,argnums=(0,1))(jnp.ones_like(a),jnp.ones_like(j))
    return ca*a,cj*j


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'READY.json').exists():raise FileExistsError('completed followup retained')
    cfg=load_age_model_config();data=load_data(cfg['input_dir'],target_device=jax.devices('cpu')[0],precision='float64',verbose=False)
    params,_=load_map_params(str(a.run));lat=auto_delta_params_to_latents(params);captured=[];old=ap.forward_sim_age_structured
    def record(*aa,**kw):
        bound=inspect.signature(old).bind(*aa,**kw);bound.apply_defaults();result=old(*aa,**kw);captured.append((bound.arguments,result));return result
    ap.forward_sim_age_structured=record
    try:handlers.seed(handlers.condition(ap.build_model_2d,data=lat),rng_seed=104)(data=data,prior_scale=1.)
    finally:ap.forward_sim_age_structured=old
    if len(captured)!=1:raise ValueError('multiple trajectories')
    x,states=captured[0];budget,functions=audit_forward(af,x,states);setup,disease=functions
    if max(np.max(np.abs(budget[n])) for n in ['replay_adult_error','replay_juvenile_error','BIDE_error'])>1e-9:raise ValueError('replay failed')
    lr,lc=x['land_rows'],x['land_cols'];shape=x['land_mask'].shape;zero=jnp.zeros(shape);years=np.asarray(data['years']);modern=np.flatnonzero((years>=2010)&(years<=2025))
    res=float(load_data_config()['grid']['target_res_m'])/1000.
    ly,lx=x['adult_fft_kernel'].shape
    dy=jnp.minimum(jnp.arange(ly),ly-jnp.arange(ly));dx=jnp.minimum(jnp.arange(lx),lx-jnp.arange(lx))
    distance=res*jnp.sqrt(dy[:,None]**2+dx[None,:]**2)
    ak=jnp.fft.ifft2(x['adult_fft_kernel'].astype(jnp.complex128)).real
    jk=jnp.fft.ifft2(x['juvenile_fft_kernel_stack'].astype(jnp.complex128),axes=(-2,-1)).real
    filters={'distance_km':distance,'tail_300km':(distance>300).astype(float),'tail_1000km':(distance>1000).astype(float)}
    results={}
    for name,filt in filters.items():
        afilt=jnp.fft.fft2(ak*filt);jfilt=jnp.fft.fft2(jk*filt,axes=(-2,-1))
        def step(_,t):
            aa=states[1][t-1];jj=states[2][t-1]
            sa,sj,f,k,c,move,diag,kw=setup(t,aa,jj)
            aa_post,js,ja=move(aa,jj);sa,sj,dm=disease(t,sa,sj,aa_post,js,ja)
            one=jnp.ones_like(aa)
            unit=af.dispersal_step_age_structured(one,one,k,return_adult_arrivals=True,**kw)
            stay=unit[0]-unit[3]
            if getattr(af,'LOCAL_SETTLEMENT',False):
                # Compose each short placement offset with the long-leg kernel
                # BEFORE filtering distance. Filtering only the first leg would
                # incorrectly label main-leg distance as final displacement.
                from src.model.local_arrival_settlement import offset_fractions,shift
                sy=jnp.where(jnp.arange(ly)<=ly//2,jnp.arange(ly),jnp.arange(ly)-ly)
                sx=jnp.where(jnp.arange(lx)<=lx//2,jnp.arange(lx),jnp.arange(lx)-lx)
                components=offset_fractions(k,af.LOCAL_SETTLEMENT_FRACTION)
                def total(a1,j1):
                    value=0.
                    for oy,ox,part in components:
                        dist=res*jnp.sqrt((sy[:,None]+oy)**2+(sx[None,:]+ox)**2)
                        # Successful off-cell displacement excludes exact returns
                        # even if the bird moved on its first leg.
                        ff=dist if name=='distance_km' else (dist>(300 if name=='tail_300km' else 1000)).astype(float)
                        mk=dict(kw,adult_fft_kernel=jnp.fft.fft2(ak*ff),juvenile_fft_kernel_stack=jnp.fft.fft2(jk*ff,axes=(-2,-1)),local_settlement=False)
                        ap1,_,ja1=af.dispersal_step_age_structured(a1,j1,k,**mk)
                        value=value+jnp.sum(shift((ap1-a1*stay)*part,oy,ox)*sa+shift(ja1*part,oy,ox)*dm)
                    return value
                ca,cj=jax.grad(total,argnums=(0,1))(one,one);ma,mj=ca*aa,cj*jj
            else:
                moment_kw=dict(kw,adult_fft_kernel=afilt,juvenile_fft_kernel_stack=jfilt)
                def mm(a1,j1):return tuple(v*x['land_mask'] for v in af.dispersal_step_age_structured(a1,j1,k,**moment_kw))
                ma,mj=moment_origins(mm,stay,sa,dm,aa,jj)
            return None,jnp.stack([ma[lr,lc],mj[lr,lc]])
        _,out=jax.jit(lambda:jax.lax.scan(step,None,jnp.asarray(modern)))()
        results[name]=np.asarray(jax.block_until_ready(out))
    # Independent local establishment perturbations: no movement, no release
    # exemption, constant modern mean environment, actual reproductive law.
    mean=lambda key:jnp.mean(x[key][jnp.asarray(modern)],axis=0)
    sa,sj,f,k,c=(mean(key) for key in ['Sa_flat','Sj_flat','Fmax_flat','K_flat','c_flat'])
    local={}
    for seed in [.01,.1,1.,10.]:
        initial=(k*seed*.5,k*seed*.5)
        def step(pair,_):
            aa,jj=pair;ss,js=sa,sj
            if x.get('disease_m_amp') is not None:
                amp=mean('disease_m_amp');n=aa+jj;h=x.get('disease_hill_n',1.)
                hn=(jnp.maximum(n,0.)+1e-12)**h
                rem=1.-amp*hn/(hn+x.get('disease_k_half',1.)**h);ss,js=sa*rem,sj*rem
            nxt=af.reproduction_age_structured(aa,jj,jnp.zeros_like(aa),ss,js,f,k,c,x['allee_gamma'],
                allee_relative=x.get('allee_relative',False),yearling_breeding=x.get('yearling_breeding',False),veit=x.get('veit_reproduction',False))
            nxt=tuple(jnp.maximum(v,0.) for v in nxt)
            return nxt,nxt[0]+nxt[1]
        _,out=jax.jit(lambda:jax.lax.scan(step,initial,None,length=80))()
        local[str(seed)]=np.asarray(jax.block_until_ready(out))
    import rasterio
    from rasterio.warp import transform
    with rasterio.open(load_data_config()['grid']['ref_raster']) as src:
        xx,yy=rasterio.transform.xy(src.transform,np.asarray(lr),np.asarray(lc));lon,latitudes=map(np.asarray,transform(src.crs,'EPSG:4326',xx,yy))
    regions={'all':np.ones(len(lr),bool),'west':lon<-100,'plains':(lon>=-100)&(lon<-95),'broad_plains':(lon>=-105)&(lon<-95)&(latitudes>=25)&(latitudes<49),'east':(lon>=-80)&(latitudes>=25)&(latitudes<49)}
    export=budget['biological_successful_breeder_export' if x.get('process_noise') is not None else 'successful_breeder_export'][modern];summary={}
    for reg,mask in regions.items():
        den=float(export[:,mask].sum());s={name:float(v[:,:,mask].sum()/den) if den>0 else None for name,v in results.items()}
        s['denominator']='successful off-cell adult survivors plus juvenile recruits to next breeding census; not attempted departure distance'
        s['local_propagule_final_over_K_median']={key:float(np.median(value[-1,mask]/np.maximum(np.asarray(k)[mask],1e-30))) for key,value in local.items()}
        s['local_propagule_cells_above_001K_after_80y']={key:float(np.mean(value[-1,mask]>.01*np.asarray(k)[mask])) for key,value in local.items()}
        summary[reg]=s
    np.savez_compressed(a.out/'movement_moments.npz',years=years[modern],land_rows=np.asarray(lr),land_cols=np.asarray(lc),export=export,**results)
    np.savez_compressed(a.out/'local_propagules.npz',K=np.asarray(k),**{'seed_'+key:v for key,v in local.items()})
    result={'run':str(a.run),'regions':summary,'scope':'Actual permanent-movement survival-weighted endpoint distance; local propagule checks are deterministic, not probability-of-establishment estimates.',
        'source_sha256':hashlib.sha256(Path(af.__file__).read_bytes()).hexdigest(),'checkpoint_sha256':hashlib.sha256((a.run/'map_checkpoint.pkl').read_bytes()).hexdigest()}
    pn=cfg['population_model'].get('process_noise',{})
    if x.get('process_noise') is not None:
        eps=np.asarray(x['process_noise']);base_states=old(**dict(x,process_noise=None))
        full_n=np.asarray(states[0])[:,np.asarray(lr),np.asarray(lc)]
        base_n=np.asarray(base_states[0])[:,np.asarray(lr),np.asarray(lc)]
        active=np.asarray(data['pn_active'])>0
        effects={}
        for reg,mask in regions.items():
            sel=np.ix_(active,mask);d=np.log(full_n[sel]+1e-6)-np.log(base_n[sel]+1e-6)
            effects[reg]=dict(log_innovation_rms=float(np.sqrt(np.mean(eps[sel]**2))),
                mean_absolute_log_population_change_without_innovations=float(np.mean(np.abs(d))),
                final_population_without_over_with=float(base_n[-1,mask].sum()/max(full_n[-1,mask].sum(),1e-30)),
                cumulative_log_innovation_q10_50_90=np.quantile(eps[:,mask].sum(0),[.1,.5,.9]).tolist())
        result['abundance_innovations']=dict(regions=effects,
            scope='Fixed-parameter zero-innovation replay; movement distances weight biological survivors before the state discrepancy. Full fitted origin C is saved separately.')
    if pn.get('enabled') and pn.get('target')=='fecundity':
        # Report the complete reproductive-growth contribution, alongside a
        # fixed-parameter zero-variation replay. Never call only the stripped
        # skeleton's lambda the growth of the fitted varying model.
        raw=jnp.asarray(lat['process_noise_u'])*data['pn_active'][:,None]
        if 'pn_P' in data:raw=data['pn_P']@raw
        eps=(float(pn.get('sigma',.1))*raw)@data['pn_S'].T
        base_f=x['Fmax_flat']*jnp.exp(-eps)
        no_variation=dict(x,Fmax_flat=base_f)
        base_states=old(**no_variation)
        full_n=np.asarray(states[0])[:,np.asarray(lr),np.asarray(lc)]
        base_n=np.asarray(base_states[0])[:,np.asarray(lr),np.asarray(lc)]
        active=np.asarray(data['pn_active'])>0
        actual_growth=np.asarray(x['Sa_flat']+x['Sj_flat']*x['Fmax_flat'])
        base_growth=np.asarray(x['Sa_flat']+x['Sj_flat']*base_f)
        effect={}
        for reg,mask in regions.items():
            sel=np.ix_(active,mask);d=np.log(full_n[sel]+1e-6)-np.log(base_n[sel]+1e-6)
            effect[reg]=dict(log_output_rms=float(np.sqrt(np.mean(np.asarray(eps)[sel]**2))),
              mean_absolute_log_population_change_without_variation=float(np.mean(np.abs(d))),
              final_population_without_over_with=float(base_n[-1,mask].sum()/max(full_n[-1,mask].sum(),1e-30)),
              actual_potential_growth_mean=float(np.mean(actual_growth[sel])),
              zero_variation_potential_growth_mean=float(np.mean(base_growth[sel])))
        result['reproductive_variation']=dict(scope='Same fitted habitat, rates, movement and founders; only the fitted reproductive variation is zeroed. Regional means are unweighted land-cell-years.',regions=effect)
    (a.out/'READY.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
