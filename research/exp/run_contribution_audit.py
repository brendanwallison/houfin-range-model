"""Replay an accepted frozen fit, validate origin budgets, save S1--S4/S6.

Execute with the frozen code as cwd and first project import path. Production
source/checkpoints are never changed. All annotations are diagnostic outputs.
"""
import argparse,hashlib,inspect,json
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
from contribution_budget import audit_forward,isolation_replay,select_main_forward,fft_floor_active


def clean(v):
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    return v


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--isolation-start',type=int,default=2010)
    p.add_argument('--accept-unrefined',default=None,metavar='REASON',
                   help='run without refine_report.json (e.g. a Laplace-marginal fit spliced into a checkpoint); the reason is recorded')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    implementation_paths=[Path(__file__),Path(__file__).with_name('contribution_budget.py')]
    implementation_start=[hashlib.sha256(p.read_bytes()).hexdigest() for p in implementation_paths]
    if (a.out/'READY.json').exists():raise FileExistsError('completed audit retained')
    if a.accept_unrefined is None:
        if not json.load(open(a.run/'refine_report.json')).get('valid'):raise ValueError('unaccepted fit')
    cfg=load_age_model_config();dcfg=load_data_config()
    if not ap.YEARLING:raise NotImplementedError('breeding metric requires annual maturation')
    data=load_data(cfg['input_dir'],target_device=jax.devices('cpu')[0],precision='float64',verbose=False)
    params,_=load_map_params(str(a.run));lat=auto_delta_params_to_latents(params)
    original=ap.forward_sim_age_structured;captured=[]
    def record(*aa,**kw):
        bound=inspect.signature(original).bind(*aa,**kw);bound.apply_defaults()
        result=original(*aa,**kw)
        captured.append((dict(bound.arguments),result))
        return result
    ap.forward_sim_age_structured=record
    try:
        handlers.seed(handlers.condition(ap.build_model_2d,data=lat),rng_seed=104)(data=data,prior_scale=1.)
    finally:ap.forward_sim_age_structured=original
    spin_spec=dict(cfg['population_model'].get('spinup') or {})
    x,states,spin_info=select_main_forward(captured,spin_spec if (getattr(ap,'SPINUP',False) and spin_spec.get('enabled')) else None)
    if int(x['time'])!=len(np.asarray(data['years'])):raise ValueError('main forward call does not span the model years')
    states=jax.tree.map(lambda v:jax.block_until_ready(v),states)
    print('FROZEN TRAJECTORY REPLAYED',spin_info,flush=True)
    arrays,setup=audit_forward(af,x,states)
    lr,lc=np.asarray(x['land_rows']),np.asarray(x['land_cols']);years=np.asarray(data['years'])
    scale=max(float(np.max(np.asarray(states[0]))),1.)
    checks={n:float(np.max(np.abs(arrays[n]))) for n in ['BIDE_error','replay_adult_error','replay_juvenile_error','survival_identity_error']}
    # Origin sums (floor-free linear operator) + explicit numerical terms (clip, FFT floor) = realised destination totals.
    checks['breeder_origin_destination_sum_error']=float(np.max(np.abs((arrays['breeder_contribution']+arrays['numerical_adult_clip_gain']+arrays['numerical_fft_floor_breeders']).sum(1)-arrays['next_adult'].sum(1))))
    checks['postbirth_origin_destination_sum_error']=float(np.max(np.abs((arrays['total_descendants_origin']+arrays['numerical_unparented_births']+arrays['numerical_adult_clip_gain']+arrays['numerical_fft_floor_postbirth']).sum(1)-(arrays['next_adult']+arrays['next_juvenile']).sum(1))))
    # Relative replay errors (task contract: <= 1e-8 of the year's largest cell state).
    na_l,nj_l=np.asarray(states[1])[:,lr,lc],np.asarray(states[2])[:,lr,lc]
    rel={'replay_adult_error_relative':float(np.max(np.abs(arrays['replay_adult_error']))/max(float(np.max(na_l)),1e-300)),
         'replay_juvenile_error_relative':float(np.max(np.abs(arrays['replay_juvenile_error']))/max(float(np.max(nj_l)),1e-300)),
         'BIDE_error_relative':float(np.max(np.abs(arrays['BIDE_error'])))/max(float(np.max(na_l+nj_l)),1e-300)}
    fd=arrays['numerical_fft_floor_destination'];post=arrays['input_adult']+arrays['input_juvenile']
    floor_info={'active':bool(fft_floor_active(af,jnp.asarray(x['Sa_flat']).dtype)),'fft_floor_ulps':af.FFT_FLOOR_ULPS,
        'fft_floor_smooth':bool(getattr(af,'FFT_FLOOR_SMOOTH',False)),'fft_pad':af.FFT_PAD_VALUE,
        'treatment':'replay uses the floored movement; origin attribution uses the floor-free linear operator of the same '
                    'primitive; realised minus floor-free arrivals enter BIDE as numerical_fft_floor_destination and the '
                    'breeder/post-birth origin sums as numerical_fft_floor_breeders/_postbirth',
        'gross_abs_destination_total':float(np.abs(fd).sum()),'net_destination_total':float(fd.sum()),
        'max_abs_cell_year':float(np.abs(fd).max()),
        'max_annual_gross_abs_over_population':float(np.max(np.abs(fd).sum(1)/np.maximum(post.sum(1),1e-300))),
        'gross_abs_breeders_total':float(np.abs(arrays['numerical_fft_floor_breeders']).sum()),
        'max_annual_gross_abs_breeders_over_next_adults':float(np.max(np.abs(arrays['numerical_fft_floor_breeders']).sum(1)/np.maximum(arrays['next_adult'].sum(1),1e-300)))}
    print('RELATIVE',rel,'FLOOR',floor_info,flush=True)
    dtypes={n:str(np.asarray(x[n]).dtype) for n in ['Sa_flat','Sj_flat','initpop_latent','adult_fft_kernel']}
    print('ACCOUNTING',checks,'DTYPES',dtypes,'scale',scale,flush=True)
    kernel_eps=np.finfo(np.asarray(x['adult_fft_kernel']).real.dtype).eps
    tol=max(1e-10,8*kernel_eps)*scale
    annual_total=float(np.max((arrays['next_adult']+arrays['next_juvenile']).sum(1)))
    tolerances={key:tol for key in checks}
    for key in ['breeder_origin_destination_sum_error','postbirth_origin_destination_sum_error']:
        tolerances[key]=max(1e-10,8*kernel_eps)*max(annual_total,1.)
    if any(checks[key]>value for key,value in tolerances.items()):raise ValueError('trajectory/accounting mismatch '+str(checks))
    if max(checks['replay_adult_error'],checks['replay_juvenile_error'])>1e-10*scale:
        raise ValueError('frozen trajectory parity failed')
    if max(rel['replay_adult_error_relative'],rel['replay_juvenile_error_relative'])>1e-8:
        raise ValueError('relative replay parity failed '+str(rel))
    checks.update(rel)
    for n in ['successful_breeder_export','immigrant_breeders','immigration','successful_emigration','journey_mortality_origin']:
        checks[n+'_minimum']=float(np.min(arrays[n]))
        if checks[n+'_minimum'] < -tol:raise ValueError('negative flow '+n)
    arrays.update(years=years,land_rows=lr,land_cols=lc)
    np.savez_compressed(a.out/'annual_budget.npz',**arrays)
    import rasterio
    from rasterio.warp import transform
    with rasterio.open(dcfg['grid']['ref_raster']) as src:
        xx,yy=rasterio.transform.xy(src.transform,lr,lc)
        lon,latitudes=map(np.asarray,transform(src.crs,'EPSG:4326',xx,yy))
    regions={'all':np.ones(len(lr),bool),'west':lon<-100,
        'plains':(lon>=-100)&(lon<-95),'broad_plains':(lon>=-105)&(lon<-95)&(latitudes>=25)&(latitudes<49),
        'east':(lon>=-80)&(latitudes>=25)&(latitudes<49)}
    # CEC Level I ecoregions (project rule: regional summaries use ecoregions; the longitude bands above are legacy,
    # kept for continuity with earlier audits and the isolation replays).
    cec_names={7:'Marine West Coast Forest',11:'Mediterranean California',6:'NW Forested Mountains',10:'N. Am. Deserts',
        13:'Temperate Sierras',12:'S. Semiarid Highlands',9:'Great Plains',3:'Taiga',5:'Northern Forests',
        8:'Eastern Temperate Forests',14:'Tropical Dry Forests',15:'Tropical Humid Forests',2:'Tundra',4:'Hudson Plain'}
    cec_path=Path(dcfg['grid']['ref_raster']).parent/'regions'/'cec_level1_54km.tif'
    if cec_path.exists():
        with rasterio.open(cec_path) as src:
            l1=np.asarray(src.read(1))[lr,lc].astype(int)
        for code,name in cec_names.items():
            if (l1==code).sum()>=5:regions['cec:'+name]=l1==code
    else:print('WARNING no CEC Level I raster at',cec_path,'- ecoregion summaries skipped',flush=True)
    eras={'early':(1902,1915),'release':(1940,1955),'bbs_early':(1966,1979),
        'middle':(1980,1994),'recent':(1995,2009),'modern':(2010,2025)}
    fields_path=Path(cfg['path_diagnostics_dir'])/'07_source_sink_fields.npz' if cfg.get('path_diagnostics_dir') else None
    # Accepted-run viz_ref location comes from frozen run name convention.
    step=cfg.get('_second_wave',{}).get('step')
    step=step or a.run.name.split('_grid54_')[-1].removesuffix('_quick90')
    root=Path(cfg['input_dir']).parents[4]/'research/grid54'
    # Derive project root from accepted checkpoint root, not the deeply nested input path.
    root=a.run.parents[2]/'research/grid54'
    fields_path=root/'runs'/step/'out/viz_ref/07_source_sink_fields.npz'
    cached={}
    if fields_path.exists():
        with np.load(fields_path) as z:
            cached={n:np.asarray(z[n])[lr,lc] for n in ['allee_dead_mask','lam_fundamental_modern','K_modern']}
    summary={'run':str(a.run),'step':step,'checks':checks,'tolerance':tol,'check_tolerances':tolerances,'dtypes':dtypes,
        'accepted_unrefined':a.accept_unrefined,'spinup':spin_info,'fft_floor':floor_info,
        'mechanisms':{'adult_survival_dd':x.get('survival_dd_c') is not None,
                      'native_departure_scale':None if x.get('departure_scale') is None else
                          [float(np.min(np.asarray(x['departure_scale']))),float(np.max(np.asarray(x['departure_scale'])))],
                      'disease_survival':x.get('disease_m_amp') is not None,'veit_departure':bool(x.get('veit_departure')),
                      'yearling_breeding':bool(x.get('yearling_breeding')),'nest_readout_only':x.get('nest_cells') is not None},
        'config':__import__('os').environ.get('AGE_MODEL_CONFIG'),
        'numerical_adjustments':{'adult_clip_gain_total':float(arrays['numerical_adult_clip_gain'].sum()),
            'unparented_births_total':float(arrays['numerical_unparented_births'].sum())},
        'census':'C_breeding: previous summer breeders + their actual young, surviving to next summer breeders; rows labeled by destination year',
        'stage_metric':'C_state: stage-specific postbirth column contributions weighted by actual adult/juvenile proportions, following Runge 2006 multiple-stage prescription',
        'source':'https://doi.org/10.1086/503531','regions':{},'isolation':{}}
    if x.get('process_noise') is not None:
        summary['process_correction']={'enabled':True,
            'interpretation':'Full fitted C includes the end-transition multiplier. Biological C and adjustment are saved separately; effective survival excludes the multiplier.',
            'net_adjustment':float(arrays['process_adjustment'].sum()),
            'gross_absolute_adjustment':float(np.abs(arrays['process_adjustment']).sum())}
    for reg,mask in regions.items():
        summary['regions'][reg]={}
        for era,(lo,hi) in eras.items():
            tt=(years>=lo)&(years<=hi)&(np.arange(len(years))>0)
            # Exogenous release mixes non-parental age cohorts: omit those cell-
            # transitions from breeding-contribution summaries, retain in raw NPZ.
            sel=tt[:,None]&mask[None]& (arrays['pulse']==0)
            adult=np.where(sel,arrays['input_adult'],0.)
            state=np.where(sel,arrays['input_adult']+arrays['input_juvenile'],0.)
            def ratio(n,d):
                den=float(np.where(sel,d,0.).sum());return float(np.where(sel,n,0.).sum()/den) if den>0 else None
            b={'C_breeding':ratio(arrays['breeder_contribution'],arrays['input_adult']),
               'R_breeding':ratio(arrays['retained_breeders'],arrays['input_adult']),
               'successful_export_per_breeder':ratio(arrays['successful_breeder_export'],arrays['input_adult']),
               'immigrant_share_of_next_breeders':ratio(arrays['immigrant_breeders'],arrays['next_breeders']),
               'C_state':ratio(arrays['total_descendants_origin'],arrays['input_adult']+arrays['input_juvenile']),
               'births':float(np.where(sel,arrays['births'],0.).sum()),
               'ordinary_deaths':float(np.where(sel,arrays['local_deaths_destination'],0.).sum()),
               'journey_deaths':float(np.where(sel,arrays['journey_mortality_origin'],0.).sum()),
               'numerical_edge_loss':float(np.where(sel,arrays['numerical_edge_loss_origin'],0.).sum()),
               'numerical_fft_floor_destination':float(np.where(sel,arrays['numerical_fft_floor_destination'],0.).sum()),
               'gross_immigration':float(np.where(sel,arrays['immigration'],0.).sum()),
               'gross_successful_emigration':float(np.where(sel,arrays['successful_emigration'],0.).sum())}
            if x.get('process_noise') is not None:
                b.update(biological_C_breeding=ratio(arrays['biological_breeder_contribution'],arrays['input_adult']),
                         process_adjustment=float(np.where(sel,arrays['process_adjustment'],0.).sum()),
                         absolute_process_adjustment=float(np.where(sel,np.abs(arrays['process_adjustment']),0.).sum()))
            b['effective_adult_survival']=ratio(arrays['input_adult']*arrays['origin_adult_survival'],arrays['input_adult'])
            b['effective_juvenile_survival']=ratio(arrays['input_juvenile']*arrays['origin_juvenile_recruitment'],arrays['input_juvenile'])
            b['local_total_replacement']=ratio(np.nan_to_num(arrays['local_total_replacement'])*(arrays['input_adult']+arrays['input_juvenile']),arrays['input_adult']+arrays['input_juvenile'])
            b['apparent_total_growth']=ratio(arrays['next_adult']+arrays['next_juvenile'],arrays['input_adult']+arrays['input_juvenile'])
            bd=np.where(sel,arrays['births']-arrays['local_deaths_destination']-arrays['journey_mortality_origin'],0.).sum(0)
            net=np.where(sel,arrays['immigration']-arrays['successful_emigration'],0.).sum(0)
            b['BIDE_quadrants_breeder_weight']={label:float(den.sum()/adult.sum()) if adult.sum()>0 else None
                for label,den in [(label,adult.sum(0)[condition]) for label,condition in {
                    'demographic_surplus_net_export':(bd>0)&(net<0),
                    'demographic_deficit_net_import':(bd<0)&(net>0),
                    'demographic_surplus_net_import':(bd>0)&(net>0),
                    'demographic_deficit_net_export':(bd<0)&(net<0)}.items()]}
            # Ratios of summed contributions for each cell, then classify cells;
            # avoid classifying year-by-year near-equilibrium rounding noise.
            denom=adult.sum(0);num=np.where(sel,arrays['breeder_contribution'],0.).sum(0)
            cr=np.divide(num,denom,out=np.full_like(num,np.nan),where=denom>1e-20)
            b['breeder_weighted_sink_share']={str(eps):float(denom[cr<1-eps].sum()/denom.sum()) if denom.sum()>0 else None for eps in [1e-6,.01,.05]}
            b['occupied_cell_sink_share']={str(eps):float(np.mean(cr[mask&(denom>1e-20)]<1-eps)) if np.any(mask&(denom>1e-20)) else None for eps in [1e-6,.01,.05]}
            localnum=np.where(sel,np.nan_to_num(arrays['local_breeder_replacement'])*arrays['input_adult'],0.).sum(0)
            local=np.divide(localnum,denom,out=np.full_like(num,np.nan),where=denom>1e-20)
            ret=np.where(sel,arrays['retained_breeders'],0.).sum(0)
            rr=np.divide(ret,denom,out=np.full_like(ret,np.nan),where=denom>1e-20)
            b['retention_sink_but_contributor_breeder_share']=float(denom[(rr<.99)&(cr>1.01)].sum()/denom.sum()) if denom.sum()>0 else None
            b['no_movement_replacement_below_099_breeder_share']=float(denom[local<.99].sum()/denom.sum()) if denom.sum()>0 else None
            if era=='modern' and cached:
                b['potential_sink_land_share']=float(np.mean(cached['lam_fundamental_modern'][mask]<1))
                b['K_Allee_dead_land_share']=float(np.mean(cached['allee_dead_mask'][mask]))
                dead=mask&cached['allee_dead_mask'].astype(bool)
                b['K_Allee_dead_C_breeding']=float(num[dead].sum()/denom[dead].sum()) if denom[dead].sum()>0 else None
                b['K_Allee_dead_breeder_weight']=float(denom[dead].sum()/denom.sum()) if denom.sum()>0 else None
            summary['regions'][reg][era]=b
    start=int(np.searchsorted(years,a.isolation_start))
    for reg in ['west','plains','broad_plains','east']:
        mask=jnp.zeros(x['land_mask'].shape).at[lr,lc].set(regions[reg])
        isolated=isolation_replay(af,x,states,setup,start,mask)
        closed=isolation_replay(af,x,states,setup,start,mask,no_movement=True)
        np.savez_compressed(a.out/f'isolation_{reg}.npz',years=years[start:],regional_no_immigration=isolated,cell_no_movement=closed)
        ref=np.asarray(states[1])[-1,lr,lc]+np.asarray(states[2])[-1,lr,lc]
        denom=float(ref[regions[reg]].sum())
        summary['isolation'][reg]={'start_year':int(years[start]),'end_year':int(years[-1]),
            'regional_no_immigration_final_over_fitted':float(isolated[-1].sum()/denom) if denom else None,
            'cell_no_movement_final_over_fitted':float(closed[-1].sum()/denom) if denom else None,
            'interpretation':'Regional boundary cuts successful incoming birds and allows losses through emigration; cell closure removes both immigration and emigration. Nonlinear departure and reproduction recomputed, rates follow fitted years.'}
    hashes={n:hashlib.sha256(Path(v).read_bytes()).hexdigest() for n,v in [('forward',af.__file__),('priors',ap.__file__),('checkpoint',a.run/'map_checkpoint.pkl'),('audit',__file__),('budget',Path(__file__).with_name('contribution_budget.py'))]}
    summary['sha256']=hashes
    if implementation_start!=[hashlib.sha256(p.read_bytes()).hexdigest() for p in implementation_paths]:
        raise RuntimeError('implementation changed while running; output not accepted')
    (a.out/'READY.json').write_text(json.dumps(clean(summary),indent=2,allow_nan=False)+'\n')
    print('CONTRIBUTION READY',step,checks,flush=True)


if __name__=='__main__':main()
