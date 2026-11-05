"""Frozen-environment origin attribution and BIDE; no demographic refitting.

Runge breeding contribution uses the previous summer's adults as breeders and
their actual local offspring as recruits, followed to next summer's breeders.
The environment is held at its fitted state during attribution, not differentiated.
"""
import jax
import jax.numpy as jnp
import numpy as np


def attributed_step(movement, a, j, sa, sj, offspring_per_survivor,
                    movement_diagonal,arriver_survival=1.):
    """Attribute a positive linear movement operator with independent stage pools.

    movement(a,j) returns pre-local-survival adult, juvenile stayer, and already
    journey-surviving juvenile arrivals. Its departure/environment MUST be frozen.
    Diagonals include dispersers that return to their origin cell.
    """
    one=jnp.ones_like(a)
    # Evaluate transposes at positive inputs: the primitive clips negative tags;
    # differentiating at zero would introduce the max(0,x) subgradient.
    def transpose(weights):
        _,pull=jax.vjp(movement,one,one)
        return pull(weights)
    ap,js,ja=movement(a,j)
    am,jm=transpose((one,one,one))
    ac,jc=transpose((sa,sj,one*arriver_survival))
    total_weight=1.+offspring_per_survivor
    at,jt=transpose((sa*total_weight,sj*total_weight,total_weight*arriver_survival))
    da,ds,dj=movement_diagonal
    self_move=a*da+j*(ds+dj)
    self_breed=a*da*sa+j*(ds*sj+dj*arriver_survival)
    origin_move=a*am+j*jm
    breed_origin=a*ac+j*jc
    breed_destination=ap*sa+js*sj+ja*arriver_survival
    return dict(adult_post=ap,juvenile_stay=js,juvenile_arrive=ja,
        movement_origin=origin_move,movement_destination=ap+js+ja,
        retained_movement=self_move,immigration=ap+js+ja-self_move,
        successful_emigration=origin_move-self_move,
        retained_breeders=self_breed,breeder_contribution=breed_origin,
        successful_breeder_export=breed_origin-self_breed,
        immigrant_breeders=breed_destination-self_breed,
        next_breeders=breed_destination,
        origin_adult_survival=ac,origin_juvenile_recruitment=jc,
        total_descendants_origin=a*at+j*jt,
        total_descendants_retained=self_breed*total_weight,
        local_deaths_destination=ap*(1-sa)+js*(1-sj)+ja*(1-arriver_survival))


def safe_ratio(n,d):
    return jnp.where(d>1e-20,n/jnp.where(d>1e-20,d,1.),jnp.nan)


def audit_forward(af,args,states):
    """Replay supported plain population model from captured forward arguments.

    Unsupported mechanisms raise; never silently drop an active population term.
    """
    x=args
    for key in ['strata_u','occupancy_b','state_warp','state_anchor',
                'attraction']:
        if x.get(key) is not None:raise NotImplementedError(key)
    if x.get('process_noise_low_density') is not None or x.get('process_noise_groups') is not None:
        raise NotImplementedError('state-dependent or conserved process multiplier')
    if x.get('settle_block',0):raise NotImplementedError('settlement')
    if af.FFT_PAD_VALUE!=0 or af.FFT_FLOOR_ULPS is not None:
        raise NotImplementedError('affine/nonlinear FFT padding or floor')
    lr,lc=x['land_rows'],x['land_cols'];land=x['land_mask'];shape=land.shape
    zero=jnp.zeros(shape);ones=jnp.ones(shape)
    def grid(v):return zero.at[lr,lc].set(v)
    dep=None if x.get('departure_scale') is None else ones.at[lr,lc].set(x['departure_scale'])
    ka=jnp.fft.ifft2(x['adult_fft_kernel'].astype(jnp.complex128)).real[0,0]
    kj=jnp.fft.ifft2(x['juvenile_fft_kernel_stack'].astype(jnp.complex128),axes=(-2,-1)).real[:,0,0]
    release=zero.at[x['inv_locations'][:,0],x['inv_locations'][:,1]].set(1.)
    na,nj=states[1],states[2]
    prev_a=jnp.concatenate((x['initpop_latent'][None]*.5,na[:-1]))
    prev_j=jnp.concatenate((x['initpop_latent'][None]*.5,nj[:-1]))

    def disease(t,sa,sj,ap,js,ja):
        if x.get('disease_m_amp') is None:return sa,sj,jnp.ones_like(sa)
        amp=grid(x['disease_m_amp'][t]);n=ap+js+ja
        hn=jnp.power(jnp.maximum(n,0.)+1e-12,x.get('disease_hill_n',1.))
        remaining=1.-amp*hn/(hn+x.get('disease_k_half',1.)**x.get('disease_hill_n',1.))
        return sa*remaining,sj*remaining,remaining

    def setup(t,a,j,freeze=True):
        sa,sj,f,k,c=(grid(x[n][t]) for n in ['Sa_flat','Sj_flat','Fmax_flat','K_flat','c_flat'])
        q=jnp.zeros((x['Q_flat'].shape[-1],)+shape).at[:,lr,lc].set(x['Q_flat'][t].T)
        weights=x.get('juvenile_kernel_weights')
        if weights is not None and weights.ndim>1:weights=weights[t]
        origin=x.get('directional_weights')
        if origin is not None:origin=jnp.ones_like(q).at[:,lr,lc].set(origin[t].T)
        kw=dict(dispersal_logit_intercept=x['dispersal_logit_intercept']+x['dispersal_random'][t],
            dispersal_logit_slope=x['dispersal_logit_slope'],target_fraction=x['target_fraction'],
            adult_edge_correction=x['adult_edge_correction'],juvenile_edge_correction_stack=x['juvenile_edge_correction_stack'],
            adult_fft_kernel=x['adult_fft_kernel'],juvenile_fft_kernel_stack=x['juvenile_fft_kernel_stack'],Q_grid=q,
            eps=1e-6,adult_logit_offset=x.get('dispersal_adult_offset',0.),crowding_beta=x.get('crowding_beta'),
            crowding_kappa=x.get('crowding_kappa',5.),kernel_weights=weights,
            veit_departure=x.get('veit_departure',False),departure_density=a+j,departure_scale=dep,origin_weights=origin)
        if getattr(af,'OVERFLOW_ENABLED',False):
            # Freeze BOTH stage-specific probabilities at ambient fitted pools.
            # Attribution follows birds, without differentiating density feedback.
            kw['departure_probabilities_override']=af.overflow_probabilities(a,j,jnp.maximum(k,kw['eps']))
        def move(aa,jj):
            out=af.dispersal_step_age_structured(aa,jj,k,**kw)
            return tuple(v*land for v in out)
        # Exact departure probabilities from the same primitive's stayer branch.
        # Adult no-departure is separated from its FFT arrivals by return flag.
        test=af.dispersal_step_age_structured(ones,ones,k,return_adult_arrivals=True,**kw)
        pa=1.-(test[0]-test[3]);pj=1.-test[1]
        da=(1-pa)+pa*ka/(x['adult_edge_correction']+1e-6)
        ds=1-pj
        w=None
        # juvenile_band_weights is handled upstream in current model; kernel_weights
        # passed here are already actual normalized directional/radial weights.
        if weights is not None:w=weights
        factor=jnp.ones_like(q) if origin is None else origin
        if w is not None:factor=factor*jnp.asarray(w).reshape((-1,1,1))
        dj=pj*jnp.sum(factor*q*kj[:,None,None]/(x['juvenile_edge_correction_stack']+1e-6),axis=0)
        if getattr(af,'LOCAL_SETTLEMENT',False):
            from src.model.local_arrival_settlement import composite_diagonal
            frac=af.LOCAL_SETTLEMENT_FRACTION
            ad=composite_diagonal(x['adult_fft_kernel'],ones,k,frac)
            jd=composite_diagonal(x['juvenile_fft_kernel_stack'],q,k,frac)
            da=(1-pa)+pa*ad/(x['adult_edge_correction']+1e-6)
            dj=pj*jnp.sum(factor*jd/(x['juvenile_edge_correction_stack']+1e-6),axis=0)
        diag=(da*land,ds*land,dj*land)
        return sa,sj,f,k,c,move,diag,kw

    def step(_,t):
        pulse_k=t-x['inv_timestep'];valid=(pulse_k>=0)&(pulse_k<x['inv_pop'].shape[1])
        val=jnp.where(valid,x['inv_pop'][:,jnp.clip(pulse_k,0,x['inv_pop'].shape[1]-1)],0.)
        pulse=zero.at[x['inv_locations'][:,0],x['inv_locations'][:,1]].add(val)
        a,j=prev_a[t]+pulse*.5,prev_j[t]+pulse*.5
        sa,sj,f,k,c,move,diag,kw=setup(t,a,j)
        potential=sa+sj*f
        sa_local,sj_local,_=disease(t,sa,sj,a,j,zero)
        localout=af.reproduction_age_structured(a,j,zero,sa_local,sj_local,f,k,c,x['allee_gamma'],
            allee_off=jnp.where((pulse_k>=0)&(pulse_k<x.get('allee_exempt_years',0)),release,0.),
            allee_relative=x.get('allee_relative',False),yearling_breeding=x.get('yearling_breeding',False),
            veit=x.get('veit_reproduction',False))
        ap,js,ja=move(a,j)
        sa,sj,dm=disease(t,sa,sj,ap,js,ja)
        off=jnp.where((pulse_k>=0)&(pulse_k<x.get('allee_exempt_years',0)),release,0.)
        new=af.reproduction_age_structured(ap,js,ja*dm,sa,sj,f,k,c,x['allee_gamma'],
             allee_off=off,allee_relative=x.get('allee_relative',False),
             yearling_breeding=x.get('yearling_breeding',False),veit=x.get('veit_reproduction',False))
        an0,jn0=(jnp.maximum(v*land,0.) for v in new)
        adult_clip_gain=an0-new[0]*land
        mult=ones if x.get('process_noise') is None else jnp.exp(grid(x['process_noise'][t]))
        an,jn=an0*mult,jn0*mult
        process_adjustment=(an+jn)-(an0+jn0)
        unparented_births=jnp.where(an<=0,jn,0.)
        f_actual=jnp.where(an>0,jn/jnp.where(an>0,an,1.),0.)
        biological=attributed_step(move,a,j,sa,sj,f_actual,diag,dm)
        b=attributed_step(move,a,j,sa*mult,sj*mult,f_actual,diag,dm*mult)
        # Innovations are a separate discrepancy, never births or negative deaths.
        b['local_deaths_destination']=biological['local_deaths_destination']
        b['biological_breeder_contribution']=biological['breeder_contribution']
        b['biological_retained_breeders']=biological['retained_breeders']
        b['biological_successful_breeder_export']=biological['successful_breeder_export']
        b['biological_total_descendants_origin']=biological['total_descendants_origin']
        b['origin_adult_with_process']=b['origin_adult_survival']
        b['origin_juvenile_with_process']=b['origin_juvenile_recruitment']
        b['origin_adult_survival']=biological['origin_adult_survival']
        b['origin_juvenile_recruitment']=biological['origin_juvenile_recruitment']
        # Repeat the movement budget with journey survival disabled. Differences
        # distinguish Q mortality from edge/numerical/off-domain mass losses.
        kw0=dict(kw,Q_grid=jnp.ones_like(kw['Q_grid']))
        def move0(aa,jj):return tuple(v*land for v in af.dispersal_step_age_structured(aa,jj,k,**kw0))
        _,pull=jax.vjp(move0,ones,ones)
        ma0,mj0=pull((ones,ones,ones))
        preq=a*ma0+j*mj0
        journey=preq-b['movement_origin']
        numerical=a+j-preq
        # Births and ordinary mortality use destination accounting; successful
        # transport and journey/numerical loss use origin accounting.
        rhs=jn0-b['local_deaths_destination']+b['immigration']-b['successful_emigration']-journey-numerical+pulse+adult_clip_gain+process_adjustment
        observed_delta=an+jn-prev_a[t]-prev_j[t]
        beta=safe_ratio(j,a)
        b.update(input_adult=a,input_juvenile=j,pulse=pulse,births=jn0,
            potential_lambda=potential,realized_offspring_per_breeder=f_actual,
            potential_lambda_with_process=potential*mult,process_multiplier=mult,
            process_adjustment=process_adjustment,
            local_adult_survival=sa,local_juvenile_stayer_survival=sj,
            next_adult=an,next_juvenile=jn,journey_mortality_origin=journey,
            numerical_edge_loss_origin=numerical,
            numerical_adult_clip_gain=adult_clip_gain*mult,numerical_unparented_births=unparented_births,
            BIDE_error=observed_delta-rhs,
            replay_adult_error=an-na[t],replay_juvenile_error=jn-nj[t],
            C_breeding=safe_ratio(b['breeder_contribution'],a),
            R_breeding=safe_ratio(b['retained_breeders'],a),
            export_breeding=safe_ratio(b['successful_breeder_export'],a),
            local_breeder_replacement=sa_local+beta*sj_local,
            apparent_breeder_growth=safe_ratio(an,a),
            apparent_total_growth=safe_ratio(an+jn,a+j),
            C_state=safe_ratio(b['total_descendants_origin'],a+j),
            local_total_replacement=safe_ratio(localout[0]+localout[1],a+j))
        return None,{n:v[lr,lc] for n,v in b.items()}
    _,result=jax.jit(lambda:jax.lax.scan(step,None,jnp.arange(x['time'])))()
    result=jax.tree.map(np.asarray,jax.block_until_ready(result))
    return result,(setup,disease)


def isolation_replay(af,x,states,setup,start,mask,no_movement=False):
    """Remove external immigration at region boundary; retain ordinary emigration.

    Starts at fitted preceding year's state. Rates vary as fitted, departure and
    reproduction respond to the intervened population. Not a marginal derivative.
    """
    setup,disease=setup
    lr,lc=x['land_rows'],x['land_cols'];zero=jnp.zeros_like(x['land_mask'],dtype=float)
    def step(carry,t):
        a,j=carry
        sa,sj,f,k,c,move,_,_=setup(t,a,j)
        ap,js,ja=(a,j,zero) if no_movement else move(a,j)
        sa,sj,dm=disease(t,sa,sj,ap*mask,js*mask,ja*mask)
        # Outside-to-inside inputs are zero because the outside state is zero.
        new=af.reproduction_age_structured(ap*mask,js*mask,ja*mask*dm,sa,sj,f,k,c,x['allee_gamma'],
            allee_relative=x.get('allee_relative',False),yearling_breeding=x.get('yearling_breeding',False),
            veit=x.get('veit_reproduction',False))
        mult=1. if x.get('process_noise') is None else jnp.exp(zero.at[lr,lc].set(x['process_noise'][t]))
        new=tuple(jnp.maximum(v*mask,0.)*mult for v in new)
        return new,jnp.stack([v[lr,lc] for v in new])
    initial=(states[1][start-1]*mask,states[2][start-1]*mask)
    _,out=jax.jit(lambda:jax.lax.scan(step,initial,jnp.arange(start,x['time'])))()
    return np.asarray(jax.block_until_ready(out))
