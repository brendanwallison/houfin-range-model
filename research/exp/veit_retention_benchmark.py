"""Controlled 1-D Veit–Lewis equations: gradual departure versus smooth overflow.

Not a full paper replication or refit. Finite-domain integrated kernels and a
small mortality sensitivity isolate the departure/crowding interaction.
"""
import json,os
from pathlib import Path
import numpy as np
from scipy.special import gammainc
from scipy.signal import fftconvolve
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('/mnt/c/Dev/houfin-range-model/research/experiments')
s=.67;c=1.5;delta=1.5;r=.15
b=(c/(1-s)-2)*delta
K=.5*(b+np.sqrt(b*b-4*r*delta))
low=.5*(b-np.sqrt(b*b-4*r*delta))
birth=lambda n:c*n*n/(r+2*n+n*n/delta)
growth=lambda n:s+np.where(n>0,c*n/(r+2*n+n*n/delta),0.)
def overflow(n,cap=K):
    u=n/(.05*cap);v=20.
    safe=np.where(np.abs(u)<1e-3,1.,u);z=1/(1+np.exp(v))
    value=(np.logaddexp(0.,u-v)-np.logaddexp(0.,-u-v))/safe
    series=2*z+(u*u/3)*z*(1-z)*(1-2*z)
    return np.where(np.abs(u)<1e-3,series,value)

dx=float(os.environ.get('VL_BENCHMARK_DX','5'));extent=float(os.environ.get('VL_BENCHMARK_EXTENT','6000'))
tag=os.environ.get('VL_BENCHMARK_TAG','');suffix=('_'+tag) if tag else ''
x=np.arange(-extent,extent+dx,dx)
def kernel(a,b):
    lo=np.maximum(np.abs(x)-dx/2,0.);hi=np.abs(x)+dx/2
    mass=.5*(gammainc(1/a,(hi/b)**a)-gammainc(1/a,(lo/b)**a))
    mass[len(x)//2]*=2
    return mass/mass.sum()
ka=kernel(.859,76.893);kj=kernel(.468,41.056)
profiles={};runs={}
for survival_multiplier in [1.,.95]:
    eq=brentq(lambda n:survival_multiplier*growth(n)-1.,np.sqrt(r*delta),10*K)
    for mode in ['ramp','overflow']:
        label=f'{mode}_mortality_{survival_multiplier}'
        # Cell-integrated initial patch: identical initial mass at every resolution.
        overlap=np.maximum(np.minimum(x+dx/2,10.)-np.maximum(x-dx/2,-10.),0.)
        n=.5*overlap/dx
        history=[];snapshots=[]
        for t in range(81):
            occupied=np.abs(x)[n>=.1]
            history.append(dict(year=t,radius_km=float(occupied.max()) if len(occupied) else None,core_density=float(n[len(x)//2]),line_integrated_population=float(n.sum()*dx)))
            snapshots.append(n.copy())
            p=np.minimum(n/K,1.) if mode=='ramp' else overflow(n)
            adult=survival_multiplier*s*n;young=survival_multiplier*birth(n)
            moved=fftconvolve(adult*p,ka,mode='same')+fftconvolve(young*p,kj,mode='same')
            n=np.maximum(adult*(1-p)+young*(1-p)+moved,0.)
        runs[label]=dict(history=history,closed_population_equilibrium=eq,equilibrium_over_original_K=eq/K,
            departure_at_closed_equilibrium=float(min(eq/K,1.) if mode=='ramp' else overflow(eq)))
        profiles[label]=np.stack(snapshots)
curve=np.linspace(.0001,1.5,600)
out=dict(paper_parameters=dict(s=s,c=c,delta=delta,r=r),stable_equilibrium=K,unstable_equilibrium=low,
    peak_growth=float(growth(np.sqrt(r*delta))),
    departure_table=[dict(fraction=q,ramp=min(q,1.),overflow=float(overflow(q*K))) for q in [.25,.5,.75,.9,1.]],runs=runs,
    scope='1-D controlled mechanism experiment; not a two-dimensional range reconstruction, numerical reproduction of the paper, or calibrated project fit',dx_km=dx,half_domain_km=extent)
(OUT/f'VEIT_RETENTION_BENCHMARK_20261102{suffix}.json').write_text(json.dumps(out,indent=2)+'\n')
np.savez_compressed(OUT/f'VEIT_RETENTION_BENCHMARK_20261102{suffix}.npz',x=x,**profiles)
fig,ax=plt.subplots(1,3,figsize=(14,4),layout='constrained')
ax[0].plot(curve,growth(curve*K),label='Original growth')
ax[0].plot(curve,.95*growth(curve*K),label='5% additional annual loss')
ax[0].axhline(1,color='grey',ls='--');ax[0].axvline(1,color='grey',ls=':')
ax[0].set(xlabel='Population / original equilibrium density',ylabel='Annual growth multiplier',title='Growth slows before the reference density');ax[0].legend(fontsize=8)
ax[1].plot(curve,np.minimum(curve,1),label='Veit-Lewis gradual departure')
ax[1].plot(curve,overflow(curve*K),label='Smooth overflow, width5%')
ax[1].set(xlabel='Population / original equilibrium density',ylabel='Fraction entering dispersal',title='Their birds leave before the cell is full',xlim=(0,1.1));ax[1].legend(fontsize=8)
labels={'ramp_mortality_1.0':'Gradual departure','overflow_mortality_1.0':'Smooth overflow','ramp_mortality_0.95':'Gradual + extra loss','overflow_mortality_0.95':'Overflow + extra loss'}
for label,q in runs.items():
    ax[2].plot([a['year'] for a in q['history']],[a['radius_km'] if a['radius_km'] is not None else np.nan for a in q['history']],label=labels[label])
ax[2].set(xlabel='Years',ylabel='Radius at density 0.1, km',title='Controlled spread comparison',xlim=(0,50),ylim=(0,2500));ax[2].legend(fontsize=8)
fig.savefig(OUT/f'VEIT_RETENTION_BENCHMARK_20261102{suffix}.png',dpi=160);plt.close(fig)
print(json.dumps(dict(K=K,low=low,peak_growth=out['peak_growth'],departure=out['departure_table'],
    results={k:{'year50':v['history'][50],'equilibrium_over_K':v['equilibrium_over_original_K'],'departure_at_equilibrium':v['departure_at_closed_equilibrium']} for k,v in runs.items()}),indent=2))
