"""Strength/extent sanity check using actual old inputs, not nominal row counts."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

REPO=Path('/mnt/c/Dev/houfin-range-model');D=REPO/'.research_runtime/historical_range_inputs_v2';E=REPO/'research/experiments'
r=json.loads((D/'PSEUDOZERO_COMPARISON.json').read_text())
calibration=json.loads((D/'ORIGINAL_PSEUDOZERO_CALIBRATION.json').read_text())
power=calibration['obs_conc_kappa']
r['original_density_dispersion_power']=power
for q in r['matched_density_strength']:
    n=q['total_population_route_scale'];factor=1+r['reference_fledgling_weight']*q['young_fraction']
    mu=max(n*factor,1e-6);c=r['reference_concentration']*max(mu,1e-3)**power
    derivative=1/(1+mu/c)
    if mu>1e-3:derivative+=(power*c/mu)*(np.log1p(mu/c)-mu/(c+mu))
    q['original_total_zero_loss']=r['old_effective_weight']*c*np.log1p(mu/c)
    q['original_gradient_wrt_total']=r['old_effective_weight']*factor*derivative if n*factor>1e-6 else 0.
with np.load(D/'comparison_maps.npz') as z:m={k:z[k] for k in z.files}
lookup={(int(a),int(b)):i for i,(a,b) in enumerate(zip(m['land_rows'],m['land_cols']))}
with np.load('/work/07980/bwa386/ls6/houfin/processed/grid54/bbs/bbs_data_obsroute_for_python.npz') as z:
    mask=(z['obs_route_idx']<0)&(z['obs_year']<1940)
    indices=np.array([lookup[(int(a),int(b))] for a,b in zip(z['obs_rows'][mask],z['obs_cols'][mask])])
oldfull=np.bincount(indices,minlength=len(lookup))>0;new=m['new_exterior'].astype(bool)
r['full_cached_old_geometry']=dict(cells=int(oldfull.sum()),new_cells=int(new.sum()),intersection=int((oldfull&new).sum()),
    old_only=int((oldfull&~new).sum()),new_only=int((~oldfull&new).sum()),
    distinction='Full cached coarse BBS pseudozero support before 1814-site model-input subsampling and observation masks')
lon,lat=m['lon'],m['lat']
for reg,mask in {'east':lon>=-80,'great_plains':(lon>=-100)&(lon<-95),'west':lon<-100,'other':(lon>=-95)&(lon<-80)}.items():
    r['extent'][reg].update(full_old_cells=int((oldfull&mask).sum()),geometric_new_only=int((~oldfull&new&mask).sum()),geometric_old_only=int((oldfull&~new&mask).sum()))
# Both families at the same point, plus original strength's fitted-concentration dependence.
grid=np.logspace(-5,1,200);k=r['reference_concentration'];phi=r['reference_fledgling_weight'];mass=r['old_effective_weight']
loss=lambda n,k:k*np.log1p(n/k)
oldconc=lambda mu:k*np.maximum(mu,1e-3)**power
def oldgrad(mu):
    c=oldconc(mu)
    return 1/(1+mu/c)+np.where(mu>1e-3,power*c/mu*(np.log1p(mu/c)-mu/(c+mu)),0.)
fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
category=np.where(oldfull&new,1,np.where(oldfull,2,np.where(new,3,0)))
colors=['#eeeeee','#777777','#dc6b43','#477ec2']
ax[0,0].scatter(lon,lat,c=category,cmap=ListedColormap(colors),vmin=0,vmax=3,s=5,marker='s',linewidths=0)
ax[0,0].set(title='Full geographic support: grey shared; blue added; red removed',xlabel='Longitude',ylabel='Latitude',xlim=(-126,-65),ylim=(24,54))
regions=['east','great_plains','west','other'];left=[r['extent'][x]['old_effective_weight'] for x in regions];right=[r['extent'][x]['new_effective_weight'] for x in regions]
x=np.arange(4);ax[0,1].bar(x-.18,left,.36,label='Original active');ax[0,1].bar(x+.18,right,.36,label='Derived')
ax[0,1].set(xticks=x,xticklabels=['East','Plains','West','Other'],ylabel='Total pre-1940 effective weight',title='Same total weight; regional allocation changes');ax[0,1].legend()
ax[1,0].loglog(grid,mass*loss(grid*(1+phi*.5),oldconc(grid*(1+phi*.5))),label='Original; young fraction .5')
ax[1,0].loglog(grid,mass*loss(grid,k),label='Derived; annual total')
ax[1,0].set(xlabel='Constant total population, route-count scale',ylabel='Total zero penalty',title='Matched density; original density-dependent dispersion');ax[1,0].legend()
ax[1,1].semilogx(grid,mass*(1+phi*.5)*oldgrad(grid*(1+phi*.5)),label='Original')
ax[1,1].semilogx(grid,mass/(1+grid/k),label='Derived')
ax[1,1].set(xlabel='Constant total population, route-count scale',ylabel='Derivative of total zero penalty',title='Matched-density pressure on population');ax[1,1].legend()
fig.savefig(E/'HISTORICAL_PSEUDOZERO_COMPARISON_20261102.png',dpi=180);plt.close(fig)
smoke=REPO/'.research_runtime/historical_range_smoke_v2/historical.json'
if smoke.exists():r['matched_reference_trajectory']=json.loads(smoke.read_text()).get('matched_reference_trajectory')
if smoke.with_suffix('.npz').exists():
    with np.load(smoke.with_suffix('.npz')) as z:population=z['population']
    with np.load(D/'historical.npz') as z:
        cells=z['hr_zero_cell'];t=z['hr_zero_t'];weight=z['hr_zero_weight']
    ll=weight*loss(population[t,cells],k)
    active=m['old_weight']>0
    r['matched_reference_trajectory']['derived_absence_by_support']={
      'original_active_cells':float(ll[active[cells]].sum()),
      'old_full_footprint_but_not_active':float(ll[(oldfull&~active)[cells]].sum()),
      'genuinely_new_geographic_cells':float(ll[(~oldfull)[cells]].sum())}
(E/'HISTORICAL_PSEUDOZERO_COMPARISON_20261102.json').write_text(json.dumps(r,indent=2)+'\n')
lines=['# Original versus derived historical pseudozeros','',
    f"The original model input contains {r['old_pseudozero_rows']:,} rows, but observation masking leaves effective weight {mass:,.0f} over {r['old_cells']:,} cells. The replacement has exactly the same pre-1940 total weight, spread across {r['derived_cells']:,} cells. These are explicit range constraints, not independent surveyed absences.",'',
    f"The full cached old footprint, before model-input subsampling/masking, contains {oldfull.sum():,} cells. The derived footprint shares {(oldfull&new).sum():,}, adds {(~oldfull&new).sum():,}, and removes {(oldfull&~new).sum():,}. Thus the difference between 1,595 active original sites and 2,181 derived cells is partly subsampling/masking, partly true geometry. Full coarse-cell areas are nominal 2,916 km², not coastal land area.",'',
    '| Region | Original full footprint | Original active sites | Derived sites | Original weight | Derived weight |','|---|---:|---:|---:|---:|---:|']
for name in regions:
    q=r['extent'][name];lines.append(f"| {name.replace('_',' ').title()} | {q['full_old_cells']} | {q['old_constrained_cells']} | {q['new_constrained_cells']} | {q['old_effective_weight']:.1f} | {q['new_effective_weight']:.1f} |")
lines+=['','## Strength at matched population','',
 f"The derived NB2 calibration is fixed at the accepted no-CBC seed41 concentration {k:.6f}. The original pseudozeros used learned density-dependent dispersion c(mu)=k*max(mu,.001)^p, with reference p={power:.6f}. Original zero loss is c(mu)*log(1+mu/c(mu)); the derived loss uses constant k. The corrected curves include both this dispersion difference and the census/readout difference. No new CBC calibration scale or effort coefficient is fitted. This comparison is conditional on that calibration, not proof that all original fits had the same strength.",'',
 'The old observation census is previous total plus fitted phi times current young; the derived constraint uses annual total. For a time-constant population with young fraction .5, the old expected count is 1.13295 times total density. The original density-dependent dispersion weakens sparse-population zero penalties relative to a constant-dispersion comparison. Consequently the derived penalty can be stronger at low matched densities even with identical weight; the earlier fixed-dispersion curves omitted this and are superseded. The table includes young fractions 0, .5 and .8 in the machine-readable output.','',
 '| Total route-scale population | Original loss (young .5) | Derived loss | Derived/original |','|---|---:|---:|---:|']
for row in r['matched_density_strength']:
    if row['young_fraction']==.5 and row['total_population_route_scale']>0:
        lines.append(f"| {row['total_population_route_scale']:g} | {row['original_total_zero_loss']:.3f} | {row['derived_total_zero_loss']:.3f} | {row['derived_total_zero_loss']/row['original_total_zero_loss']:.3f} |")
if r.get('matched_reference_trajectory'):
    q=r['matched_reference_trajectory'];lines+=['','## Actual reference trajectory','',
      f"At the identical accepted demographic point, original pseudozero loss is {q['original_pseudozero_loss']:.3f}; derived pre-1940 absence loss is {q['derived_range_absence_loss']:.3f}; added presence loss is {q['derived_range_presence_loss']:.3f}. This includes geometry/readout differences at a common trajectory, not effects of refitting. Old pseudozero weight remaining in the replacement fit is exactly {q['old_pseudozero_weight_remaining']:.0f}."]
    if q.get('derived_absence_by_support'):
        c=q['derived_absence_by_support'];lines += ['',f"Derived absence loss decomposes into {c['original_active_cells']:.3f} at originally active sites, {c['old_full_footprint_but_not_active']:.3f} inside the old full footprint but not actively weighted, and {c['genuinely_new_geographic_cells']:.3f} in genuinely added geographic cells. Thus the larger reference loss must not be described as an elevenfold uniform increase in the zero penalty."]
lines+=['','![Extent, regional allocation, loss and gradient](HISTORICAL_PSEUDOZERO_COMPARISON_20261102.png)','',
 'Coverage weighting is 1+log(1+distinct CBC circle/decade units within 216 km), normalized to the fixed weight budget. Regions without CBC coverage retain baseline historical-prior weight; they are not called surveyed absences. These choices and the fixed dispersion need sensitivity checks if they determine the scientific conclusion. A wider extent with lower per-cell weights may constrain new locations while reducing pressure at previously weighted sites.']
(E/'HISTORICAL_PSEUDOZERO_COMPARISON_20261102.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(r['full_cached_old_geometry'],indent=2))
