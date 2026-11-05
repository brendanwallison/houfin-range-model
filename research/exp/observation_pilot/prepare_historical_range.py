"""Build effort-free range inputs and compare against the actual inherited pseudozeros."""
import hashlib,json,pickle
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import pandas as pd
import rasterio
from rasterio.warp import transform
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint,Point
from shapely.ops import unary_union
from src.config_utils import load_age_model_config
from src.model.data_loading import load_data
from src.model.checkpoints import load_map_params,auto_delta_params_to_latents

REPO=Path('/mnt/c/Dev/houfin-range-model');OUT=REPO/'.research_runtime/historical_range_inputs_v2';OUT.mkdir(exist_ok=True)
cfg=load_age_model_config();m=pickle.load(open(Path(cfg['input_dir'])/'metadata.pkl','rb'))
d=load_data(cfg['input_dir'],precision='float64',verbose=False)
ow=np.asarray(d.get('obs_weight',np.ones_like(m['observed_results'])),float);del d
p,_=load_map_params('/work/07980/bwa386/ls6/houfin/processed/model_results/refined_float64__age_map_float32_grid54_step38_cbc1951_noncbc_free_s41_quick90')
lat=auto_delta_params_to_latents(p);k=float(lat['concentration']);phi=float(lat['obs_fledgling_weight']);power=float(lat['obs_conc_kappa'])
(OUT/'ORIGINAL_PSEUDOZERO_CALIBRATION.json').write_text(json.dumps(dict(concentration=k,obs_conc_kappa=power,obs_fledgling_weight=phi),indent=2)+'\n')
rr,cc=np.asarray(m['land_rows']),np.asarray(m['land_cols']);years=np.asarray(m['years'])
ot=np.asarray(m['obs_time_indices']);oy=years[ot]
old=(np.asarray(m['obs_route_idx'])<0)&(oy<1940)&(np.asarray(m['observed_results'])==0)
if int(old.sum())!=68932:raise ValueError('unexpected inherited pseudozero batch')
oldmass=float(ow[old].sum());oldyears=np.unique(oy[old]);oldannual=oldmass/len(oldyears)
with rasterio.open('/work/07980/bwa386/ls6/houfin/processed/grid54/ref_grid_54km.tif') as src:
    xx,yy=rasterio.transform.xy(src.transform,rr,cc);xy=np.column_stack([xx,yy]);crs=src.crs;gridshape=src.shape
    lon,latgrid=map(np.asarray,transform(src.crs,'EPSG:4326',xx,yy))
forward=Transformer.from_crs('EPSG:4326',crs,always_xy=True)
tree=cKDTree(xy)
f=pd.read_parquet(REPO/'.research_runtime/cbc_early_inputs/rows.parquet')
circles=pd.read_parquet(REPO/'.research_runtime/cbc_early_inputs/circles.parquet')
with np.load(REPO/'.research_runtime/cbc_early_inputs/cbc_early.npz') as z:
    sourceidx=z['cbc_source_idx'];supportw=np.maximum(z['cbc_w_local'],z['cbc_w_seasonal'])
pos=f[f['count']>0].copy()
native=pos[pos.summer_year<1940]
nx,ny=forward.transform(native.longitude.to_numpy(),native.latitude.to_numpy())
nativehull=MultiPoint(np.column_stack([nx,ny])).convex_hull
native_envelope=nativehull.buffer(700000.)
pre_out=np.array([not native_envelope.covers(Point(x,y)) for x,y in xy])
native_seed=np.asarray(m['initpop_latent'])
if native_seed.ndim==2:native_seed=native_seed[rr,cc]
native_conflict=pre_out&(native_seed>0)
if native_conflict.any():
    # Preserve the project's independent native-range support, rather than declaring
    # unsampled known-native cells absent because the early CBC network was sparse.
    seed_hull=MultiPoint(xy[native_seed>0]).convex_hull.buffer(700000.)
    native_envelope=unary_union([native_envelope,seed_hull])
    pre_out=np.array([not native_envelope.covers(Point(x,y)) for x,y in xy])
lookup={(int(r),int(c)):i for i,(r,c) in enumerate(zip(rr,cc))}
with np.load('/work/07980/bwa386/ls6/houfin/processed/grid54/bbs/bbs_data_obsroute_for_python.npz') as _b:
    _o=(_b['obs_route_idx']<0)&(_b['obs_year']<1940)
    _ix=[lookup[(int(r),int(c))] for r,c in zip(_b['obs_rows'][_o],_b['obs_cols'][_o])]
old_full=np.bincount(_ix,minlength=len(rr))>0
# The CBC extract is bounded. Do not synthesize NEW absences beyond that domain.
# Preserve inherited exterior support there as historical prior, explicitly.
survey_domain=(lon>=-125)&(lon<=-66)&(latgrid>=24)&(latgrid<=53)
unguarded=pre_out.copy()
pre_out=np.where(survey_domain,pre_out,old_full)
ci_xy=np.column_stack(forward.transform(circles.longitude.to_numpy(),circles.latitude.to_numpy()))

def coverage(lo,hi):
    # Actual survey locations/decades, not imputed hours or independent repeated absences.
    sub=f[f.summer_year.between(lo,hi)].copy();sub['decade']=sub.summer_year//10
    units=sub[['circle_index','decade']].drop_duplicates()
    counts=np.zeros(len(rr))
    for i in units.circle_index:
        counts[tree.query_ball_point(ci_xy[int(i)],216000.)]+=1
    return counts,1+np.log1p(counts)

pre_coverage,pre_strength=coverage(1902,1939)
preweights=np.where(pre_out,pre_strength,0.)
preweights*=oldannual/preweights.sum()

def positive_rows(lo,hi,period):
    q=pos[pos.summer_year.between(lo,hi)].copy()
    # One presence per coarse cell/period. Keep the first actual date in that unit.
    px,py=forward.transform(q.longitude.to_numpy(),q.latitude.to_numpy());_,cell=tree.query(np.column_stack([px,py]))
    q['cell']=cell;q['period']=(q.summer_year-lo)//period
    return q.sort_values('date').drop_duplicates(['cell','period'])

prepositive=positive_rows(1902,1939,10)
laterpositive=positive_rows(1951,1965,1)
def arrays(invasion):
    zt=[];zc=[];zw=[];periods=[]
    cells=np.flatnonzero(pre_out)
    for year in oldyears:
        zt.extend([int(np.flatnonzero(years==year)[0])]*len(cells));zc.extend(cells);zw.extend(preweights[cells])
    if invasion:
        # Annual, retrospective occurrence envelopes; no later-year lookahead.
        # 330km is a conservative assumed spatial allowance, not a hard kernel maximum.
        for year in range(1951,1966):
            eastern=pos[(pos.summer_year.between(1951,year))&(pos.longitude>-85)&(pos.latitude>=33)]
            if len(eastern)==0:raise ValueError('no introduced core detections')
            ex,ey=forward.transform(eastern.longitude.to_numpy(),eastern.latitude.to_numpy())
            introduced=MultiPoint(np.column_stack([ex,ey])).convex_hull.buffer(330000.)
            extent=unary_union([native_envelope,introduced])
            exterior=np.array([not extent.covers(Point(x,y)) for x,y in xy])
            exterior=np.where(survey_domain,exterior,old_full)
            # Areas allowed by native historical range are never newly forced absent.
            count,strength=coverage(max(1951,year-2),year)
            weights=np.where(exterior,strength,0.);weights*=oldannual/weights.sum()
            cells=np.flatnonzero(exterior)
            zt.extend([int(np.flatnonzero(years==year)[0])]*len(cells));zc.extend(cells);zw.extend(weights[cells])
            periods.append(dict(year=year,cells=len(cells),eastern_positive_records=len(eastern),effective_weight=float(weights.sum()),survey_supported_cells=int((exterior&(count>0)).sum())))
    pr=pd.concat([prepositive,laterpositive],ignore_index=True) if invasion else prepositive
    ci=pr.circle_index.to_numpy(int)
    data=dict(hr_zero_t=np.asarray(zt,np.int32),hr_zero_cell=np.asarray(zc,np.int32),hr_zero_weight=np.asarray(zw),
      hr_positive_t=pd.Index(years).get_indexer(pr.summer_year).astype(np.int32),hr_positive_idx=sourceidx[ci].astype(np.int32),
      hr_positive_w=supportw[ci],hr_positive_weight=np.ones(len(pr)))
    data['hr_old_zero_mask']=old.astype(np.uint8)
    data['hr_old_zero_weight']=ow*old
    # No positive source footprint can contradict its corresponding exterior mask.
    # Seasonal source support is a lower-bound availability operator, not inferred migration.
    return data,periods

for mode in ['historical','invasion']:
    data,periods=arrays(mode=='invasion');path=OUT/f'{mode}.npz';np.savez_compressed(path,**data)
    receipt=dict(mode=mode,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),fixed_concentration=k,
      fixed_reference_fledgling_weight=phi,zero_rows=len(data['hr_zero_t']),zero_effective_weight=float(data['hr_zero_weight'].sum()),
      positive_constraints=len(data['hr_positive_t']),native_buffer_km=700,introduced_buffer_km=330,
      period_records=periods,old_pseudozeros_removed=int(old.sum()),old_pseudozero_effective_weight=oldmass,
      old_annual_effective_weight=oldannual,coverage='1+log(1+distinct circle/decade units within216km), normalized to fixed annual mass; coverage is not detection probability',
      source_support='union/max of local-circle and 300km SW-source weights; no migrant fraction or fitted detection scale',
      initially_conflicting_native_seed_cells=int(native_conflict.sum()),
      domain_guard='No new CBC-derived absence outside raw extraction box125W-66W,24N-53N; inherited old footprint retained there',
      unguarded_new_outside_domain_cells=int((unguarded&~old_full&~survey_domain).sum()),
      dates='actual December y / January y+1 -> summer y; first adjacent eastern detection December28 1952',
      scope='historical range constraint with fixed NB2 censoring calibration; not a generative CBC effort model')
    (OUT/f'{mode}_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')

lookup={(int(r),int(c)):i for i,(r,c) in enumerate(zip(rr,cc))}
oi=np.array([lookup[(int(r),int(c))] for r,c in zip(np.asarray(m['obs_rows'])[old],np.asarray(m['obs_cols'])[old])])
oldsite=np.bincount(oi,weights=ow[old],minlength=len(rr));oldmask=oldsite>0
regions={'east':lon>=-80,'great_plains':(lon>=-100)&(lon<-95),'west':lon<-100,'other':(lon>=-95)&(lon<-80)}
extent={}
for region,mask in regions.items():
    extent[region]=dict(old_constrained_cells=int((oldmask&mask).sum()),new_constrained_cells=int((pre_out&mask).sum()),
      intersection=int((oldmask&pre_out&mask).sum()),old_only=int((oldmask&~pre_out&mask).sum()),new_only=int((~oldmask&pre_out&mask).sum()),
      old_effective_weight=float(oldsite[mask].sum()),new_effective_weight=float(preweights[mask].sum()*len(oldyears)),
      new_supported_by_CBC=int((pre_out&mask&(pre_coverage>0)).sum()))
oldreadout=lambda n,young_fraction:n*(1+phi*young_fraction)
zero=lambda n:k*np.log1p(n/k)
grad=lambda n:1/(1+n/k)
strength=[]
for n in [0.,1e-4,.001,.01,.1,1.,10.]:
    for young in [0.,.5,.8]:
        scale=1+phi*young;mu=oldreadout(n,young);mu_safe=max(mu,1e-6)
        c=k*max(mu_safe,1e-3)**power
        old_gradient=1/(1+mu_safe/c)
        if mu_safe>1e-3:old_gradient+=power*c/mu_safe*(np.log1p(mu_safe/c)-mu_safe/(c+mu_safe))
        strength.append(dict(total_population_route_scale=n,young_fraction=young,
          original_total_zero_loss=oldmass*c*np.log1p(mu_safe/c),derived_total_zero_loss=oldmass*zero(n),
          original_gradient_wrt_total=oldmass*old_gradient*scale if mu>1e-6 else 0.,derived_gradient_wrt_total=oldmass*grad(n)))
report=dict(reference_concentration=k,reference_fledgling_weight=phi,old_pseudozero_rows=int(old.sum()),
    old_effective_weight=oldmass,derived_effective_weight=float(preweights.sum()*len(oldyears)),
    old_cells=int(oldmask.sum()),derived_cells=int(pre_out.sum()),grid_cell_area_km2=54**2,
    extent=extent,matched_density_strength=strength,
    interpretation='Old extent is actual weighted sampled pseudozero support, not the unsampled continuous old hull mask. Old breeding readout uses previous total plus phi*current young; comparison assumes temporally constant population.',
    calibration='Fixed to the accepted no-CBC seed41 NB2 concentration. Evaluate additional concentration scenarios in sensitivity analysis; not an estimate of CBC detection.')
(OUT/'PSEUDOZERO_COMPARISON.json').write_text(json.dumps(report,indent=2)+'\n')
np.savez_compressed(OUT/'comparison_maps.npz',land_rows=rr,land_cols=cc,lon=lon,lat=latgrid,old_weight=oldsite,new_weight=preweights*len(oldyears),coverage=pre_coverage,new_exterior=pre_out)
np.savez_compressed(OUT/'remove_old.npz',old_mask=old.astype(np.uint8))
print(json.dumps(dict(oldmass=oldmass,newmass=float(preweights.sum()*len(oldyears)),oldcells=int(oldmask.sum()),newcells=int(pre_out.sum()),concentration=k,phi=phi),indent=2),flush=True)
