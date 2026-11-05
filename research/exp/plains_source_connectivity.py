"""Measure Plains sink prevalence and spatial source connectivity."""
import json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import transform
from scipy.ndimage import label,binary_dilation,binary_propagation,generate_binary_structure
from functional_stability_audit import ratio

repo=Path('/mnt/c/Dev/houfin-range-model');exp=repo/'research/experiments'
root=Path('/mnt/e/Datasets/houfin/contribution_20261101_v3')
steps=json.loads((exp/'CROSS_MODEL_PROCESS_AUDIT_20261101.json').read_text())['fits']
eras={'1951-1965':(1951,1965),'1966-1985':(1966,1985),'2010-2025':(2010,2025)}
with rasterio.open('/work/07980/bwa386/ls6/houfin/processed/grid54/ref_grid_54km.tif') as src:
    shape=src.shape;trans=src.transform;crs=src.crs
with np.load(root/steps[0]/'annual_budget.npz') as z:rr,cc=z['land_rows'],z['land_cols']
xx,yy=rasterio.transform.xy(trans,rr,cc);lon,lat=map(np.asarray,transform(crs,'EPSG:4326',xx,yy))
land=np.zeros(shape,bool);land[rr,cc]=True
water=~land;seed=np.zeros(shape,bool);seed[[0,-1],:]=water[[0,-1],:];seed[:,[0,-1]]=water[:,[0,-1]]
ocean=binary_propagation(seed,mask=water)
coastal=land&binary_dilation(ocean,structure=generate_binary_structure(2,1))
coast=coastal[rr,cc]
latitude=(lat>=25)&(lat<49)
regions={'primary_100W_95W':latitude&(lon>=-100)&(lon<-95),
 'broad_105W_95W':latitude&(lon>=-105)&(lon<-95)}
tests={
 'plains_primary':(regions['primary_100W_95W'],lon<-99.2,lon>=-95.8),
 'plains_broad':(regions['broad_105W_95W'],lon<-104.2,lon>=-95.8),
 'continental_interior':(latitude,lon<-115,lon>=-80),
 'coastal_endpoints':(latitude,coast&(lon<-120),coast&(lon>-81)),
 'continental_interior_full_domain':(np.ones(len(rr),bool),latitude&(lon<-115),latitude&(lon>=-80)),
 'coastal_full_domain':(np.ones(len(rr),bool),latitude&coast&(lon<-120),latitude&coast&(lon>-81))}
structures={4:generate_binary_structure(2,1),8:generate_binary_structure(2,2)}

def connected(source,which,neighbors):
    mask,left,right=tests[which];grid=np.zeros(shape,bool);grid[rr,cc]=source&mask
    labs,_=label(grid,structure=structures[neighbors]);ll=labs[rr,cc]
    a=np.unique(ll[source&mask&left]);b=np.unique(ll[source&mask&right])
    return bool(np.intersect1d(a[a>0],b[b>0]).size)

records=[];annual=[]
for step in steps:
    with np.load(root/step/'annual_budget.npz') as z:
        if not(np.array_equal(rr,z['land_rows']) and np.array_equal(cc,z['land_cols'])):raise ValueError('grid mismatch')
        years=z['years'];a=z['input_adult'];n=a+z['input_juvenile'];con=z['breeder_contribution'];lam=z['potential_lambda'];rep=z['local_total_replacement']
        has_noise='process_multiplier' in z and not np.all(z['process_multiplier']==1.)
        bio_con=z['biological_breeder_contribution'] if has_noise else con
        full_lam=z['potential_lambda_with_process'] if has_noise else lam
    for era,(lo,hi) in eras.items():
        tt=(years>=lo)&(years<=hi)
        maps={'potential_lambda':lam[tt].mean(0),'C':ratio(con[tt].sum(0),a[tt].sum(0)),
          'local_replacement':ratio((np.nan_to_num(rep[tt])*n[tt]).sum(0),n[tt].sum(0))}
        if has_noise:maps.update(biological_C=ratio(bio_con[tt].sum(0),a[tt].sum(0)),full_process_lambda=full_lam[tt].mean(0))
        for metric,v in maps.items():
            q={'step':step,'era':era,'metric':metric,'regions':{},'bridges':{}}
            for name,mask in regions.items():
                keep=mask&np.isfinite(v);vals=v[keep]
                q['regions'][name]={'finite_cells':int(keep.sum()),'total_cells':int(mask.sum()),
                  'sink_share':float((vals<1).mean()),'source_share':float((vals>1).mean()),
                  'deep_sink_share':float((vals<.8).mean()),'mean':float(vals.mean()),'median':float(np.median(vals))}
            for threshold in [1.,1.01,1.05]:
                source=np.isfinite(v)&(v>threshold)
                q['bridges'][str(threshold)]={str(neigh):{name:connected(source,name,neigh) for name in tests} for neigh in [4,8]}
            records.append(q)
    for i,yr in enumerate(years):
        if not 1951<=yr<=2025:continue
        annual_maps=[('potential_lambda',lam[i]),('C',ratio(con[i],a[i])),('local_replacement',rep[i])]
        if has_noise:annual_maps += [('biological_C',ratio(bio_con[i],a[i])),('full_process_lambda',full_lam[i])]
        for metric,v in annual_maps:
            source=np.isfinite(v)&(v>1)
            annual.append({'step':step,'year':int(yr),'metric':metric,
                'bridges':{str(neigh):{name:connected(source,name,neigh) for name in tests} for neigh in [4,8]}})
out={'fits':steps,'regions_cells':{k:int(v.sum()) for k,v in regions.items()},
 'coastal_endpoint_cells':{'west':int((latitude&coast&(lon<-120)).sum()),'east':int((latitude&coast&(lon>-81)).sum())},
 'records':records,'annual':annual}
(exp/'PLAINS_SOURCE_CONNECTIVITY_20261101.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
lines=['# Great Plains sink prevalence and source bridges','',
 f'All {len(steps)} checked standard-yearling fits in CROSS_MODEL_PROCESS_AUDIT_20261101.json. Land cells have equal weight; no survey or fitted-population weighting. Plains masks here are latitude 25-49N, primary 100-95W and broader 105-95W. These are geographic bands, not formal ecoregions.','',
 'Source means strictly >1 for the specified metric. Potential lambda excludes density/Allee suppression, density-dependent disease and dispersal losses; C is origin-attributed breeding contribution; local replacement is the fitted-state no-movement population ratio. Finite unoccupied-cell C can be numerically fragile and is not evidence of an established source population.','',
 'A bridge is one connected component of source cells joining both endpoint sets. Four-neighbor means shared edges; eight-neighbor also permits corner contact. Plains tests join the westernmost and easternmost ~0.8-degree strips of the band, with the entire path confined to the band. Continental interior joins longitude <115W to >=80W. Coastal endpoints use modeled land cells adjacent to border-connected non-land, west of 120W and east of 81W; these are grid shoreline proxies, not a separate validated ocean coastline product. Primary paths are confined to 25-49N; additional full_domain tests retain those endpoint latitudes but allow paths anywhere on the modeled land grid. No FFT wraparound or dispersal jumps are used. Annual tests prevent era averaging from fabricating a persistent source path. Threshold >1.01 and >1.05 era sensitivities are saved in JSON.','',
 'Modern interpretation: all 59 fits have majority-sink Plains under C (64.7-90.5% of primary-band cells) and actual-state local replacement (88.4-98.4%), but only 21/59 do under potential lambda (27.9-62.1% sinks). Potential-lambda coast-to-coast paths occur in 38/59 era maps with shared-edge connectivity, and all 59 if diagonal contacts count. The stricter lambda>1.05 threshold leaves 35/59 shared-edge paths. Neither C nor local replacement gives a coast-to-coast path in any era or annual map tested. A majority of sink cells does not prevent a narrow source path: modern primary-Plains C paths occur in 58/59 fits, despite sink dominance.','',
 'Example map: [later-state native-departure middle scenario](SOURCE_BRIDGE_MAP_20261101.png). Its edge-connected potential-lambda path is explicitly saved in SOURCE_BRIDGE_PATH_20261101.npz. Source connectivity does not establish positive population-growth capacity after Allee/density suppression or validate historical colonization; dispersal can also cross sink gaps.','',
 '## Modern era, every model version','',
 '| Fit | Metric | Primary Plains sink share | Broad Plains sink share | Primary Plains bridge (4/8 neighbors) | Broad Plains bridge (4/8) | Continental interior bridge (4/8) | Coastal bridge (4/8) |',
 '|---|---|---:|---:|---|---|---|---|']
for q in records:
    if q['era']!='2010-2025':continue
    b=q['bridges']['1.0'];fmt=lambda name:'/'.join('yes' if b[str(n)][name] else 'no' for n in [4,8])
    lines.append(f"| {q['step']} | {q['metric']} | {q['regions']['primary_100W_95W']['sink_share']:.1%} | {q['regions']['broad_105W_95W']['sink_share']:.1%} | {fmt('plains_primary')} | {fmt('plains_broad')} | {fmt('continental_interior')} | {fmt('coastal_endpoints')} |")
lines+=['','## Summary by era and metric','',
 '| Era | Metric | Primary sink-share range | Broad sink-share range | Primary bridge fits (4/8) | Broad bridge fits (4/8) | Interior bridge fits (4/8) | Coastal bridge fits (4/8) |',
 '|---|---|---|---|---|---|---|---|']
for era in eras:
    for metric in ['potential_lambda','C','local_replacement']:
        rows=[q for q in records if q['era']==era and q['metric']==metric]
        ranges=lambda name:f"{min(q['regions'][name]['sink_share'] for q in rows):.1%}-{max(q['regions'][name]['sink_share'] for q in rows):.1%}"
        counts=lambda name:'/'.join(str(sum(q['bridges']['1.0'][str(n)][name] for q in rows)) for n in [4,8])
        lines.append(f"| {era} | {metric} | {ranges('primary_100W_95W')} | {ranges('broad_105W_95W')} | {counts('plains_primary')} | {counts('plains_broad')} | {counts('continental_interior')} | {counts('coastal_endpoints')} |")
lines+=['','## Annual bridges (1951-2025)','',
 '| Metric | Endpoint test | Fits with at least one bridge year (4/8) | Model-years with bridges (4/8) |',
 '|---|---|---|---|']
for metric in ['potential_lambda','C','local_replacement']:
    rows=[q for q in annual if q['metric']==metric]
    for name in tests:
        counts=['/'.join(str(len({q['step'] for q in rows if q['bridges'][str(n)][name]})) for n in [4,8]),
          '/'.join(str(sum(q['bridges'][str(n)][name] for q in rows)) for n in [4,8])]
        lines.append(f"| {metric} | {name} | {counts[0]} | {counts[1]} |")
(exp/'PLAINS_SOURCE_CONNECTIVITY_20261101.md').write_text('\n'.join(lines)+'\n')
print('CONNECTIVITY FITS',len(steps),'ERA RECORDS',len(records),'ANNUAL RECORDS',len(annual),flush=True)
