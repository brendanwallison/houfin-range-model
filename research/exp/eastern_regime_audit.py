"""Separate eastern growth levels/time transitions from within-era fit stability."""
import json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import transform
from functional_stability_audit import compare,ratio

repo=Path('/mnt/c/Dev/houfin-range-model');exp=repo/'research/experiments'
root=Path('/mnt/e/Datasets/houfin/contribution_20261101_v3')
pairs=json.loads((exp/'FUNCTIONAL_STABILITY_20261101.json').read_text())
steps=sorted({r[k] for r in pairs for k in ['left','right']})
eras={'1951-1965':(1951,1965),'1966-1985':(1966,1985),'2010-2025':(2010,2025)}
cache={};levels=[];time_changes=[];reference_grid=None
for s in steps:
    with np.load(root/s/'annual_budget.npz') as z:
        years=z['years'];rr,cc=z['land_rows'],z['land_cols']
        if reference_grid is None:
            reference_grid=(rr.copy(),cc.copy())
            with rasterio.open('/work/07980/bwa386/ls6/houfin/processed/grid54/ref_grid_54km.tif') as src:
                xx,yy=rasterio.transform.xy(src.transform,rr,cc)
                lon,lat=map(np.asarray,transform(src.crs,'EPSG:4326',xx,yy))
            mask=(lon>=-80)&(lat>=25)&(lat<49)
        elif not all(np.array_equal(u,v) for u,v in zip(reference_grid,(rr,cc))):raise ValueError('grid mismatch')
        a=z['input_adult'];c=z['breeder_contribution'];l=z['potential_lambda']
    cache[s]={}
    for era,(lo,hi) in eras.items():
        tt=(years>=lo)&(years<=hi);den=a[tt].sum(0)
        # Arithmetic lambda maps avoid changing model-population weights.
        maps={'C':ratio(c[tt].sum(0),den),'potential_lambda':l[tt].mean(0)}
        cache[s][era]=maps
        for metric,v in maps.items():
            keep=mask&np.isfinite(v);q=v[keep]
            levels.append({'step':s,'era':era,'metric':metric,'cells':int(keep.sum()),
              'mean':float(q.mean()),'median':float(np.median(q)),
              'sink_cells':int((q<1).sum()),'source_cells':int((q>1).sum()),
              'deep_deficit_cells':int((q<.8).sum()),'large_surplus_cells':int((q>1.2).sum())})
    for metric in ['C','potential_lambda']:
        x,y=(cache[s][e][metric] for e in ['1951-1965','2010-2025'])
        keep=mask&np.isfinite(x)&np.isfinite(y)
        time_changes.append({'step':s,'metric':metric,'cells':int(keep.sum()),
          'sink_to_source_cells':int(np.count_nonzero(keep&(x<1)&(y>1))),
          'source_to_sink_cells':int(np.count_nonzero(keep&(x>1)&(y<1))),
          'below_08_to_above_12_cells':int(np.count_nonzero(keep&(x<.8)&(y>1.2)))})
contrasts=[]
for r in pairs:
    for era in eras:
        for metric in ['C','potential_lambda']:
            contrasts.append({'left':r['left'],'right':r['right'],'contrast':r['contrast'],'era':era,'metric':metric,
              **compare(cache[r['left']][era][metric],cache[r['right']][era][metric],mask.astype(float),True)})
(exp/'EASTERN_REGIME_AUDIT_20261101.json').write_text(json.dumps({'levels':levels,'temporal_transitions':time_changes,'cross_fit':contrasts},indent=2,allow_nan=False)+'\n')
lines=['# Eastern regime levels and transitions','',
 'Scope: the completed fits participating in the functional comparisons, not the entire historical project archive. East is longitude >= -80, latitude 25-49. Every finite land cell has equal weight; no positive-observation or model-abundance weighting across cells. Lambda maps are arithmetic annual means of potential Sa+Sj*Fmax (before Allee/density suppression, density-dependent disease and movement losses). C maps retain breeder-exposure weighting within cells over time. C in essentially unoccupied habitat is not evidence of an established local population; finite common support is reported. The two metrics and census definitions must not be interchanged.','',
 'Correlation measures spatial pattern similarity, not equality of levels or constancy of source/sink status through time. The earlier high-correlation tables predominantly concern within-structure founder contrasts in the same era. They cannot clear the historical eastern growth-opening pathology.','',
 '| Fit | Era | Metric | Cells | Mean | Median | Sink cells (<1) | Source cells (>1) | Deep deficit (<.8) | Large surplus (>1.2) |',
 '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
for r in levels:
    lines.append(f"| {r['step']} | {r['era']} | {r['metric']} | {r['cells']} | {r['mean']:.3f} | {r['median']:.3f} | {r['sink_cells']} | {r['source_cells']} | {r['deep_deficit_cells']} | {r['large_surplus_cells']} |")
lines+=['','## Same-cell temporal transitions: 1951-1965 to 2010-2025','',
 '| Fit | Metric | Common finite cells | Sink to source | Source to sink | Below .8 to above 1.2 |',
 '|---|---|---:|---:|---:|---:|']
for r in time_changes:
    lines.append(f"| {r['step']} | {r['metric']} | {r['cells']} | {r['sink_to_source_cells']} | {r['source_to_sink_cells']} | {r['below_08_to_above_12_cells']} |")
(exp/'EASTERN_REGIME_AUDIT_20261101.md').write_text('\n'.join(lines)+'\n')
print('EASTERN REGIME FITS',len(steps),'CELLS',int(mask.sum()))
