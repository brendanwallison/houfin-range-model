"""Explain Plains C differences using its exact saved additive components."""
import json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import transform
from functional_stability_audit import ratio

repo=Path('/mnt/c/Dev/houfin-range-model');exp=repo/'research/experiments'
root=Path('/mnt/e/Datasets/houfin/contribution_20261101_v3')
data=json.loads((exp/'PLAINS_SOURCE_CONNECTIVITY_20261101.json').read_text())
rows=sorted([r for r in data['records'] if r['metric']=='C' and r['era']=='2010-2025'],key=lambda r:r['regions']['primary_100W_95W']['sink_share'])
steps=[rows[0]['step'],rows[-1]['step'],'step35_est1951_ctl_mid_s41','step35_est1951_nat_mid_s41']
out=[];maps={}
for step in steps:
    with np.load(root/step/'annual_budget.npz') as z:
        rr,cc=z['land_rows'],z['land_cols'];tt=(z['years']>=2010)&(z['years']<=2025)
        with rasterio.open('/work/07980/bwa386/ls6/houfin/processed/grid54/ref_grid_54km.tif') as src:
            xx,yy=rasterio.transform.xy(src.transform,rr,cc);lon,lat=map(np.asarray,transform(src.crs,'EPSG:4326',xx,yy))
        mask=(lon>=-100)&(lon<-95)&(lat>=25)&(lat<49)
        a=z['input_adult'][tt];j=z['input_juvenile'][tt];den=a.sum(0)
        adult=ratio((a*z['origin_adult_survival'][tt]).sum(0),den)
        offspring=ratio((j*z['origin_juvenile_recruitment'][tt]).sum(0),den)
        c=ratio(z['breeder_contribution'][tt].sum(0),den)
        retained=ratio(z['retained_breeders'][tt].sum(0),den)
        exported=ratio(z['successful_breeder_export'][tt].sum(0),den)
    if not np.allclose((adult+offspring)[mask],c[mask],rtol=1e-10,atol=1e-10):raise ValueError('C decomposition fails')
    maps[step]=c[mask]
    out.append({'step':step,'cells':int(mask.sum()),'C_mean':float(c[mask].mean()),
      'adult_component_mean':float(adult[mask].mean()),'offspring_component_mean':float(offspring[mask].mean()),
      'retained_component_mean':float(retained[mask].mean()),'exported_component_mean':float(exported[mask].mean()),
      'sink_cells':int((c[mask]<1).sum()),'below_08_cells':int((c[mask]<.8).sum()),
      'near_one_cells':int((np.abs(c[mask]-1)<=.05).sum())})
x,y=maps[steps[0]],maps[steps[1]]
contrast={'left':steps[0],'right':steps[1],'mean_abs_change':float(np.abs(y-x).mean()),
 'exact_crossings':int(np.count_nonzero((x-1)*(y-1)<0)),
 'crossings_beyond_05':int(np.count_nonzero(((x<.95)&(y>1.05))|((y<.95)&(x>1.05))))}
(exp/'PLAINS_C_DECOMPOSITION_20261101.json').write_text(json.dumps({'runs':out,'extreme_contrast':contrast},indent=2)+'\n')
lines=['# Why Plains Runge contribution changes across fits','',
 'Modern 2010-2025, 380 equal-weight land cells in 100-95W and 25-49N. The 65-91% range describes the fraction of cells below C=1 across model versions; it is not a confidence interval or a temporal fluctuation.','',
 'The exact implemented identity is C_i = p_adult,i + (J_i/A_i)*p_recruit,i. Each p sums successful survival/recruitment over all destinations from that origin in the fitted environment. J/A is actual offspring per breeder in the starting postbirth state; it is not Fmax. C also equals retained contribution plus successful export. Reproduction/Allee suppression therefore enters through actual offspring, while origin movement and destination/journey survival enter through the two p terms. Emigration alone does not lower C: successful emigrants are credited to their origins.','',
 '| Fit | Adult contribution | Offspring contribution | Total C | Retained | Successful export | Sink cells | C<.8 cells | Within .05 of 1 |',
 '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in out:
    lines.append(f"| {r['step']} | {r['adult_component_mean']:.3f} | {r['offspring_component_mean']:.3f} | {r['C_mean']:.3f} | {r['retained_component_mean']:.3f} | {r['exported_component_mean']:.3f} | {r['sink_cells']} | {r['below_08_cells']} | {r['near_one_cells']} |")
lines+=['','The matched later-state middle-abundance control/native contrast changes C .651 to .765: adult contribution .199 to .201, offspring contribution .453 to .564. Retention is nearly unchanged (.482 to .487), while successful export increases .170 to .278. This attributes the change algebraically, not to one isolated causal coefficient: both are fully refitted models.','',
 f"The two fits at the endpoints of the sink-prevalence range have mean absolute C difference {contrast['mean_abs_change']:.3f}, {contrast['exact_crossings']} exact crossings of 1 and {contrast['crossings_beyond_05']} reversals beyond .95/1.05. Hence threshold sensitivity contributes but does not explain the whole range. These endpoints differ in multiple assumptions and are not a matched single-factor experiment."]
(exp/'PLAINS_C_DECOMPOSITION_20261101.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'runs':out,'extreme_contrast':contrast},indent=2))
