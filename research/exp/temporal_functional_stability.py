"""Era stability on fixed modern surveyed support, including pre-BBS years."""
import json,pickle
from pathlib import Path
import numpy as np
from functional_stability_audit import compare,ratio

repo=Path('/mnt/c/Dev/houfin-range-model')
exp=repo/'research/experiments'
root=Path('/mnt/e/Datasets/houfin/contribution_20261101_v3')
audit=json.loads((exp/'FULL_PATHOLOGY_AUDIT_20261101_CONTRIBUTION.json').read_text())
ref=next(r for r in audit['runs'] if r['step']=='step33_nestcbc_abs_s41')
m=pickle.load(open(Path(ref['config']['input_dir'])/'metadata.pkl','rb'))
with np.load(root/'step35_est1951_ctl_mid_s41/annual_budget.npz') as z:
    rr,cc=z['land_rows'],z['land_cols']
index={(int(r),int(c)):i for i,(r,c) in enumerate(zip(rr,cc))}
oy=np.asarray(m['years'])[np.asarray(m['obs_time_indices'],int)]
oi=np.array([index.get((int(r),int(c)),-1) for r,c in zip(m['obs_rows'],m['obs_cols'])])
use=(oy>=2010)&(oy<=2025)&(np.asarray(m['observed_results'])>0)&(oi>=0)
w=np.bincount(oi[use],minlength=len(rr)).astype(float)
eras={'1951-1965':(1951,1965),'1966-1985':(1966,1985),'1986-2009':(1986,2009),'2010-2025':(2010,2025)}
cache={}
for typ in ['ctl','nat']:
    for level in ['lo','mid','hi']:
        step=f'step35_est1951_{typ}_{level}_s41'
        with np.load(root/step/'annual_budget.npz') as z:
            if not(np.array_equal(rr,z['land_rows']) and np.array_equal(cc,z['land_cols'])):raise ValueError('grid mismatch')
            years=z['years'];a=z['input_adult'];c=z['breeder_contribution'];l=z['potential_lambda']
        cache[step]={}
        for era,(lo,hi) in eras.items():
            tt=(years>=lo)&(years<=hi);den=a[tt].sum(0)
            cache[step][era]={'C':ratio(c[tt].sum(0),den),'lambda':ratio((l[tt]*a[tt]).sum(0),den)}
out=[]
for typ in ['ctl','nat']:
    x=f'step35_est1951_{typ}_mid_s41'
    for level in ['lo','hi']:
        y=f'step35_est1951_{typ}_{level}_s41'
        for era in eras:
            out.append({'type':typ,'level':level,'left':x,'right':y,'era':era,
                'metrics':{metric:compare(cache[x][era][metric],cache[y][era][metric],w,True) for metric in ['C','lambda']}})
(exp/'TEMPORAL_FUNCTIONAL_STABILITY_20261101.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
lines=['# Temporal functional stability across founder scenarios','',
 'All eras use the same 1,689 cells with positive 2010-2025 BBS records, weighted by those record counts. Thus 1951-1965 is evaluated on later surveyed habitat, not historical BBS evidence. This is a fitted-process sensitivity analysis, not validation of the unobserved historical state. Both metrics use starting-breeder temporal exposure weights within cells. Lambda is potential Sa + Sj*Fmax, excluding density/Allee suppression, density-dependent disease and movement losses. Results pool years within each era and do not certify every annual map.','',
 '| Branch | Scenario vs middle | Era | C correlation | Lambda correlation | Mean absolute C change | Mean absolute lambda change | Cells crossing C=1 | Cells crossing lambda=1 | Common weighted coverage |',
 '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
for r in out:
    c,l=(r['metrics'][k] for k in ['C','lambda'])
    lines.append(f"| {r['type']} | {r['level']} | {r['era']} | {c['weighted_pearson']:.3f} | {l['weighted_pearson']:.3f} | {c['mean_abs_change']:.3f} | {l['mean_abs_change']:.3f} | {c['categories']['0.0']['source_sink_reversal_cells']} | {l['categories']['0.0']['source_sink_reversal_cells']} | {c['coverage']:.3f} |")
(exp/'TEMPORAL_FUNCTIONAL_STABILITY_20261101.md').write_text('\n'.join(lines)+'\n')
print('TEMPORAL COMPARISONS',len(out))
