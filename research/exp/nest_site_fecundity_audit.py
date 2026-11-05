"""Evaluate annual offspring at identical nest sites even when nests were withheld."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
d=json.load(open(a.audit));root=Path('/mnt/e/Datasets/houfin/contribution_20261101_v3')
with np.load('/mnt/e/Datasets/houfin/grid54/nestwatch/nests_hofi_v6.npz') as z:n={k:z[k] for k in z}
keep=(n['year']>=1902)&(n['year']<=2020)&(n['precision']<=1)&((n['has_exp']==1)|((n['outcome']==1)&(n['fledged']>=1)))&(n['t_idx']>=0)&(n['t_idx']<124)
rows=[]
for r in d['runs']:
    rep=(r.get('config') or {}).get('population_model',{}).get('reproduction',{})
    if not r['valid'] or not rep.get('yearling_breeding') or rep.get('form','standard')!='standard':continue
    path=root/r['step']/'annual_budget.npz'
    if not path.exists():continue
    with np.load(path) as z:
        f=z['realized_offspring_per_breeder'][n['t_idx'][keep],n['land_idx'][keep]]
        positive=z['next_adult'][n['t_idx'][keep],n['land_idx'][keep]]>0
    if not positive.all():raise ValueError('ratio unavailable at zero-breeder nest site '+r['step'])
    result={'step':r['step'],'n_attempts':int(keep.sum()),'annual_offspring_at_nest_sites_mean':float(f.mean()),
        'annual_offspring_q10_50_90':np.quantile(f,[.1,.5,.9]).tolist(),
        'scope':'Realized annual output at identical training nest cell-years, not fledglings per attempt or an independently fitted nest likelihood'}
    saved=(r.get('pathology') or {}).get('nests',{}).get('nest_F_real')
    if saved:
        result['mean_parity_error']=abs(float(f.mean())-saved['mean'])
        if result['mean_parity_error']>1e-8:raise ValueError('production nest readout parity failed '+r['step'])
    rows.append(result)
a.out.write_text(json.dumps(rows,indent=2)+'\n');print('NEST SITE READOUTS',len(rows),'attempts',int(keep.sum()))
for r in rows:
    if r['step'].startswith(('step35_','step36_')):print(r['step'],r['annual_offspring_at_nest_sites_mean'])
