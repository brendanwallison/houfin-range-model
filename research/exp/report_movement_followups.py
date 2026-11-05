"""Report actual movement moments and distinguish local propagule failure modes."""
import argparse,json,os
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
lines=['# Permanent movement and local establishment checks — November 1, 2026','',
 'These are calculated from the frozen fitted movement kernels, actual annual departure and destination survival. Distances concern successful off-cell adult survivors and juvenile recruits entering the next breeding census. They are not distances of attempted departures, movements of all birds, seasonal migration or direct banding estimates. Every completed audit is shown.','',
 '| Fit | Region | Mean successful movement km | Fraction >300 km | Fraction >1000 km |','|---|---|---:|---:|---:|']
records=[]
for f in sorted(a.root.glob('*/READY.json')):
 try:r=json.loads(f.read_text())
 except json.JSONDecodeError:continue
 records.append((f,r))
 for reg in ['west','plains','broad_plains','east']:
  b=r['regions'][reg]
  if not (-1e-6<=b['tail_1000km']<=b['tail_300km']+1e-6<=1.000002):raise ValueError('invalid tail moments '+str(f))
  lines.append(f"| {f.parent.name} | {reg} | {b['distance_km']:.1f} | {b['tail_300km']:.3%} | {b['tail_1000km']:.3%} |")
lines+=['','## Finite local propagules, separated by mechanism','',
 'Constant modern mean fields, actual reproductive law and disease mortality response, no movement and no release exemption; 80 years from equal adult/young starting densities at .01K, .1K, K and 10K. Persistence below means final abundance >.01K, not an establishment probability. Cached class labels use the saved mean-environment definition and are not recomputed from the outcome. Fundamentally unfavorable cells, viable cells with an Allee threshold and positive-growth cells with K/Allee-induced nonviability must be interpreted separately. No demographic stochasticity or within-cell spatial pairing is simulated.','',
 '| Fit | Local mechanism class (whole domain) | Cells | Persist from .01K | Persist from .1K | Persist from K | Persist from 10K |','|---|---|---:|---:|---:|---:|---:|']
for f,r in records:
 field=Path('/mnt/e/Datasets/houfin/contribution_20261101_v3')/f.parent.name/'source_sink_fields.npz'
 if not field.exists():continue
 with np.load(f.parent/'movement_moments.npz') as z:lr,lc=z['land_rows'],z['land_cols']
 with np.load(field) as z:
  dead=z['allee_dead_mask'][lr,lc].astype(bool);potential=z['lam_fundamental_modern'][lr,lc]>1
 with np.load(f.parent/'local_propagules.npz') as z:
  k=z['K'];classes={'potential growth ≤1':~potential,'positive growth, K/Allee nonviable':dead,'positive growth, locally viable':potential&~dead}
  for label,mask in classes.items():
   if not mask.any():continue
   values=[float(np.mean(z['seed_'+s][-1,mask]>.01*k[mask])) for s in ['0.01','0.1','1.0','10.0']]
   lines.append(f"| {f.parent.name} | {label} | {int(mask.sum())} | "+' | '.join(f'{v:.1%}' for v in values)+' |')
lines+=['',f'Artifacts: `{a.root}`. Independent two-cell enumeration verifies that distance attribution excludes stayers and weights movers by destination survival. Full origin/time moment arrays and complete local trajectories are preserved. These checks supplement contribution/BIDE and do not select a configuration or establish that the inferred permanent movement distribution is biologically credible.']
tmp=a.out.with_name(a.out.name+f'.{os.getpid()}.tmp');tmp.write_text('\n'.join(lines)+'\n');tmp.replace(a.out)
print('MOVEMENT REPORT',len(records),'completed audits')
