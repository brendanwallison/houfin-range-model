"""Diagnostic-only order correction; never changes a fitted biological model."""
from pathlib import Path
import json,shutil,time
from stage_overflow_early import sha,write
R=Path('/work/07980/bwa386/ls6/houfin/research/grid54')
specs=json.loads((R/'monitor/misspecification_20261104_specs.json').read_text())
assert not list((R/'queue/running').glob('*_step48_*.sh')),'amend only while suite is idle'
records=[]
for s in specs:
 run=R/'runs'/s['step'];code=run/'code';p=code/'research/misspecification_audit/contribution_budget.py'
 old="        amp=grid(x['disease_m_amp'][t]);n=ap+js+ja"
 text=p.read_text();assert text.count(old)==1
 backup=run/'diagnostic_amendments/disease_order_20261104';backup.mkdir(parents=True,exist_ok=False)
 shutil.copy2(p,backup/p.name);shutil.copy2(run/'CODE_MANIFEST.sha256',backup/'CODE_MANIFEST.sha256')
 p.write_text(text.replace(old,"""        amp=grid(x['disease_m_amp'][t])
        # Fitted settlement reweighting occurs after density-dependent disease.
        ja_exposure = ja/jnp.maximum(grid(x['mm_arrival_weight'][t]),1e-30) if 'mm_arrival_weight' in x else ja
        n=ap+js+ja_exposure"""))
 manifest=run/'CODE_MANIFEST.sha256'
 lines=[]
 for line in manifest.read_text().splitlines():
  h,rel=line.split('  ',1)
  if rel.endswith('/research/misspecification_audit/contribution_budget.py'):h=sha(code/rel)
  assert sha(code/rel)==h,rel
  lines.append(h+'  '+rel)
 manifest.write_text('\n'.join(lines)+'\n')
 records.append(dict(step=s['step'],budget_sha256=sha(p),model_unchanged=True))
write(R/'monitor/MISSPEC_DISEASE_AUDIT_AMENDMENT_20261104.json',dict(epoch=time.time(),reason='Replay had applied destination settlement weights before computing disease exposure; forward applies them afterward.',records=records))
print('DIAGNOSTIC_ONLY_AMENDMENT',len(records))
