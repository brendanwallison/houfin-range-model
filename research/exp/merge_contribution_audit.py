"""Join checked receipts into the full audit without replacing older snapshots."""
import argparse,json
from pathlib import Path
from full_pathology_audit import render

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--audit',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args();d=json.loads(a.audit.read_text())
hmcfile=a.audit.parents[2]/'.research_runtime/hmc_diagnostic_status.json'
hmc=json.loads(hmcfile.read_text()).get('runs',{}) if hmcfile.exists() else {}
for r in d['runs']:
    movement=a.root.parent/'movement_followup_20261101'/r['step']/'READY.json'
    if movement.exists():r['movement_diagnostics']=json.loads(movement.read_text())
    entries={b['mode']:b for b in hmc.values() if b['step']==r['step']}
    if entries:r['hmc_status']=entries
    receipt=a.root/r['step']/'READY.json'
    if receipt.exists():
        c=json.loads(receipt.read_text())
        if Path(c['run']).name!=Path(r['refined_dir']).name:raise ValueError('checkpoint run mismatch')
        context=receipt.parent/'CONTEXT.json'
        if context.exists():
            for reg,v in json.loads(context.read_text()).items():c['regions'][reg]['modern'].update(v)
        r['contribution_budget']=c
d['contribution_coverage']=sum('contribution_budget' in r for r in d['runs'])
d['movement_coverage']=sum('movement_diagnostics' in r for r in d['runs'])
a.out.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
render(d,a.out.with_suffix('.md'))
print('FULL AUDIT CONTRIBUTIONS',d['contribution_coverage'])
