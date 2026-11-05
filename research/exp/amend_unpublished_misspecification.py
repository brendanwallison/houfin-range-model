"""Update only unpublished, never-started experiment copies, then revalidate."""
from pathlib import Path
import json,shutil,time
from stage_overflow_early import sha,write
R=Path('/work/07980/bwa386/ls6/houfin/research/grid54');specs=json.loads((R/'monitor/misspecification_20261104_specs.json').read_text())
assert not any(p for state in ['pending','running','done','failed','local_running'] for p in (R/'queue'/state).glob('*_step48_*.sh'))
changes=[]
for spec in specs:
 run=R/'runs'/spec['step'];assert not list(run.glob('ATTEMPT_*.json'))
 manifest=run/'CODE_MANIFEST.sha256';old=sha(manifest)
 for line in manifest.read_text().splitlines():
  h,rel=line.split('  ',1);assert sha(run/'code'/rel)==h
 shutil.copy2(R/'monitor/misspecification_suite.py',run/'code/src/model/misspecification_suite.py')
 code=run/'code';manifest.write_text('\n'.join(sha(p)+'  ./'+p.relative_to(code).as_posix() for p in sorted(code.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')+'\n')
 changes.append(dict(step=spec['step'],old_manifest=old,new_manifest=sha(manifest)))
proof=R/'monitor/misspecification_20261104_VALIDATED.json'
if proof.exists():proof.rename(proof.with_name('misspecification_20261104_VALIDATED_before_hook_fix.json'))
write(R/'monitor/misspecification_20261104_hook_amendment.json',dict(epoch=time.time(),published=False,reason='Preserve external diagnostic capture/counterfactual hooks; no scientific run has started.',changes=changes))
print('AMENDED_UNPUBLISHED',len(changes),flush=True)
