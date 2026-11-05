"""Two seed checks and two matched later-state NestWatch ablations."""
import argparse,copy,fcntl,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
R=Path('/work/07980/bwa386/ls6/houfin/research/grid54')
CASES=[('step35_est1951_ctl_mid_s42','step35_est1951_ctl_mid_s41',42,False),
 ('step35_est1951_nat_mid_s42','step35_est1951_nat_mid_s41',42,False),
 ('step36_est1951_nonest_ctl_s41','step35_est1951_ctl_mid_s41',41,True),
 ('step36_est1951_nonest_nat_s41','step35_est1951_nat_mid_s41',41,True)]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute');a=p.parse_args()
if a.execute:
 if a.execute not in [s for s,_,_,_ in CASES]:raise ValueError('invalid probe')
 run=R/'runs'/a.execute
 with (run/'attempt.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (run/'ATTEMPT_STARTED.json').exists():
   if not (run/'out/MAP_DONE').exists() or (run/'out/FIT_DONE').exists():raise RuntimeError('partial attempt preserved for review')
   write(run/f'RESUME_STARTED_{time.time_ns()}.json',{'epoch':time.time()})
  else:write(run/'ATTEMPT_STARTED.json',{'epoch':time.time()})
  start=time.time();rc=subprocess.run(['timeout','--kill-after=30','7200','bash',str(run/'run_candidate.sh')]).returncode
  write(run/'RUN_RECEIPT.json',{'exit_code':rc,'elapsed_seconds':time.time()-start,'started':start})
 raise SystemExit(rc)
new=[]
for step,parent,seed,nonest in CASES:
 src=R/'runs'/parent;out=R/'runs'/step
 if out.exists():raise FileExistsError(out)
 for line in (src/'CODE_MANIFEST.sha256').read_text().splitlines():
  sha,rel=line.split('  ',1)
  if digest(src/'code'/rel)!=sha:raise ValueError('parent source mismatch')
 out.mkdir();shutil.copytree(src/'code',out/'code',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 code=out/'code';cfg=json.loads((code/f'config/overlays/grid54_{parent}.json').read_text());pop=cfg['population_model']
 if nonest:pop['nest_likelihood']['enabled']=False
 spec=copy.deepcopy(json.loads((src/'RUN_SPEC.json').read_text()))
 spec.update(step=step,replicate_of=parent,seed=seed,queue_priority='0635',nest_likelihood=not nonest,
   purpose='Matched later-state NestWatch ablation' if nonest else 'Cold seed sensitivity of later-state compensation')
 cfg.pop('_pathology_candidate',None);cfg['_later_established_state']=spec
 for k in ['map','vis_map']:cfg['run_names'][k]='age_map_{precision}_grid54_'+step
 write(code/f'config/overlays/grid54_{step}.json',cfg);write(out/'RUN_SPEC.json',spec)
 shutil.copy2(src/'grid54_fit.sh',out/'grid54_fit.sh')
 script=(src/'run_candidate.sh').read_text().replace(parent,step).replace('HOUFIN_MAP_SEED=41',f'HOUFIN_MAP_SEED={seed}')
 (out/'run_candidate.sh').write_text(script)
 (out/'CODE_MANIFEST.sha256').write_text('\n'.join(digest(f)+'  ./'+f.relative_to(code).as_posix() for f in sorted(code.rglob('*')) if f.is_file())+'\n')
 q=R/'queue/pending'/f'0635_{step}.sh';tmp=q.with_suffix('.tmp')
 tmp.write_text(f'#!/bin/bash\nset -euo pipefail\nexport GRID54_MEM_FRACTION=.85 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4\nexec {R.parents[1]}/venv/bin/python {R}/monitor/stage_initialization_probes.py --execute {step}\n')
 os.rename(tmp,q);new.append(spec);print('QUEUED',step,flush=True)
for nonest,filename in [(False,'later_state_specs.json'),(True,'nest_ablation_specs.json')]:
 path=R/'monitor'/filename;old=json.loads(path.read_text());old.extend(s for s in new if s['step'].startswith('step36_')==nonest);write(path,old)
write(R/'monitor/initialization_probe_specs.json',new)
