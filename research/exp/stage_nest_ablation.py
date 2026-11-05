"""Execute the omitted matched NestWatch ablation on the current absolute model."""
import argparse,fcntl,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--execute');a=p.parse_args()
if a.root.resolve()!=Path('/work/07980/bwa386/ls6/houfin/research/grid54'):raise ValueError('wrong root')
if a.execute:
 if not a.execute.startswith('step36_nonest_') or '/' in a.execute:raise ValueError('invalid step')
 run=a.root/'runs'/a.execute
 with (run/'attempt.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (run/'ATTEMPT_STARTED.json').exists():raise RuntimeError('existing attempt preserved')
  start=time.time();write(run/'ATTEMPT_STARTED.json',{'epoch':start})
  rc=subprocess.run(['timeout','--kill-after=30','7200','bash',str(run/'run_candidate.sh')]).returncode
  write(run/'RUN_RECEIPT.json',{'exit_code':rc,'elapsed_seconds':time.time()-start,'started':start})
 raise SystemExit(rc)
specs=[]
for native in [False,True]:
 for seed in [41,42]:
  parent=(f'step34_abs_natdep_s{seed}' if native else 'step33_nestcbc_abs_s41' if seed==41 else 'step34_abs_s42')
  name=f"step36_nonest_{'nat' if native else 'ctl'}_s{seed}"
  src=a.root/'runs'/parent;run=a.root/'runs'/name
  if run.exists():raise FileExistsError(run)
  for line in (src/'CODE_MANIFEST.sha256').read_text().splitlines():
   sha,rel=line.split('  ',1)
   if digest(src/'code'/rel)!=sha:raise ValueError('changed parent source')
  run.mkdir();shutil.copytree(src/'code',run/'code',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
  code=run/'code';cfg=json.loads((code/f'config/overlays/grid54_{parent}.json').read_text())
  cfg['population_model']['nest_likelihood']['enabled']=False
  cfg.pop('_second_wave',None)
  spec={'step':name,'parent':parent,'seed':seed,'native_departure':native,'nest_likelihood':False,
     'input_observations_features_and_all_other_biological_settings_unchanged':True,
     'purpose':'Matched absolute-Allee NestWatch off/on comparison conditional on existing initialization; not a claim that 1940 founder treatment is validated'}
  cfg['_nest_ablation']=spec
  for key in ['map','vis_map']:cfg['run_names'][key]='age_map_{precision}_grid54_'+name
  write(code/f'config/overlays/grid54_{name}.json',cfg)
  shutil.copy2(src/'grid54_fit.sh',run/'grid54_fit.sh')
  script=(src/'run_candidate.sh').read_text().replace(parent,name)
  script+=f'\nexport JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2\nexport PYTHONPATH=$CODE:$CODE/src:{a.root}/monitor\n"$PY" {a.root}/monitor/run_contribution_audit.py --run "$REF" --out "$R/out/contribution" > "$R/out/contribution.log" 2>&1\n'
  (run/'run_candidate.sh').write_text(script);write(run/'RUN_SPEC.json',spec)
  (run/'CODE_MANIFEST.sha256').write_text('\n'.join(digest(f)+'  ./'+f.relative_to(code).as_posix() for f in sorted(code.rglob('*')) if f.is_file())+'\n')
  q=a.root/'queue/pending'/f'0620_{name}.sh';tmp=q.with_suffix('.tmp')
  tmp.write_text(f'#!/bin/bash\nset -euo pipefail\nexport GRID54_MEM_FRACTION=.85 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4\nexec {a.root.parents[1]}/venv/bin/python {a.root}/monitor/stage_nest_ablation.py --root {a.root} --execute {name}\n')
  os.rename(tmp,q);specs.append(spec);print('QUEUED',name,flush=True)
write(a.root/'monitor/nest_ablation_specs.json',specs)
