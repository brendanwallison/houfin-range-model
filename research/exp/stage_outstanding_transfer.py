"""Matched weak-arrival and historical-range/floor transfer tests."""
import argparse,fcntl,json,math,os,shutil,subprocess,time
from pathlib import Path
from stage_overflow_early import replace,sha,write
R=Path('/work/07980/bwa386/ls6/houfin/research/grid54')
REG='outstanding_transfer'
def cases():
 for seed in [41,42,43]:
  for form in ['uniform05','capacity05']:
   yield dict(step=f'step46_{form}_s{seed}',parent='step43_settlement_floor4_s41',seed=seed,family=form,
      reference=f'step39_kfloor4p0_s{seed}' if seed!=43 else 'step44_shared_s43',queue_priority='0612')
 for seed in [41,42]:
  for form,parent in [('original',f'step38_cbc1951_noncbc_free_s{seed}'),('historical',f'step40_rangev2_historical_s{seed}'),('invasion',f'step40_rangev2_invasion_s{seed}')]:
   yield dict(step=f'step47_{form}_floor4_s{seed}',parent=parent,seed=seed,family=form,reference=parent,queue_priority='0613')
def stage():
 specs=[]
 for s in cases():
  source=R/'runs'/s['parent'];run=R/'runs'/s['step']
  for line in (source/'CODE_MANIFEST.sha256').read_text().splitlines():
   h,p=line.split('  ',1);assert sha(source/'code'/p)==h
  if run.exists():raise FileExistsError(run)
  run.mkdir();shutil.copytree(source/'code',run/'code',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
  code=run/'code';cfg=json.loads((code/f'config/overlays/grid54_{s["parent"]}.json').read_text());pop=cfg['population_model']
  assert not pop['cbc_likelihood']['enabled'] and not pop.get('native_departure',{}).get('enabled') and not pop.get('process_noise',{}).get('enabled')
  pop['capacity_floor_internal']=4*math.log1p(math.exp(pop['pinned_sites']['n50_raw']))/20
  if s['step'].startswith('step46_'):
   assert pop['pinned_sites']['log_inv_total']==0
   pop['local_arrival_settlement'].update(fraction=.05,weight_mode=s['family'])
   if s['family']=='uniform05':
    p=code/'src/model/local_arrival_settlement.py';t=p.read_text();assert t.count('w=jnp.maximum(capacity,0.);')==3
    p.write_text(t.replace('w=jnp.maximum(capacity,0.);','w=(capacity>0).astype(capacity.dtype);'))
  else:
   assert 'log_inv_total' not in pop['pinned_sites']
   p=code/'src/model/age_fields.py'
   replace(p,'from jax import lax, checkpoint',"from jax import lax, checkpoint\nfrom src.config_utils import load_age_model_config\n_FLOOR=float(load_age_model_config()['population_model']['capacity_floor_internal'])")
   replace(p,'        # 5. Map path habitat to Journey Survival','        K_val=K_val+_FLOOR\n\n        # 5. Map path habitat to Journey Survival')
  for k in ['map','vis_map']:cfg['run_names'][k]='age_map_{precision}_grid54_'+s['step']
  s.update(registered='OUTSTANDING_WORK_20261103.md',capacity_floor_internal=pop['capacity_floor_internal'],initial_abundance='fixed nominal' if s['step'].startswith('step46') else 'free',local_arrival_settlement=pop.get('local_arrival_settlement'),historical_range=pop.get('historical_range'))
  cfg['_outstanding_transfer']=s;write(code/f'config/overlays/grid54_{s["step"]}.json',cfg)
  audit=code/'research/outstanding_transfer_audit';audit.mkdir(parents=True)
  for name in ['contribution_budget.py','run_contribution_audit.py']:shutil.copy2(R/'monitor'/name,audit/name)
  shutil.copy2(source/'grid54_fit.sh',run/'grid54_fit.sh')
  runner=(source/'run_candidate.sh').read_text().replace(s['parent'],s['step'])
  import re
  runner=re.sub(r'HOUFIN_MAP_SEED=\d+',f'HOUFIN_MAP_SEED={s["seed"]}',runner)
  runner=re.sub(r'\S*/run_contribution_audit.py',str(audit/'run_contribution_audit.py'),runner)
  runner='\n'.join(line for line in runner.splitlines() if 'score_cbc_joint.py' not in line)+'\n'
  (run/'run_candidate.sh').write_text(runner);write(run/'RUN_SPEC.json',s)
  (run/'CODE_MANIFEST.sha256').write_text('\n'.join(sha(p)+'  ./'+p.relative_to(code).as_posix() for p in sorted(code.rglob('*')) if p.is_file())+'\n');specs.append(s)
 write(R/f'monitor/{REG}_specs.json',specs);print('STAGED',len(specs),flush=True)
def publish():
 v=json.loads((R/f'monitor/{REG}_VALIDATED.json').read_text());assert v['passed']
 for s in json.loads((R/f'monitor/{REG}_specs.json').read_text()):
  run=R/'runs'/s['step'];assert sha(run/'CODE_MANIFEST.sha256')==v['manifests'][s['step']]
  for line in (run/'CODE_MANIFEST.sha256').read_text().splitlines():
   h,p=line.split('  ',1);assert sha(run/'code'/p)==h
  name=f'{s["queue_priority"]}_{s["step"]}.sh'
  if any((R/'queue'/state/name).exists() for state in ['pending','running','local_running','done','failed']):continue
  p=R/'queue/pending'/name;tmp=p.with_suffix('.tmp')
  tmp.write_text(f'#!/bin/bash\nset -euo pipefail\nexport GRID54_MEM_FRACTION=.85 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4\nexec {R.parents[1]}/venv/bin/python {R}/monitor/stage_outstanding_transfer.py --execute {s["step"]}\n');tmp.replace(p)
 print('PUBLISHED',12,flush=True)
def execute(step):
 assert step in [s['step'] for s in json.loads((R/f'monitor/{REG}_specs.json').read_text())]
 run=R/'runs'/step
 with (run/'attempt.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.time()
  write(run/f'ATTEMPT_{time.time_ns()}.json',dict(epoch=start,slurm_job=os.environ.get('SLURM_JOB_ID'),gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))
  rc=subprocess.run(['timeout','--kill-after=30','14400','bash',str(run/'run_candidate.sh')]).returncode
  write(run/'RUN_RECEIPT.json',dict(exit_code=rc,elapsed_seconds=time.time()-start))
 raise SystemExit(rc)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--publish',action='store_true');p.add_argument('--execute');a=p.parse_args()
 execute(a.execute) if a.execute else publish() if a.publish else stage()