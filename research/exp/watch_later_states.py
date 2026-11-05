"""Persistent collection and scoring of the launched established-state fits."""
import json,subprocess,time,fcntl,os
from pathlib import Path
HOST='bwa386@ls6.tacc.utexas.edu';R='/work/07980/bwa386/ls6/houfin/research/grid54'
REPO=Path('/mnt/c/Dev/houfin-range-model');OUT=Path('/mnt/e/Datasets/houfin/contribution_20261101_v3')
SSH=['ssh','-S',str(Path.home()/'.ssh/cm-tacc'),'-o','BatchMode=yes',HOST]
SCP=['scp','-r','-o','ControlPath='+str(Path.home()/'.ssh/cm-tacc')]
P=str(Path.home()/'.local/bin/pueue');PY='/home/breallis/houfin/venv/bin/python'
state=REPO/'.research_runtime/later_state_status.json'
lock=(REPO/'.research_runtime/later_state_watch.lock').open('w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
while not (REPO/'.research_runtime/STOP_LATER_STATE_WATCH').exists():
 try:
  raw=subprocess.check_output(SSH+[f'/work/07980/bwa386/ls6/houfin/venv/bin/python {R}/monitor/later_state_status.py --root {R}'],timeout=45)
  d=json.loads(raw);tmp=state.with_name(f'.later_state_status.{os.getpid()}.tmp');tmp.write_bytes(raw);tmp.replace(state)
  tasks=json.loads(subprocess.check_output([P,'status','--json']))['tasks']
  labels={t.get('label') for t in tasks.values()}
  for s,b in d['runs'].items():
   if b['contribution_done'] and not (OUT/s/'READY.json').exists():
    staging=OUT/('.collect_'+s);staging.mkdir(parents=True,exist_ok=True)
    subprocess.run(SCP+[f'{HOST}:{R}/runs/{s}/out/contribution/.',str(staging)+'/'],check=True,timeout=180)
    if (OUT/s).exists():(OUT/s).rename(OUT/f'.partial_{s}_{time.time_ns()}')
    staging.rename(OUT/s)
   if (REPO/'.research_runtime/CODEX_LOCAL_COMPUTE_DISABLED').exists():continue
   label='range-later/rescore/'+s
   if b['fit_done'] and b.get('refine_report',{}).get('valid') and label not in labels:
    subprocess.run([P,'add','--group','observation-cpu','--label',label,'--',PY,str(REPO/'research/exp/range_followups.py'),'rescore',s],check=True)
   if s.startswith('step38_') and b.get('cbc_score'):
    folder=REPO/'.research_runtime/cbc_scores'/s
    if not (folder/'cbc_score.json').exists():
     folder.mkdir(parents=True,exist_ok=True)
     subprocess.run(SCP+[f'{HOST}:{R}/runs/{s}/out/cbc_score.*',str(folder)+'/'],check=True,timeout=90)
   movement_label='range-audit/movement-later/'+s
   rescored=(REPO/f'.research_runtime/range_monitor/rescore_{s}.done.json').exists()
   early_label='range-audit/early-cbc/'+s
   if s.startswith('step39_early_') and b['contribution_done'] and rescored and early_label not in labels:
    subprocess.run([P,'add','--group','observation-cpu','--label',early_label,'--','bash',str(REPO/'.research_runtime/run_early_score.sh'),s],check=True)
   range_label='range-audit/historical-range/'+s
   if s.startswith('step40_') and b['contribution_done'] and rescored and range_label not in labels:
    subprocess.run([P,'add','--group','observation-cpu','--label',range_label,'--','bash',str(REPO/'.research_runtime/run_historical_range_score.sh'),s],check=True)
   recovery_label='cbc-recovery/'+s
   if s.startswith('step38_') and b['queue_state']=='failed' and b['fit_done'] and rescored and not b['contribution_done'] and recovery_label not in labels:
    subprocess.run([P,'add','--group','cbc-recovery','--label',recovery_label,'--','bash',str(REPO/'.research_runtime/recover_cbc_fit.sh'),s],check=True)
   if s.startswith(tuple(f'step{n}_' for n in range(34,48))) and b['contribution_done'] and rescored and movement_label not in labels and not (Path('/mnt/e/Datasets/houfin/movement_followup_20261101')/s/'READY.json').exists():
    subprocess.run([P,'add','--group','range-movement-cpu','--label',movement_label,'--','bash',str(REPO/'.research_runtime/run_movement_followup.sh'),s],check=True)
   feedback_label='range-audit/crowding-feedback/'+s
   if s.startswith(tuple(f'step{n}_' for n in range(42,48))) and (Path('/mnt/e/Datasets/houfin/movement_followup_20261101')/s/'READY.json').exists() and feedback_label not in labels:
    subprocess.run([P,'add','--group','observation-cpu','--label',feedback_label,'--',PY,str(REPO/'research/exp/score_crowding_feedback.py'),s],check=True)
  (REPO/'research/experiments/LATER_ESTABLISHED_STATE_STATUS_20261101.md').write_text(
    '# Later established-state fits — live status\n\nActual shared-queue jobs; fixed-state sensitivity, not founder-prior inference. Full specifications: LATER_ESTABLISHED_STATE_RUNS_20261101.json.\n\n'+
    '| Fit | Queue | Accepted fit | Contribution/budget complete |\n|---|---|---|---|\n'+
    '\n'.join(f"| {s} | {b['queue_state']} | {bool(b.get('refine_report',{}).get('valid'))} | {b['contribution_done']} |" for s,b in d['runs'].items())+'\n')
  cbc={s:b for s,b in d['runs'].items() if s.startswith(('step37_','step38_'))}
  if cbc:
   (REPO/'research/experiments/CBC_ESTABLISHED_STATUS_20261101.md').write_text(
    '# Later-state plus CBC: live status\n\nPrimary batch has no native-departure process. Queue retains original execution status: the first six accepted fits had a post-fit CPU/GPU diagnostic error; recovered Contribution/CBC columns show current analysis completion. No refitting was required.\n\n'
    '| Fit | Queue | Accepted | Contribution | CBC score |\n|---|---|---|---|---|\n'+
    '\n'.join(f"| {s} | {b['queue_state']} | {bool(b.get('refine_report',{}).get('valid'))} | {b['contribution_done']} | {b.get('cbc_score',{}).get('status','pending')} |" for s,b in cbc.items())+'\n')
  if all(b['queue_state'] in ['done','failed'] for b in d['runs'].values()):break
 except Exception as e:
  (REPO/'.research_runtime/later_state_watch_error.json').write_text(json.dumps({'epoch':time.time(),'error':str(e)}))
 time.sleep(30)
