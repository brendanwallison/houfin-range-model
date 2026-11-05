"""Checkpoint-preserving priority preemption inside the current dev allocation."""
import os,sys,json,time,signal,hashlib,socket
from pathlib import Path
R=Path('/work/07980/bwa386/ls6/houfin/research/grid54');q=R/'queue';step='step45_cell_iid_s020_s41'
sys.path.insert(0,str(R/'monitor'))
from range_watch import HostUnpickler
from rolling_dev_dispatch import session_members
run=R/'runs'/step;task=q/f'running/0609_{step}.sh'
assert task.exists() and task.with_suffix('.owner').read_text().strip()=='3548968'
assert (q/'pending/0609_step45_cell_iid_s010_s41.sh').exists()
checkpoint=R.parents[1]/f'processed/model_results/age_map_float32_grid54_{step}_quick90/map_checkpoint.pkl'
with checkpoint.open('rb') as f:c=HostUnpickler(f).load()
assert 25<=c['step']<400,c['step']
found=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  argv=(p/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')
  if len(argv)>3 and argv[1]==str(R/'monitor/stage_minor_flexibility.py').encode() and argv[-1]==step.encode():found.append(int(p.name))
 except OSError:pass
assert len(found)==1,found
pid=found[0];assert os.getsid(pid)==pid,'Only terminate an isolated rolling-worker task session'
receipt=dict(epoch=time.time(),job='3548968',host=socket.gethostname(),interrupted_step=step,checkpoint_step=c['step'],checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),reason='Prioritize warm independent sigma .01 recovery after diagnostic repair; retain cold sigma .02 checkpoint')
(run/'out/PRIORITY_PREEMPTION_20261103.json').write_text(json.dumps(receipt,indent=2)+'\n')
for victim in session_members(pid):
 try:os.kill(victim,signal.SIGTERM)
 except ProcessLookupError:pass
for _ in range(30):
 failed=q/'failed'/task.name
 if failed.exists():break
 time.sleep(1)
else:raise RuntimeError('Dispatcher has not yet reconciled terminated task; checkpoint remains intact')
failed.rename(q/'pending'/task.name)
receipt['state']='pending';(q/'logs'/f'{task.stem}.dispatch.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('COLD_SENSITIVITY_REQUEUED',step,'SAVED_STEP',c['step'],flush=True)