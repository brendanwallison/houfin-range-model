"""Operational invariants for the rolling Slurm dispatcher; no scientific fits."""
import json,sys,subprocess,os,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research/exp'))
import rolling_dev_dispatch as w


def setup(tmp_path):
 for name in ['pending','running','done','failed','logs']:(tmp_path/'queue'/name).mkdir(parents=True)
 return tmp_path


def test_remaining_time_selects_resumable_work_and_keeps_job_affinity(tmp_path):
 root=setup(tmp_path)
 for name in ['0600_step44_dispatcher_refresh_111.sh','0609_step45_cell_iid_s010_s41.sh','0615_step43_reprovar_floor4_s41.sh']:(root/'queue/pending'/name).write_text('true')
 warm=root/'runs/step43_reprovar_floor4_s41/out';warm.mkdir(parents=True);(warm/'MAP_DONE').touch()
 task=w.claim(root,'222',400)
 assert task.name=='0615_step43_reprovar_floor4_s41.sh'
 assert task.with_suffix('.owner').read_text()=='222\n'
 assert (root/'queue/pending/0600_step44_dispatcher_refresh_111.sh').exists()


def test_deadline_returns_task_to_pending_instead_of_failed(tmp_path):
 root=setup(tmp_path);task=root/'queue/running/0609_step44_kprivate10_s41.sh';task.write_text('true');task.with_suffix('.owner').write_text('222')
 assert w.finish(root,task,-15,True)=='pending'
 assert (root/'queue/pending'/task.name).exists()
 assert not task.with_suffix('.owner').exists()
 assert not list((root/'queue/failed').iterdir())
 assert json.loads((root/'queue/logs'/f'{task.stem}.dispatch.json').read_text())['state']=='pending'


def test_true_failure_is_preserved(tmp_path):
 root=setup(tmp_path);task=root/'queue/running/0609_step44_kprivate10_s41.sh';task.touch()
 assert w.finish(root,task,1,False)=='failed'
 assert (root/'queue/failed'/task.name).exists()


def test_completed_task_at_deadline_is_not_repeated(tmp_path):
 root=setup(tmp_path);task=root/'queue/running/0609_step44_kprivate10_s41.sh';task.touch()
 assert w.finish(root,task,0,True)=='done'
 assert (root/'queue/done'/task.name).exists()


def test_termination_reaches_descendant_process_groups(tmp_path):
 childfile=tmp_path/'child_pid'
 code="import os,subprocess,time; p=subprocess.Popen(['sleep','60'],preexec_fn=os.setpgrp); open("+repr(str(childfile))+",'w').write(str(p.pid)); time.sleep(60)"
 proc=subprocess.Popen([sys.executable,'-c',code],start_new_session=True)
 try:
  for _ in range(100):
   if childfile.exists():break
   time.sleep(.01)
  child=int(childfile.read_text());assert child in w.session_members(proc.pid)
  w.terminate_session(proc,grace=1)
  path=Path('/proc')/str(child)/'stat'
  assert not path.exists() or path.read_text().rsplit(')',1)[1].split()[0]=='Z'
 finally:
  if proc.poll() is None:w.terminate_session(proc,grace=1)
