"""Checkpoint-aware rolling dispatch within an existing Slurm allocation."""
import argparse,fcntl,json,os,signal,subprocess,time
from pathlib import Path
from dispatch_order import dispatch_order
R=Path('/work/07980/bwa386/ls6/houfin/research/grid54')


def eligible(root,job):
    files=[p for n in range(39,49) for p in (root/'queue/pending').glob(f'*_step{n}_*.sh')]
    return sorted((p for p in files if '_dispatcher_refresh_' not in p.name or p.stem.endswith('_'+job)),key=lambda p:dispatch_order(p.name))


def claim(root,job,remaining):
    for p in eligible(root,job):
        step='_'.join(p.stem.split('_')[1:])
        warm=(root/'runs'/step/'out/MAP_DONE').exists()
        minimum=300 if warm else 600
        if remaining<minimum:continue
        target=root/'queue/running'/p.name
        try:p.rename(target)
        except FileNotFoundError:continue
        target.with_suffix('.owner').write_text(job+'\n')
        return target
    return None


def finish(root,task,rc,interrupted):
    state='done' if rc==0 else 'pending' if interrupted else 'failed'
    task.rename(root/'queue'/state/task.name);task.with_suffix('.owner').unlink(missing_ok=True)
    receipt=dict(epoch=time.time(),job=os.environ.get('SLURM_JOB_ID'),exit_code=rc,state=state,
                 reason='allocation deadline; saved checkpoints retained' if interrupted and rc!=0 else 'task exit')
    (root/'queue/logs'/f'{task.stem}.dispatch.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return state


def session_members(sid):
    members=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            tail=(p/'stat').read_text().rsplit(')',1)[1].split()
            if int(tail[3])==sid:members.append(int(p.name))
        except (OSError,ValueError,IndexError):pass
    return members


def terminate_session(proc,grace=20):
    # Nested timeout utilities create process groups; terminate the whole session.
    for pid in session_members(proc.pid):
        try:os.kill(pid,signal.SIGTERM)
        except ProcessLookupError:pass
    try:proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:pass
    for pid in session_members(proc.pid):
        try:os.kill(pid,signal.SIGKILL)
        except ProcessLookupError:pass
    return proc.wait()


def original_alive(pid):
    try:
        p=Path('/proc')/str(pid)
        return (p/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z' and b'/monitor/overflow_dev_worker.py' in (p/'cmdline').read_bytes()
    except (OSError,ValueError,IndexError):return False


def run_lane(root,job,lane,deadline,wait_pid=None):
    if wait_pid:
        print('WAIT_ORIGINAL_LANE',lane,wait_pid,flush=True)
        while original_alive(wait_pid) and time.time()<deadline-120:time.sleep(5)
    os.environ['CUDA_VISIBLE_DEVICES']=str(lane)
    os.environ.update(HOUFIN_MAP_CKPT_EVERY='25',HOUFIN_REFINE_CKPT_EVERY='50',
                      OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    lock=(root/f'monitor/rolling_{job}_lane{lane}.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    print('ROLLING_LANE_READY',lane,'SECONDS_LEFT',int(deadline-time.time()),flush=True)
    while deadline-time.time()>300:
        task=claim(root,job,deadline-time.time())
        if task is None:break
        start=time.time();print('START',task.stem,'LANE',lane,flush=True)
        interrupted=False
        with (root/'queue/logs'/f'{task.stem}.log').open('a') as log:
            proc=subprocess.Popen(['bash',str(task)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while proc.poll() is None:
                if time.time()>=deadline-120:
                    interrupted=True;terminate_session(proc);break
                time.sleep(5)
            rc=proc.wait()
        state=finish(root,task,rc,interrupted)
        print('FINISH',task.stem,state,'EXIT',rc,'SECONDS',round(time.time()-start),flush=True)
        if interrupted:break
    print('ROLLING_LANE_FINISHED',lane,'SECONDS_LEFT',int(deadline-time.time()),flush=True)


def original_workers(job):
    found={}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            cmd=(p/'cmdline').read_bytes()
            argv=cmd.rstrip(b'\0').split(b'\0')
            if len(argv)<2 or argv[1]!=str(R/'monitor/overflow_dev_worker.py').encode():continue
            env=dict(x.split(b'=',1) for x in (p/'environ').read_bytes().split(b'\0') if b'=' in x)
            if env.get(b'SLURM_JOB_ID',b'').decode()!=job:continue
            found[int(env[b'SLURM_LOCALID'])]=int(p.name)
        except (OSError,KeyError,ValueError):pass
    return found


def relay(root,job,lane,deadline):
    lock=(root/f'monitor/rolling_{job}_relay.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old=original_workers(job);print('RELAY_BEGIN',job,'ANCHOR',lane,'ORIGINAL',old,flush=True)
    children=[];logs=[]
    for peer in range(3):
        if peer==lane:continue
        log=(root/f'monitor/rolling_{job}_lane{peer}.log').open('a');logs.append(log)
        argv=['srun','--jobid='+job,'--overlap','--nodes=1','--ntasks=1','--cpus-per-task=8',
              str(root.parents[1]/'venv/bin/python'),str(root/'monitor/rolling_dev_dispatch.py'),
              '--lane',str(peer),'--deadline',str(deadline)]
        if peer in old:argv+=['--wait-pid',str(old[peer])]
        children.append(subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT))
    # This original worker remains occupied until its successor and peers finish,
    # keeping the batch script/allocation alive throughout the handover.
    try:run_lane(root,job,lane,deadline)
    finally:
        for child in children:
            rc=child.wait();print('PEER_EXIT',rc,flush=True)
        for log in logs:log.close()
    print('RELAY_FINISHED',job,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lane',type=int,default=int(os.environ.get('SLURM_LOCALID','0')))
    p.add_argument('--deadline',type=int,default=int(os.environ.get('SLURM_JOB_END_TIME','0')))
    p.add_argument('--wait-pid',type=int);p.add_argument('--relay',action='store_true');a=p.parse_args()
    job=os.environ['SLURM_JOB_ID']
    relay(R,job,a.lane,a.deadline) if a.relay else run_lane(R,job,a.lane,a.deadline,a.wait_pid)
