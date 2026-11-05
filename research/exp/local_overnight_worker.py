"""Persistent RTX 3090 float32 MAP lane, with atomic TACC refinement handoff.

Only validated, already registered real-data jobs are eligible. A persistent
token and local checkpoint allow safe restart. No frozen scientific source or
configuration is edited, and this worker never performs float64 refinement.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

REPO = Path('/mnt/c/Dev/houfin-range-model')
STATE = REPO / '.research_runtime/local_overnight'
LOCAL = Path('/mnt/e/Datasets/houfin/grid54')
REMOTE = '/work/07980/bwa386/ls6/houfin'
R = REMOTE + '/research/grid54'
PY = '/home/breallis/houfin/venv/bin/python'
SOCKET = '/home/breallis/.ssh/cm-tacc'
HOST = 'bwa386@ls6.tacc.utexas.edu'
SSH = ['ssh', '-S', SOCKET, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', HOST]
SCP = ['scp', '-o', 'ControlPath=' + SOCKET, '-o', 'BatchMode=yes']
RUNTIME = '/home/breallis/houfin/work/houfin/research/FINAL_LOCAL_JAX_20261025/runtime'


def write(path, data):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def command(args, timeout=120, **kwargs):
    p = subprocess.run(args, capture_output=True, text=True, timeout=timeout, **kwargs)
    if p.returncode:
        raise RuntimeError(f'{args[0]} exited {p.returncode}: {p.stderr[-1500:]} {p.stdout[-500:]}')
    return p.stdout


def remote(action, state, record=None, **kwargs):
    # Send structured arguments to a fixed executable; JSON is not shell quoted.
    argv = [action, '--token', state['token']]
    if record is not None:
        argv += ['--record', json.dumps(record)]
    for key, value in kwargs.items():
        argv += ['--' + key.replace('_', '-'), str(value)]
    script = ('import json,subprocess,sys\n'
              'args=json.loads(sys.stdin.read())\n'
              f'p=subprocess.run([{REMOTE + "/venv/bin/python"!r},{R + "/monitor/local_map_dispatch.py"!r}]+args)\n'
              'raise SystemExit(p.returncode)\n')
    # Use python -c with POSIX quoting only for this fixed bootstrap. All dynamic
    # tokens (including full reservation JSON) are carried on stdin.
    import shlex
    return json.loads(command(SSH + [REMOTE + '/venv/bin/python', '-c', shlex.quote(script)],
                              input=json.dumps(argv), timeout=240))


def event(state, kind, **kwargs):
    d = dict(epoch=time.time(), event=kind, **kwargs)
    with (STATE / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(d) + '\n')
    state['last_event'] = d
    state['heartbeat'] = time.time()
    write(STATE / 'status.json', state)
    print(json.dumps(d), flush=True)


def verify_source(run):
    for line in (run / 'CODE_MANIFEST.sha256').read_text().splitlines():
        expected, rel = line.split('  ', 1)
        path = (run / 'code' / rel).resolve()
        if not path.is_relative_to((run / 'code').resolve()) or digest(path) != expected:
            raise ValueError('local frozen source mismatch')


def prepare(record):
    step = record['step']
    if not re.fullmatch(r'step(?:41|42|43|44|45|46|47)_[a-zA-Z0-9_]+', step):
        raise ValueError('unsupported registered step')
    run = LOCAL / 'runs' / step
    if not (run / 'code').exists():
        stage = LOCAL / 'runs' / ('.local3090_download_' + step)
        stage.mkdir(exist_ok=True)
        for name in ['code', 'CODE_MANIFEST.sha256', 'RUN_SPEC.json']:
            command(SCP + ['-r', f'{HOST}:{R}/runs/{step}/{name}', str(stage) + '/'], timeout=600)
        verify_source(stage)
        if run.exists():
            raise ValueError('partial local source retained; no overwrite')
        stage.rename(run)
    verify_source(run)
    if digest(run / 'CODE_MANIFEST.sha256') != record['manifest_sha256']:
        raise ValueError('reserved and local manifests differ')
    spec = json.loads((run / 'RUN_SPEC.json').read_text())
    if spec['seed'] != record['seed']:
        raise ValueError('registered seed differs')
    (run / 'out').mkdir(exist_ok=True)
    return run


def environment(run, record):
    code = run / 'code'
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(('HOUFIN_REFINE_', 'HOUFIN_HMC_')) or key in (
                'HOUFIN_SYNTHETIC_OBS', 'HOUFIN_MAP_REUSE', 'HOUFIN_MODEL_INPUT_RESIDENCY', 'HOUFIN_MAP_STEPS'):
            env.pop(key, None)
    env.update(PYTHONPATH=f'{RUNTIME}:{code}:{code}/src',
               HOUFIN_DATA='/home/breallis/houfin/scratch/houfin/data',
               HOUFIN_PROCESSED=REMOTE + '/processed',
               AGE_MODEL_CONFIG=str(code / f"config/overlays/grid54_{record['step']}.json"),
               DATA_CONFIG=str(code / 'config/overlays/grid54_data.json'),
               HOUFIN_MAP_SEED=str(record['seed']), HOUFIN_MAP_PROFILE='quick90',
               HOUFIN_MODEL_PRECISION='float32', HOUFIN_Z_FLOAT32='1',
               OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               CUDA_VISIBLE_DEVICES='0', PYTHONUNBUFFERED='1',
               PYTHONFAULTHANDLER='1',
               JAX_PLATFORMS='cuda', XLA_PYTHON_CLIENT_MEM_FRACTION='0.55',
               XLA_FLAGS='--xla_gpu_enable_command_buffer=')
    env['PATH'] = '/home/breallis/houfin/venv/lib/python3.12/site-packages/triton/backends/nvidia/bin:' + env['PATH']
    return env


def execute(state, record, lock):
    previous = remote('inspect', state, record)
    if previous['state'] == 'returned_to_tacc':
        if previous['reason'] != 'validated_local_MAP_ready_for_float64':
            raise RuntimeError('reservation already returned without accepted MAP')
        event(state, 'recovered_completed_handoff', step=record['step'])
        return
    event(state, 'preparing', step=record['step'])
    remote('heartbeat', state, record)
    run = prepare(record)
    env = environment(run, record)
    # Check CUDA availability and canonical data paths before spending the MAP.
    preflight = ('import json,jax,pathlib; from src.config_utils import load_age_model_config; '
                 'c=load_age_model_config(); assert jax.devices()[0].platform=="gpu"; '
                 'assert pathlib.Path(c["input_dir"],"metadata.pkl").is_file(); '
                 'assert pathlib.Path(c["population_model"]["nest_likelihood"]["path"]).is_file(); '
                 'print(json.dumps({"devices":[str(x) for x in jax.devices()],"input_dir":c["input_dir"]}))')
    probe = command([PY, '-c', preflight], env=env, cwd=run / 'code', timeout=180)
    (run / 'out/LOCAL3090_PREFLIGHT.json').write_text(probe)
    limit = min(7200, int(state['deadline_epoch'] - time.time() - 600))
    if limit < 900:
        raise TimeoutError('insufficient time before morning deadline')
    logpath = run / 'out/local3090_map.log'
    event(state, 'map_running', step=record['step'], seed=record['seed'], log=str(logpath))
    with logpath.open('a') as log:
        child = subprocess.Popen(['timeout', '--kill-after=30', str(limit), 'taskset', '-c', '0-7',
                                  PY, '-m', 'src.model.age_run_map'], cwd=run / 'code', env=env,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                 pass_fds=(lock.fileno(),))
        state['child_pid'] = child.pid
        write(STATE / 'status.json', state)
        try:
            last_remote = time.time()
            while child.poll() is None:
                if (STATE / 'STOP').exists() or time.time() >= state['deadline_epoch'] - 480:
                    os.killpg(child.pid, signal.SIGTERM)
                    raise TimeoutError('stop requested or morning deadline')
                if time.time() - last_remote >= 60:
                    try:
                        remote('heartbeat', state, record)
                        last_remote = time.time()
                    except Exception as e:
                        event(state, 'reservation_heartbeat_error', error=str(e), step=record['step'])
                        if time.time() - last_remote > 300:
                            raise RuntimeError('lost reservation heartbeat; stop before remote lease expires')
                state['heartbeat'] = time.time()
                write(STATE / 'status.json', state)
                time.sleep(15)
            if child.returncode:
                raise RuntimeError(f'MAP exit {child.returncode}; see {logpath}')
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=35)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
    mapdir = Path(REMOTE) / 'processed/model_results' / f"age_map_float32_grid54_{record['step']}_quick90"
    # Trusted local checkpoint, NumPy-only reader; acceptance repeats all checks.
    from range_watch import HostUnpickler
    with (mapdir / 'map_checkpoint.pkl').open('rb') as stream:
        ck = HostUnpickler(stream).load()
    import numpy as np
    if ck.get('step') != 400 or not np.isfinite(ck['losses']).all():
        raise ValueError('local MAP not complete and finite')
    cfg = ck['fingerprint_payload']['age_model_config']
    transfer = dict(seed=record['seed'], token=record['token'], device='RTX 3090', precision='float32',
                    code_manifest=digest(run / 'CODE_MANIFEST.sha256'),
                    nests=digest(Path(cfg['population_model']['nest_likelihood']['path'])),
                    checkpoint_sha256=digest(mapdir / 'map_checkpoint.pkl'),
                    params_sha256=digest(mapdir / 'map_params.pkl'))
    write(mapdir / 'local_map_transfer.json', transfer)
    event(state, 'uploading', step=record['step'])
    upload = f"{R}/monitor/local3090_upload_{record['step']}_{state['token']}"
    # Only validated alphanumeric step/token strings enter these fixed paths.
    command(SSH + ['mkdir', '-p', upload])
    command(SCP + ['-r', str(mapdir) + '/.', HOST + ':' + upload + '/'], timeout=600)
    result = remote('accept', state, record)
    write(run / 'out/LOCAL3090_TRANSFERRED.json', result)
    event(state, 'handed_to_tacc_float64', step=record['step'], checkpoint_sha256=transfer['checkpoint_sha256'])


def record_native_failure(state, record, error):
    """Stop launching other scientific cases after repeated native crashes."""
    if not re.search(r'MAP exit (?:-11|-6|139|134)(?:;|$)',str(error)):
        return False
    state['consecutive_native_failures']=state.get('consecutive_native_failures',0)+1
    receipt=dict(epoch=time.time(),step=record['step'],error=str(error),
                 consecutive_native_failures=state['consecutive_native_failures'])
    try:
        kernel=subprocess.run(['dmesg'],capture_output=True,text=True,timeout=10)
        receipt['kernel_tail']=kernel.stdout[-24000:]
        receipt['kernel_read_error']=kernel.stderr[-1000:]
    except Exception as exc:receipt['kernel_read_error']=str(exc)
    write(STATE/f'native_failure_{record["step"]}_{time.time_ns()}.json',receipt)
    if state['consecutive_native_failures']>=2:
        write(STATE/'NATIVE_CRASH_HOLD.json',receipt)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--deadline', default='2026-11-03T10:00:00')
    a = p.parse_args()
    deadline = datetime.fromisoformat(a.deadline).replace(tzinfo=ZoneInfo('America/Chicago')).timestamp()
    STATE.mkdir(parents=True, exist_ok=True)
    with (REPO / '.research_runtime/range_monitor/local_gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = STATE / 'status.json'
        state = json.loads(path.read_text()) if path.exists() else dict(token=uuid.uuid4().hex, completed=[], failed=[], active=None)
        state.update(deadline_epoch=deadline, worker_pid=os.getpid(),minor_flexibility_protocol=True)
        event(state, 'worker_started', deadline=a.deadline + ' America/Chicago')
        stop_reason='deadline_or_STOP'
        while time.time() < deadline - 2400 and not (STATE / 'STOP').exists():
            if (STATE/'NATIVE_CRASH_HOLD.json').exists():
                event(state,'native_crash_hold',reason='Repeated native failures require diagnosis before another model is launched')
                stop_reason='native_crash_hold'
                break
            record = state.get('active')
            try:
                if record is None:
                    record = remote('claim', state, exclude=','.join(state['completed'] + state['failed']),allow_minor='1',allow_transfer='1')
                    if record is None:
                        event(state, 'waiting_for_eligible_pending_work')
                        time.sleep(45)
                        continue
                    state['active'] = record
                    event(state, 'claimed', step=record['step'])
                execute(state, record, lock)
                state['completed'].append(record['step'])
                state['consecutive_native_failures']=0
                state['active'] = None
                state.pop('child_pid', None)
                event(state, 'ready_for_next')
            except Exception as e:
                event(state, 'attempt_error', step=record['step'] if record else None, error=str(e))
                if record:record_native_failure(state,record,e)
                if record:
                    try:
                        info = remote('inspect', state, record)
                        if info['state'] == 'returned_to_tacc':
                            target = 'completed' if info['reason'] == 'validated_local_MAP_ready_for_float64' else 'failed'
                            state[target].append(record['step'])
                            state['active'] = None
                            state.pop('child_pid', None)
                            event(state, 'reconciled_remote_return', step=record['step'], reason=info['reason'])
                            continue
                        # accept can have completed before an SSH response was lost.
                        # Re-accept idempotently first when an upload exists.
                        try:
                            recovered = remote('accept', state, record)
                        except Exception:
                            recovered = remote('release', state, record, reason='local_failure_preserved_logs')
                        state['failed'].append(record['step'])
                        state['active'] = None
                        state.pop('child_pid', None)
                        event(state, 'reservation_released_after_error', step=record['step'], result=recovered)
                    except Exception as release_error:
                        event(state, 'release_pending_retry', error=str(release_error))
                        # Retain ownership and checkpoint; never claim a second task.
                        time.sleep(45)
                        continue
                time.sleep(30)
        event(state, 'worker_stopped', reason=stop_reason, completed=state['completed'])


if __name__ == '__main__':
    main()
