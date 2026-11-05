"""Read-only status snapshot; run with Python on TACC, without importing JAX."""
import json
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path('/work/07980/bwa386/ls6/houfin')
RUN = ROOT / 'research/final_age_handoff_20261025'
QUEUE = ROOT / 'research/age_model/queue2'
tasks = {}
owners = {}
try:
    active_jobs = set(subprocess.check_output(['squeue', '-h', '-u', 'bwa386', '-t', 'R,CG', '-o', '%i'], text=True, timeout=15).split())
except (subprocess.SubprocessError, OSError):
    active_jobs = None
for number in (9942, 9943, 9944, 9950, 9951, 9952, 9953, 9954, 9955, 9956, 9957, 9958, 9959,
               9960, 9961, 9962, 9963, 9964, 9965, 9966, 9967, 9968, 9970, 9971, 9972, 9973, 9974, 9975):
    matches = [(state, path.name) for state in ('pending', 'running', 'done', 'failed')
               for path in (QUEUE / state).glob(f'{number}_*.sh')]
    if number == 9975 and not matches:
        continue  # Conditional continuation is unnecessary if the pilot converges.
    tasks[str(number)] = matches[0] if matches else ['not_dispatched', None]
    if matches and matches[0][0] == 'running':
        owner = (QUEUE / 'running' / matches[0][1]).with_suffix('.owner')
        if owner.exists():
            job = owner.read_text().strip()
            owners[str(number)] = job
            if active_jobs is not None and job not in active_jobs:
                tasks[str(number)] = ['awaiting_native_recovery', matches[0][1]]

reports = {}
results = ROOT / 'processed/model_results'
for pattern in ('refined_float64__age_map_float32_fd_*',
                'refined_float64__age_map_float32_run_81a_quick90__finite_restart1'):
    for directory in results.glob(pattern):
        report = directory / 'refine_report.json'
        if report.exists():
            reports[directory.name] = json.loads(report.read_text())

arms = ('fd_control64', 'fd_v1_r64', 'fd_v2_r128', 'fd_v2_r64', 'fd_clean_v2_m2_r128')
arm_status = {}
for arm in arms:
    out = ROOT / 'research/age_model/FINAL_DESK_MAP' / arm
    logs = {}
    for name in ('map.log', 'refine.log', 'viz_ref.log', 'finite_restart1/refine.log',
                 'finite_restart1/viz_ref.log', 'blockscale1/refine.log', 'blockscale1/refine_continue8000.log'):
        log = out / name
        if log.exists():
            # Bound output while retaining enough context for convergence/failure checks.
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size - 3000))
                logs[name] = stream.read().decode(errors='replace').splitlines()[-4:]
    arm_status[arm] = {'complete': (RUN / f'{arm}_DONE').exists(), 'logs': logs}

audits = {path.name: json.loads(path.read_text()) for path in (RUN / 'gradient_audit').glob('*.json')
          if '.tmp.' not in path.name}
print(json.dumps({'timestamp': datetime.now().astimezone().isoformat(),
                  'tasks': tasks, 'arms': arm_status, 'refinement_reports': reports,
                  'task_owners': owners, 'active_worker_jobs': None if active_jobs is None else sorted(active_jobs),
                  'gradient_audits': audits,
                  'builds_done': {f: (RUN / f'build_{f}_DONE').exists() for f in ('v1', 'v2')}},
                 indent=2))
