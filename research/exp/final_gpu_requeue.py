"""Observe expired tasks; verified native workers exclusively own queue recovery."""
import json
import subprocess
import time
from pathlib import Path

root = Path('/work/07980/bwa386/ls6/houfin/research/final_clean_20261025')
queue = root.parent / 'age_model/queue2'
active = set(subprocess.check_output(['squeue', '-h', '-u', 'bwa386', '-o', '%i'], text=True).split())
state_path = root / 'absent_owners.json'
try:
    previous = json.loads(state_path.read_text()) if state_path.exists() else {}
except json.JSONDecodeError:
    previous = {}  # A reset requires a fresh120second absence observation window.
absent = {}
now = time.time()
numbers = (9957, 9958, 9959, 9960, 9961, 9962, 9963, 9964, 9965, 9966, 9967, 9968, 9970, 9971, 9972, 9973, 9974, 9975)
for number in numbers:
    for script in (queue / 'running').glob(f'{number}_*.sh'):
        owner = script.with_suffix('.owner')
        if not owner.exists():
            continue
        job = owner.read_text().strip()
        if not job.isdigit() or job in active:
            continue
        key = script.name + ':' + job
        first = previous.get(key, now)
        absent[key] = first
        if now - first < 120:
            continue
        # Native worker recover_orphans owns every queue move. A second mover could
        # race a newly claimed task; this process records evidence only.
        archive = root / 'expired_owner_audit'
        archive.mkdir(exist_ok=True)
        evidence = archive / f'{job}_{script.stem}.observed.json'
        if not evidence.exists():
            evidence.write_text(json.dumps({'job': job, 'task': script.name,
                'first_absent': first, 'observed': now,
                'action': 'native worker will recover; monitor does not move queue files'}, indent=2))
            print('observed expired allocation; native recovery owns task', job, script.name, flush=True)
temporary = state_path.with_suffix('.tmp.json')
temporary.write_text(json.dumps(absent, indent=2)); temporary.replace(state_path)
remaining = [str(path) for number in numbers for state in ('pending', 'running')
             for path in (queue / state).glob(f'{number}_*.sh')]
clean_fit_terminal = any((queue / state / '9974_clean_v2_mean2_housefinch.sh').exists() for state in ('done', 'failed'))
status = {'remaining': remaining, 'ready_to_stop': clean_fit_terminal and not remaining}
p = root / 'queue_watch_status.json'; temporary = p.with_suffix('.tmp.json')
temporary.write_text(json.dumps(status, indent=2)); temporary.replace(p)
marker = root / 'QUEUE_WATCH_DONE'
if status['ready_to_stop']:
    marker.touch()
else:
    marker.unlink(missing_ok=True)
