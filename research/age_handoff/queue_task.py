"""Atomically dispatch fixed tasks to existing workers; never submit a Slurm allocation."""
import sys
from pathlib import Path

R = Path('/work/07980/bwa386/ls6/houfin/research/final_age_handoff_20261025')
Q = Path('/work/07980/bwa386/ls6/houfin/research/age_model/queue2')
TASKS = {('build', 'v1'): (9943, 'build.sh'), ('build', 'v2'): (9944, 'build.sh'),
         ('fit', 'fd_control64'): (9950, 'run_arm.sh'), ('fit', 'fd_v1_r64'): (9951, 'run_arm.sh'),
         ('fit', 'fd_v2_r128'): (9952, 'run_arm.sh'), ('fit', 'fd_v2_r64'): (9953, 'run_arm.sh')}
key = tuple(sys.argv[1:])
number, script = TASKS[key]
name = f'{number}_final_age_{key[0]}_{key[1]}.sh'
if any((Q / state / name).exists() for state in ['pending', 'running', 'done', 'failed']):
    print('already registered', name)
else:
    temp = Q / 'pending' / (name + '.tmp')
    temp.write_text(f'#!/bin/bash\nset -euo pipefail\nbash {R}/code/research/age_handoff/{script} {key[1]}\n')
    temp.replace(Q / 'pending' / name)
    print('queued', name)
