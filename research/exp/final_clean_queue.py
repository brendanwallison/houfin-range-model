"""Queue a fixed two-seed completeness intervention on existing workers only."""
from pathlib import Path
root = Path('/work/07980/bwa386/ls6/houfin/research/final_clean_20261025')
queue = root.parent / 'age_model/queue2'
for number, family, seed in [(9961,'v1',2),(9962,'v2',1),(9963,'v2',2)]:
    name = f'{number}_final_clean_{family}_s{seed}.sh'
    if any((queue / state / name).exists() for state in ('pending','running','done','failed')):
        continue
    temp = queue / 'pending' / (name + '.tmp')
    temp.write_text(f'#!/bin/bash\nset -euo pipefail\nbash {root}/code/research/bin/final_clean_train.sh {family} {seed}\n')
    temp.replace(queue / 'pending' / name)
    print('queued',name)
