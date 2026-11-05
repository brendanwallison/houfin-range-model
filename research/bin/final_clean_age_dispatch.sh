#!/bin/bash
set -euo pipefail
for tick in $(seq 1 600); do
 if ssh -S /home/breallis/.ssh/cm-tacc -o BatchMode=yes -o ConnectTimeout=10 bwa386@ls6.tacc.utexas.edu python3 - <<'PY'
from pathlib import Path
e=Path('/work/07980/bwa386/ls6/houfin/processed/encoder')
if not all((e/f'desk_final_clean_v2_s{s}/resume_checkpoint.pt.completed').exists()
           or (e/f'cube_desk_final_clean_v2_s{s}/cube_meta.json').exists() for s in [1,2]):raise SystemExit(1)
r=Path('/work/07980/bwa386/ls6/houfin/research/final_age_handoff_20261025');q=r.parent/'age_model/queue2';name='9973_clean_v2_mean2_age_inputs.sh'
if not any((q/s/name).exists() for s in ['pending','running','done','failed']):
 p=q/'pending'/(name+'.tmp');p.write_text(f'#!/bin/bash\nset -euo pipefail\nbash {r}/diagnostics/final_clean_age_build.sh\n');p.replace(q/'pending'/name);print('queued',name)
PY
 then exit 0; fi
 sleep 60
done
exit 2
