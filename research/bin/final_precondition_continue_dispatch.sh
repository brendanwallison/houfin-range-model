#!/bin/bash
set -euo pipefail
for tick in $(seq 1 480); do
 set +e
 ssh -S /home/breallis/.ssh/cm-tacc -o BatchMode=yes -o ConnectTimeout=10 bwa386@ls6.tacc.utexas.edu python3 - <<'PY'
from pathlib import Path
import json
r=Path('/work/07980/bwa386/ls6/houfin/research/final_age_handoff_20261025');q=r.parent/'age_model/queue2'
name='9970_preconditioned_control64.sh'
if not any((q/s/name).exists() for s in ['done','failed']):raise SystemExit(1)
reports=list((r.parents[1]/'processed/model_results').glob('refined_float64__age_map_float32_fd_control64*blockscale1/refine_report.json'))
if len(reports)!=1:raise SystemExit(2)
d=json.loads(reports[0].read_text())
if d['valid']:print('Pilot converged; no continuation');raise SystemExit(0)
if d['status']!='not_converged' or not d['reason'].startswith('max_iter'):
 print('No continuation: numerical guard failed',d['reason']);raise SystemExit(2)
if not d.get('coordinate_scale_by_site') or d['U_end']>=d['U_start']-1:
 print('No continuation: insufficient progress');raise SystemExit(2)
name='9975_preconditioned_control64_continue8000.sh'
if not any((q/s/name).exists() for s in ['pending','running','done','failed']):
 p=q/'pending'/(name+'.tmp');p.write_text(f'#!/bin/bash\nset -euo pipefail\nbash {r}/diagnostics/final_preconditioned_continue8000.sh\n');p.replace(q/'pending'/name);print('queued',name)
PY
 result=$?
 set -e
 [ "$result" = 0 ] && exit 0
 [ "$result" = 2 ] && exit 2
 sleep 60
done
exit 2
