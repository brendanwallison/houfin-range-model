#!/bin/bash
# Bounded read-only monitoring on WSL; uses the existing SSH socket, no allocation.
set -uo pipefail
CODE=/mnt/c/Dev/houfin-range-model
OUT="$CODE/research/experiments/final_age_handoff_20261025"
mkdir -p "$OUT"
for tick in $(seq 1 540); do
  if ssh -S /home/breallis/.ssh/cm-tacc -o BatchMode=yes -o ConnectTimeout=10 \
    bwa386@ls6.tacc.utexas.edu python3 - < "$CODE/research/age_handoff/status.py" > "$OUT/status.tmp.json"; then
    /home/breallis/houfin/venv/bin/python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]); s = json.loads((p/'status.tmp.json').read_text())
(p/'status.tmp.json').replace(p/'status.json')
terminal = all(v[0] in ('done','failed') for v in s['tasks'].values())
build_failed = any(s['tasks'][k][0]=='failed' for k in ('9943','9944'))
if terminal or build_failed:
    (p/'MONITOR_TERMINAL.json').write_text(json.dumps(s, indent=2))
PY
    if [ -f "$OUT/MONITOR_TERMINAL.json" ]; then exit 0; fi
  fi
  sleep 60
done
echo 'Monitoring window ended; inspect status.json and queue2.'
exit 2
