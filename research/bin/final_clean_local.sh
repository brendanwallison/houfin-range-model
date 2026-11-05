#!/bin/bash
set -euo pipefail
ROOT=/home/breallis/houfin/work/houfin/research/FINAL_CLEAN_20261025
for tick in $(seq 1 120); do
 [ -f "$ROOT/PREPARED" ] && break
 sleep 30
done
test -f "$ROOT/PREPARED"
export FINAL_CLEAN_LOCAL=1
set +e
bash "$ROOT/code/research/bin/final_clean_train.sh" v1 1 > "$ROOT/local_v1_s1.log" 2>&1
result=$?
set -e
if [ "$result" -ne 0 ]; then
 if grep -Eq 'CUDA out of memory|OutOfMemoryError|CUDA driver error: unknown error' "$ROOT/local_v1_s1.log"; then
  # Preserve the recipe and partial local output; use a larger existing GPU.
  for tick in $(seq 1 120); do [ -f "$ROOT/DISPATCHED" ] && break; sleep 30; done
  test -f "$ROOT/DISPATCHED"
  ssh -S /home/breallis/.ssh/cm-tacc bwa386@ls6.tacc.utexas.edu python3 - <<'PY'
from pathlib import Path
r=Path('/work/07980/bwa386/ls6/houfin/research/final_clean_20261025')
q=r.parent/'age_model/queue2'; name='9960_final_clean_v1_s1.sh'
if not any((q/s/name).exists() for s in ('pending','running','done','failed')):
 p=q/'pending'/(name+'.tmp'); p.write_text(f'#!/bin/bash\nset -euo pipefail\nbash {r}/code/research/bin/final_clean_train.sh v1 1\n'); p.replace(q/'pending'/name)
PY
  date -Is > "$ROOT/LOCAL_GPU_FAILURE_REMOTE_FALLBACK"
  exit 0
 fi
 exit "$result"
fi
for d in desk_final_clean_v1_s1 cube_desk_final_clean_v1_s1; do
 rsync -a -e 'ssh -S /home/breallis/.ssh/cm-tacc' \
  "/home/breallis/houfin/work/houfin/processed/encoder/$d/" \
  "bwa386@ls6.tacc.utexas.edu:/work/07980/bwa386/ls6/houfin/processed/encoder/$d/"
done
date -Is > "$ROOT/LOCAL_EXPORTED"
