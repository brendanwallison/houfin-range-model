#!/bin/bash
set -uo pipefail
CODE=/mnt/c/Dev/houfin-range-model
LOCAL=/home/breallis/houfin/work/houfin/research/FINAL_SHARED_ROUND1_20261025
REMOTE=/work/07980/bwa386/ls6/houfin/research/final_shared_round1_20261025
OUT="$CODE/research/experiments/final_shared_round1_20261025"
PY=/home/breallis/houfin/venv/bin/python
mkdir -p "$OUT"
for tick in $(seq 1 720); do
 if ssh -S /home/breallis/.ssh/cm-tacc -o BatchMode=yes -o ConnectTimeout=15 bwa386@ls6.tacc.utexas.edu \
  python3 - --root "$REMOTE" --processed /work/07980/bwa386/ls6/houfin/processed --jobs 3534241 3534267 \
  < "$CODE/research/exp/final_shared_status.py" > "$OUT/remote.tmp.json"; then
  mv "$OUT/remote.tmp.json" "$OUT/remote.json"
 fi
 "$PY" "$CODE/research/exp/final_shared_status.py" --root "$LOCAL" --processed /home/breallis/houfin/work/houfin/processed > "$OUT/local.tmp.json" && mv "$OUT/local.tmp.json" "$OUT/local.json"
 # Collect the earlier clean diagnostic separately: its source/model differs.
 rsync -a --include='*/' --include='*.json' --include='*DONE' --exclude='*' \
  -e 'ssh -S /home/breallis/.ssh/cm-tacc -o BatchMode=yes -o ConnectTimeout=15' \
  bwa386@ls6.tacc.utexas.edu:/work/07980/bwa386/ls6/houfin/research/final_age_handoff_20261025/dev_diagnostics_clean2/ \
  "$OUT/earlier_clean_split_diagnostics/" || true
 sleep 60
done
