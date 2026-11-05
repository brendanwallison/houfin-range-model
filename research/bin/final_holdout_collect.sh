#!/bin/bash
set -uo pipefail
ROOT=/home/breallis/houfin/work/houfin/research/FINAL_HOLDOUT_20261025
REMOTE=/work/07980/bwa386/ls6/houfin/research/final_holdout_20261025
PY=/home/breallis/houfin/venv/bin/python
REPORT=/mnt/c/Dev/houfin-range-model/research/experiments/FINAL_HELDOUT_RESULTS.md
for tick in $(seq 1 180); do
  rsync -a --include '*/' --include DONE --include '*.json' --include 'species*.npz' --exclude '*' \
    -e 'ssh -S /home/breallis/.ssh/cm-tacc -o BatchMode=yes -o ConnectTimeout=10' \
    bwa386@ls6.tacc.utexas.edu:"$REMOTE/summer/" "$ROOT/summer/" || true
  scp -q -o ControlPath=/home/breallis/.ssh/cm-tacc -o BatchMode=yes -o ConnectTimeout=10 \
    bwa386@ls6.tacc.utexas.edu:"$REMOTE/sealed_reads.csv" "$ROOT/remote_sealed_reads.csv" || true
  "$PY" "$ROOT/code_reporting/research/exp/final_holdout_summary.py" --root "$ROOT" --report "$REPORT" || exit 1
  if [ "$(find "$ROOT/summer" -name DONE | wc -l)" -ge 26 ] && [ -f "$ROOT/winter/damage_check.json" ]; then
    date -Is > "$ROOT/EVALUATION_DONE"
    exit 0
  fi
  sleep 60
done
echo 'Collection window ended; inspect FINAL_HELDOUT_STATUS.md and worker logs.'
exit 2
