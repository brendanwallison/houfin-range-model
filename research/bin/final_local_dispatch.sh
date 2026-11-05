#!/bin/bash
# Keep GPU waits on a CPU queue; the GPU queue holds only executable work.
set -euo pipefail
ROOT=/home/breallis/houfin/work/houfin/research/FINAL_CLEAN_20261025
PUEUE=/home/breallis/.local/bin/pueue
ready() {
 ssh -S /home/breallis/.ssh/cm-tacc -o BatchMode=yes -o ConnectTimeout=10 bwa386@ls6.tacc.utexas.edu \
  'E=/work/07980/bwa386/ls6/houfin/processed/encoder; for tag in v1_s1 v1_s2 v2_s1 v2_s2; do test -f "$E/cube_desk_final_clean_${tag}/cube_meta.json" || exit 1; done'
}
for tick in $(seq 1 180); do
 [ -f "$ROOT/evaluation/original2/EVALUATION_DONE" ] && break
 sleep 30
done
test -f "$ROOT/evaluation/original2/EVALUATION_DONE"
# If the current process has already begun clean scoring, leave it running.
if [ -d "$ROOT/evaluation/clean2/features" ] || ready; then exit 0; fi
"$PUEUE" kill 758
"$PUEUE" add -g gpu -- bash "$ROOT/calibration_curve.sh"
for tick in $(seq 1 720); do
 if ready; then
  "$PUEUE" add -g gpu -- env FINAL_CLEAN_EVAL_MODES=clean2 bash "$ROOT/evaluate_phase.sh"
  date -Is > "$ROOT/CLEAN_EVAL_REQUEUED"
  exit 0
 fi
 sleep 60
done
exit 2
