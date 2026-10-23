#!/bin/bash
# Queue entry for E080 (how much temporal habitat change: 80a static, 80b DESK-calibrated .14, 80c free; all on 75c, no K trend).
set -euo pipefail
JOB=$(basename "$0" .sh)
REPO=/work/07980/bwa386/ls6/houfin/age-research
source "$REPO/research/age_model/bin/task_header.sh"
CODE=$(printf '%s' "$JOB" | sed -E 's/.*(80[abcdef]).*/\1/')
[[ "$CODE" =~ ^80[abcdef]$ ]] || { echo "E080 arm not encoded in task filename: $JOB" >&2; exit 2; }
export HOUFIN_Z_FLOAT32=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export OVERLAY="$TWT/config/overlays/map_run${CODE}_temporal.json" ARM="$CODE" OUT_ROOT="$WORK/houfin/research/age_model/E080"
unset HOUFIN_SYNTHETIC_OBS
bash research/age_model/exp/run_e064_arm.sh
