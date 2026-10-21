#!/bin/bash
# Queue entry for E075: retests of approaches rejected under confounded numerics, on the 70z holdout base (no floor).
# 75a occupancy (E063), 75b yearling breeding (E052), 75c/75d split readout rank 32/64 (E066b/c).
set -euo pipefail
JOB=$(basename "$0" .sh)
REPO=/work/07980/bwa386/ls6/houfin/age-research
source "$REPO/research/age_model/bin/task_header.sh"
CODE=$(printf '%s' "$JOB" | sed -E 's/.*(75[abcd]).*/\1/')
[[ "$CODE" =~ ^75[abcd]$ ]] || { echo "E075 arm not encoded in task filename: $JOB" >&2; exit 2; }
export HOUFIN_Z_FLOAT32=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95   # rank-64 cube in a float64 refine (E048)
export OVERLAY="$TWT/config/overlays/map_run${CODE}_retest.json" ARM="$CODE" OUT_ROOT="$WORK/houfin/research/age_model/E075"
unset HOUFIN_SYNTHETIC_OBS
bash research/age_model/exp/run_e064_arm.sh
