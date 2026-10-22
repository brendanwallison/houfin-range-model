#!/bin/bash
# Queue entry for E079 (mate-finding Allee form on the 78r base): 79a aggregated k=1, 79b k=3, 79c relative, 79d off; 79e = 79a at rank 64.
set -euo pipefail
JOB=$(basename "$0" .sh)
REPO=/work/07980/bwa386/ls6/houfin/age-research
source "$REPO/research/age_model/bin/task_header.sh"
CODE=$(printf '%s' "$JOB" | sed -E 's/.*(79[abcde]).*/\1/')
[[ "$CODE" =~ ^79[abcde]$ ]] || { echo "E079 arm not encoded in task filename: $JOB" >&2; exit 2; }
export HOUFIN_Z_FLOAT32=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export OVERLAY="$TWT/config/overlays/map_run${CODE}_allee_form.json" ARM="$CODE" OUT_ROOT="$WORK/houfin/research/age_model/E079"
unset HOUFIN_SYNTHETIC_OBS
bash research/age_model/exp/run_e064_arm.sh
