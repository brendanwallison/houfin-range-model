#!/bin/bash
# Queue entry for E082 (robustness of the leading readouts): 82a 80e seed 7, 82b 80h seed 7, 82c 80h with Allee k = 1.
set -euo pipefail
JOB=$(basename "$0" .sh)
REPO=/work/07980/bwa386/ls6/houfin/age-research
source "$REPO/research/age_model/bin/task_header.sh"
CODE=$(printf '%s' "$JOB" | sed -E 's/.*(82[abc]).*/\1/')
[[ "$CODE" =~ ^82[abc]$ ]] || { echo "E082 arm not encoded in task filename: $JOB" >&2; exit 2; }
export HOUFIN_Z_FLOAT32=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
[[ "$CODE" == 82a || "$CODE" == 82b ]] && export HOUFIN_MAP_SEED=7
export OVERLAY="$TWT/config/overlays/map_run${CODE}_robust.json" ARM="$CODE" OUT_ROOT="$WORK/houfin/research/age_model/E082"
unset HOUFIN_SYNTHETIC_OBS
bash research/age_model/exp/run_e064_arm.sh
