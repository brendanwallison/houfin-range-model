#!/bin/bash
# Queue entry for E078 (informed combinations + reruns). 78a split r32 + sedentary native range; 78b sedentary + settlement;
# 78c all three; 78s 75c seed replicate (HOUFIN_MAP_SEED 7); 78o occupancy / 78f fitness settlement with the float32-only floor.
set -euo pipefail
JOB=$(basename "$0" .sh)
REPO=/work/07980/bwa386/ls6/houfin/age-research
source "$REPO/research/age_model/bin/task_header.sh"
CODE=$(printf '%s' "$JOB" | sed -E 's/.*(78[abcsofrq]).*/\1/')
[[ "$CODE" =~ ^78[abcsofrq]$ ]] || { echo "E078 arm not encoded in task filename: $JOB" >&2; exit 2; }
export HOUFIN_Z_FLOAT32=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
[[ "$CODE" == 78s ]] && export HOUFIN_MAP_SEED=7
export OVERLAY="$TWT/config/overlays/map_run${CODE}_combo.json" ARM="$CODE" OUT_ROOT="$WORK/houfin/research/age_model/E078"
unset HOUFIN_SYNTHETIC_OBS
bash research/age_model/exp/run_e064_arm.sh
