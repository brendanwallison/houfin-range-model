#!/bin/bash
# Queue entry for E076: B3 candidates refitted on the E070 holdout mask (76a = 69c, 76b = 67c2).
set -euo pipefail
JOB=$(basename "$0" .sh)
REPO=/work/07980/bwa386/ls6/houfin/age-research
source "$REPO/research/age_model/bin/task_header.sh"
CODE=$(printf '%s' "$JOB" | sed -E 's/.*(76[ab]).*/\1/')
[[ "$CODE" =~ ^76[ab]$ ]] || { echo "E076 arm not encoded in task filename: $JOB" >&2; exit 2; }
export OVERLAY="$TWT/config/overlays/map_run${CODE}_b3_holdout.json" ARM="$CODE" OUT_ROOT="$WORK/houfin/research/age_model/E076"
unset HOUFIN_SYNTHETIC_OBS
bash research/age_model/exp/run_e064_arm.sh
