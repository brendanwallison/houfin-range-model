#!/bin/bash
# Queue entry for E074 (ideal-free settlement variants on the 70z holdout base). Filename selects 74a-74e.
set -euo pipefail
JOB=$(basename "$0" .sh)
REPO=/work/07980/bwa386/ls6/houfin/age-research
source "$REPO/research/age_model/bin/task_header.sh"
CODE=$(printf '%s' "$JOB" | sed -E 's/.*(74[abcde]).*/\1/')
[[ "$CODE" =~ ^74[abcde]$ ]] || { echo "E074 arm not encoded in task filename: $JOB" >&2; exit 2; }
export OVERLAY="$TWT/config/overlays/map_run${CODE}_settlement.json" ARM="$CODE" OUT_ROOT="$WORK/houfin/research/age_model/E074"
unset HOUFIN_SYNTHETIC_OBS
bash research/age_model/exp/run_e064_arm.sh
