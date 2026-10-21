#!/bin/bash
# Queue entry for E069: 69a no-FFT-floor, 69b smooth-floor B1 bases (cold MAP -> MAP plots -> refine -> plots + audit); arm from the task name.
set -euo pipefail
REPO=/work/07980/bwa386/ls6/houfin/age-research
source "$REPO/research/age_model/bin/task_header.sh"
CODE=$(basename "$0" .sh | sed -E "s/.*(69[ab]).*/\1/")
case "$CODE" in 69a) F=map_run69a_b1_nofloor.json ;; 69b) F=map_run69b_b1_smoothfloor.json ;; *) echo "bad arm $CODE" >&2; exit 2 ;; esac
export OVERLAY="$TWT/config/overlays/$F" ARM=$CODE OUT_ROOT="$WORK/houfin/research/age_model/E069"
unset HOUFIN_SYNTHETIC_OBS
bash research/age_model/exp/run_e064_arm.sh
