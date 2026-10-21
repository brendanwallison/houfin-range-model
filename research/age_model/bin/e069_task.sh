#!/bin/bash
# Queue entry for E069a: the no-FFT-floor B1 baseline (cold MAP -> MAP plots -> refine -> plots + audit).
set -euo pipefail
REPO=/work/07980/bwa386/ls6/houfin/age-research
source "$REPO/research/age_model/bin/task_header.sh"
export OVERLAY="$TWT/config/overlays/map_run69a_b1_nofloor.json" ARM=69a OUT_ROOT="$WORK/houfin/research/age_model/E069"
unset HOUFIN_SYNTHETIC_OBS
bash research/age_model/exp/run_e064_arm.sh
