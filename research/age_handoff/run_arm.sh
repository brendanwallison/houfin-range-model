#!/bin/bash
set -euo pipefail
ARM="$1"
ROOT=/work/07980/bwa386/ls6/houfin/research/final_age_handoff_20261025
source /work/07980/bwa386/ls6/houfin/age-research/scripts/tacc/env.sh
export TWT="$ROOT/code"
cd "$TWT"
export PYTHONPATH="$TWT:$TWT/src" PYTHONUNBUFFERED=1
export HOUFIN_Z_FLOAT32=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 HOUFIN_REFINE_MAX_ITER=16000 HOUFIN_MAP_PROFILE=quick90
unset HOUFIN_MAP_SEED HOUFIN_REFINE_TAG HOUFIN_REFINE_INIT HOUFIN_REFINE_FREEZE HOUFIN_REFINE_FRESH HOUFIN_SYNTHETIC_OBS
export ARM OVERLAY="$TWT/config/overlays/map_run_${ARM}.json" OUT_ROOT="$WORK/houfin/research/age_model/FINAL_DESK_MAP"
bash research/age_model/exp/run_e064_arm.sh
date -Is > "$ROOT/${ARM}_DONE"
