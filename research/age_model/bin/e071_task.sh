#!/bin/bash
# Queue entry for E071 (Allee sensitivity: 71a n50/4, 71b n50/100) and E072 (route-occupancy zero inflation: 72a real,
# 72n null = no-ZI counts at 69a MAP, 72r recovery = ZI counts with m = .3 at 69a MAP), all on the 70z holdout base.
set -euo pipefail
JOB=$(basename "$0" .sh)
REPO=/work/07980/bwa386/ls6/houfin/age-research
source "$REPO/research/age_model/bin/task_header.sh"
CODE=$(printf '%s' "$JOB" | sed -E 's/.*(7[12][abnr]).*/\1/')
case "$CODE" in
  71a|71b) F="map_run${CODE}_allee_sensitivity.json"; EXP=E071 ;;
  72a|72n|72r) F="map_run${CODE}_route_occupancy.json"; EXP=E072 ;;
  *) echo "arm not encoded in task filename: $JOB" >&2; exit 2 ;;
esac
export OVERLAY="$TWT/config/overlays/$F" ARM="$CODE" OUT_ROOT="$WORK/houfin/research/age_model/$EXP"
unset HOUFIN_SYNTHETIC_OBS
if [[ "$CODE" == 72n || "$CODE" == 72r ]]; then
  mkdir -p "$OUT_ROOT/$CODE"
  if [[ "$CODE" == 72n ]]; then SIM_CFG="$TWT/config/overlays/map_run70z_holdout_base.json"; SET=();
  else SIM_CFG="$OVERLAY"; SET=(--set "route_occupancy_log_m=$(python -c 'import math; print(math.log(0.3))')"); fi
  AGE_MODEL_CONFIG="$SIM_CFG" HOUFIN_MODEL_PRECISION=float64 python research/age_model/exp/simulate_observations.py \
    --run-dir "$HOUFIN_PROCESSED/model_results/age_map_float32_run_69a_quick90" \
    --out "$OUT_ROOT/$CODE/synthetic_observations.npz" "${SET[@]}" --seed 72 > "$OUT_ROOT/$CODE/simulate.log" 2>&1
  grep -a "simulate\]" "$OUT_ROOT/$CODE/simulate.log"
  export HOUFIN_SYNTHETIC_OBS="$OUT_ROOT/$CODE/synthetic_observations.npz"
fi
bash research/age_model/exp/run_e064_arm.sh
