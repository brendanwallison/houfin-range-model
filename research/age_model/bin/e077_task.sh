#!/bin/bash
# Queue entry for E077 (native-range departure scaling). 77a real; 77n null (r = 0) / 77r recovery (r = .6), both
# simulated at 69a's MAP through 77a's structure.
set -euo pipefail
JOB=$(basename "$0" .sh)
REPO=/work/07980/bwa386/ls6/houfin/age-research
source "$REPO/research/age_model/bin/task_header.sh"
CODE=$(printf '%s' "$JOB" | sed -E 's/.*(77[anr]).*/\1/')
[[ "$CODE" =~ ^77[anr]$ ]] || { echo "E077 arm not encoded in task filename: $JOB" >&2; exit 2; }
export OVERLAY="$TWT/config/overlays/map_run${CODE}_native_departure.json" ARM="$CODE" OUT_ROOT="$WORK/houfin/research/age_model/E077"
unset HOUFIN_SYNTHETIC_OBS
if [[ "$CODE" == 77n || "$CODE" == 77r ]]; then
  mkdir -p "$OUT_ROOT/$CODE"; R=0.0; [[ "$CODE" == 77r ]] && R=0.6
  AGE_MODEL_CONFIG="$OVERLAY" HOUFIN_MODEL_PRECISION=float64 python research/age_model/exp/simulate_observations.py \
    --run-dir "$HOUFIN_PROCESSED/model_results/age_map_float32_run_69a_quick90" \
    --out "$OUT_ROOT/$CODE/synthetic_observations.npz" --set "native_departure_r=$R" --seed 77 > "$OUT_ROOT/$CODE/simulate.log" 2>&1
  grep -a "simulate\]" "$OUT_ROOT/$CODE/simulate.log"
  export HOUFIN_SYNTHETIC_OBS="$OUT_ROOT/$CODE/synthetic_observations.npz"
fi
bash research/age_model/exp/run_e064_arm.sh
