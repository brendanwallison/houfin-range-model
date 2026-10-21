#!/bin/bash
# Queue entry for E067 (smooth spatiotemporal habitat correction). Filename selects: "build_st_basis" or 67a/67c/67n/67r.
set -euo pipefail
JOB=$(basename "$0" .sh)
REPO=/work/07980/bwa386/ls6/houfin/age-research
source "$REPO/research/age_model/bin/task_header.sh"
if [[ "$JOB" == *build_st_basis* ]]; then
  AGE_MODEL_CONFIG="$TWT/config/overlays/map_run62b_numerics.json" python -m src.data.preprocess.st_correction_basis --n-freq 3
  exit 0
fi
CODE=$(printf '%s' "$JOB" | sed -E 's/.*(67[acnr]2?).*/\1/')
[[ "$CODE" =~ ^67[acnr]2?$ ]] || { echo "E067 arm not encoded in task filename: $JOB" >&2; exit 2; }
for i in $(seq 1 240); do [ -f "$HOUFIN_PROCESSED/model_inputs/numpyro_input_vl/st_correction_basis.npz" ] && break; sleep 60; done
export OVERLAY="$TWT/config/overlays/map_run${CODE}_st_correction.json" ARM="$CODE" OUT_ROOT="$WORK/houfin/research/age_model/E067"
unset HOUFIN_SYNTHETIC_OBS
if [[ "$CODE" == 67n* || "$CODE" == 67r* ]]; then
  mkdir -p "$OUT_ROOT/$CODE"
  NB=15; W=$(python -c "import json; w=[[0.0] for _ in range($NB)]; print(json.dumps(w))")
  if [[ "$CODE" == 67r* ]]; then
    # basis index of cos(pi x) (i=0, j=1) in generate_spatial_basis's order (i outer, j inner, (0,0) dropped) = 0
    W=$(python -c "import json; w=[[0.0] for _ in range($NB)]; w[0][0]=0.5; print(json.dumps(w))")
  fi
  AGE_MODEL_CONFIG="$OVERLAY" HOUFIN_MODEL_PRECISION=float64 python research/age_model/exp/simulate_observations.py \
    --run-dir "$HOUFIN_PROCESSED/model_results/age_map_float32_run_62b_quick90" \
    --out "$OUT_ROOT/$CODE/synthetic_observations.npz" --set "st_correction_w=$W" --seed 67 > "$OUT_ROOT/$CODE/simulate.log" 2>&1
  grep -a "simulate\]" "$OUT_ROOT/$CODE/simulate.log"
  export HOUFIN_SYNTHETIC_OBS="$OUT_ROOT/$CODE/synthetic_observations.npz"
fi
bash research/age_model/exp/run_e064_arm.sh
