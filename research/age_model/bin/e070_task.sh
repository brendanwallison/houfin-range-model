#!/bin/bash
# Queue entry for E070 stage 1 (process noise on abundance). Filename selects 70z (holdout baseline), 70a-d, or the
# guards 70n (null) / 70r (recovery): synthetic counts at 69a's MAP through 70b's structure with process_noise_u set.
set -euo pipefail
JOB=$(basename "$0" .sh)
REPO=/work/07980/bwa386/ls6/houfin/age-research
source "$REPO/research/age_model/bin/task_header.sh"
CODE=$(printf '%s' "$JOB" | sed -E 's/.*(70[zabcdnr]).*/\1/')
[[ "$CODE" =~ ^70[zabcdnr]$ ]] || { echo "E070 arm not encoded in task filename: $JOB" >&2; exit 2; }
F="map_run${CODE}_process_noise.json"; [[ "$CODE" == 70z ]] && F="map_run70z_holdout_base.json"
export OVERLAY="$TWT/config/overlays/$F" ARM="$CODE" OUT_ROOT="$WORK/houfin/research/age_model/E070"
unset HOUFIN_SYNTHETIC_OBS
if [[ "$CODE" == 70n || "$CODE" == 70r ]]; then
  mkdir -p "$OUT_ROOT/$CODE"
  # u_raw (time, n_basis): 0 for the null; recovery = +2.5 on basis 0 (cos pi x: i=0, j=1 in generate_spatial_basis's
  # order) over 1975-1985, inside the BBS era so the pulse is directly observed.
  python - "$CODE" "$OUT_ROOT/$CODE/truth_u.npy" <<'PY'
import sys, numpy as np
u = np.zeros((2025 - 1902 + 1, 15))
if sys.argv[1] == "70r":
    u[1975 - 1902:1985 - 1902 + 1, 0] = 2.5
np.save(sys.argv[2], u)
PY
  AGE_MODEL_CONFIG="$OVERLAY" HOUFIN_MODEL_PRECISION=float64 python research/age_model/exp/simulate_observations.py \
    --run-dir "$HOUFIN_PROCESSED/model_results/age_map_float32_run_69a_quick90" \
    --out "$OUT_ROOT/$CODE/synthetic_observations.npz" --set "process_noise_u=@$OUT_ROOT/$CODE/truth_u.npy" --seed 70 \
    > "$OUT_ROOT/$CODE/simulate.log" 2>&1
  grep -a "simulate\]" "$OUT_ROOT/$CODE/simulate.log"
  export HOUFIN_SYNTHETIC_OBS="$OUT_ROOT/$CODE/synthetic_observations.npz"
fi
bash research/age_model/exp/run_e064_arm.sh
