#!/bin/bash
# Copy this file into queue2/pending under a name containing 64a..64j, 64n or
# 64r. worker.slurm calls it with bash; the filename selects the preregistered arm.
set -euo pipefail
JOB=$(basename "$0" .sh)
CODE=$(printf '%s' "$JOB" | sed -E 's/.*(64[a-jnr]).*/\1/')
case "$CODE" in
  64a) FILE=map_run64a_subcell_smooth.json ;;
  64b) FILE=map_run64b_subcell_route.json ;;
  64c) FILE=map_run64c_capacity_only.json ;;
  64d) FILE=map_run64d_survival_only.json ;;
  64e) FILE=map_run64e_capacity_deficit.json ;;
  64f) FILE=map_run64f_survival_cold.json ;;
  64g) FILE=map_run64g_survival_snow.json ;;
  64h) FILE=map_run64h_survival_harsh.json ;;
  64i) FILE=map_run64i_compensation_both.json ;;
  64j) FILE=map_run64j_compensation_route.json ;;
  64n) FILE=map_run64n_compensation_null.json ;;
  64r) FILE=map_run64r_compensation_recovery.json ;;
  *) echo "E064 arm not encoded in task filename: $JOB" >&2; exit 2 ;;
esac

REPO=/work/07980/bwa386/ls6/houfin/age-research
source "$REPO/research/age_model/bin/task_header.sh"
export OVERLAY="$TWT/config/overlays/$FILE"
export ARM="$CODE"
export OUT_ROOT="$WORK/houfin/research/age_model/E064"
# Repair tasks named *fresh* deliberately archive an incompatible/invalid
# L-BFGS state and restart refinement from the verified quick MAP.  Ordinary
# retries remain resumable.
if [[ "$JOB" == *fresh* ]]; then
  export HOUFIN_REFINE_FRESH=1
fi

if [[ "$CODE" == 64n || "$CODE" == 64r ]]; then
  mkdir -p "$OUT_ROOT/$CODE"
  BASE59=$(find "$HOUFIN_PROCESSED/model_results" -maxdepth 1 -type d \
    -name 'refined_float64__age_map_float32_run_59a*' | sort | tail -1)
  test -n "$BASE59"
  SYN="$OUT_ROOT/$CODE/synthetic_observations.npz"
  SETS=(--set subcell_beta_k=0 --set subcell_sigma_s=0 \
        --set subcell_beta_k_mod=0 --set subcell_sigma_s_mod=0)
  if [[ "$CODE" == 64r ]]; then
    SETS=(--set subcell_beta_k=0.5 --set subcell_sigma_s=0.25 \
          --set subcell_beta_k_mod=0.5 --set subcell_sigma_s_mod=0.35)
  fi
  AGE_MODEL_CONFIG="$OVERLAY" HOUFIN_MODEL_PRECISION=float64 \
    python research/age_model/exp/simulate_observations.py \
      --run-dir "$BASE59" --out "$SYN" "${SETS[@]}" --seed 64 \
      > "$OUT_ROOT/$CODE/simulate.log" 2>&1
  export HOUFIN_SYNTHETIC_OBS="$SYN"
fi

bash research/age_model/exp/run_e064_arm.sh
