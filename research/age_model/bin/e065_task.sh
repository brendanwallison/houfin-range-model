#!/bin/bash
# Queue entry for the preregistered E065 no-survival-sorting correction.
set -euo pipefail
JOB=$(basename "$0" .sh)
CODE=$(printf '%s' "$JOB" | sed -E 's/.*(65[df-jnr]).*/\1/')
case "$CODE" in
  65d) FILE=map_run65d_survival_nosort.json ;;
  65f) FILE=map_run65f_survival_cold_nosort.json ;;
  65g) FILE=map_run65g_survival_snow_nosort.json ;;
  65h) FILE=map_run65h_survival_harsh_nosort.json ;;
  65i) FILE=map_run65i_compensation_both_nosort.json ;;
  65j) FILE=map_run65j_compensation_route_nosort.json ;;
  65n) FILE=map_run65n_compensation_null_nosort.json ;;
  65r) FILE=map_run65r_compensation_recovery_nosort.json ;;
  *) echo "E065 arm not encoded in task filename: $JOB" >&2; exit 2 ;;
esac

REPO=/work/07980/bwa386/ls6/houfin/age-research
source "$REPO/research/age_model/bin/task_header.sh"
export OVERLAY="$TWT/config/overlays/$FILE"
export ARM="$CODE"
export OUT_ROOT="$WORK/houfin/research/age_model/E065"

if [[ "$CODE" == 65n || "$CODE" == 65r ]]; then
  mkdir -p "$OUT_ROOT/$CODE"
  BASE59=$(find "$HOUFIN_PROCESSED/model_results" -maxdepth 1 -type d \
    -name 'refined_float64__age_map_float32_run_59a*' | sort | tail -1)
  test -n "$BASE59"
  SYN="$OUT_ROOT/$CODE/synthetic_observations.npz"
  SETS=(--set subcell_beta_k=0 --set subcell_sigma_s=0 \
        --set subcell_beta_k_mod=0 --set subcell_sigma_s_mod=0)
  if [[ "$CODE" == 65r ]]; then
    SETS=(--set subcell_beta_k=0.5 --set subcell_sigma_s=0.25 \
          --set subcell_beta_k_mod=0.5 --set subcell_sigma_s_mod=0.35)
  fi
  AGE_MODEL_CONFIG="$OVERLAY" HOUFIN_MODEL_PRECISION=float64 \
    python research/age_model/exp/simulate_observations.py \
      --run-dir "$BASE59" --out "$SYN" "${SETS[@]}" --seed 65 \
      > "$OUT_ROOT/$CODE/simulate.log" 2>&1
  export HOUFIN_SYNTHETIC_OBS="$SYN"
fi

bash research/age_model/exp/run_e064_arm.sh
