#!/bin/bash
# Queue entry for E073 (E070 stage 2: lambda(t) corrections with detrended process noise). Filename selects
# 73s0/73h0/73s/73h/73sh (real) or the cross-recovery guards 73xh (inject H only) / 73xe (inject a noise pulse only),
# both simulated at 69a's MAP through 73h's structure and fitted with it.
set -euo pipefail
JOB=$(basename "$0" .sh)
REPO=/work/07980/bwa386/ls6/houfin/age-research
source "$REPO/research/age_model/bin/task_header.sh"
CODE=$(printf '%s' "$JOB" | sed -E 's/.*(73(s0|h0|sh|xh|xe|s|h)).*/\1/')
[[ "$CODE" =~ ^73(s0|h0|sh|xh|xe|s|h)$ ]] || { echo "E073 arm not encoded in task filename: $JOB" >&2; exit 2; }
for i in $(seq 1 30); do [ -f "$HOUFIN_PROCESSED/model_inputs/numpyro_input_vl/st_correction_basis.npz" ] && break; sleep 60; done
export HOUFIN_Z_FLOAT32=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95   # split readout carries a second Z copy (E048/E066)
export OVERLAY="$TWT/config/overlays/map_run${CODE}_lambda_t.json" ARM="$CODE" OUT_ROOT="$WORK/houfin/research/age_model/E073"
unset HOUFIN_SYNTHETIC_OBS
if [[ "$CODE" == 73x* ]]; then
  mkdir -p "$OUT_ROOT/$CODE"
  python - "$CODE" "$OUT_ROOT/$CODE" <<'PY'
import sys, numpy as np
code, out = sys.argv[1], sys.argv[2]
w = np.zeros((15, 1)); u = np.zeros((2025 - 1902 + 1, 15))
if code == "73xh":
    w[0, 0] = 0.5                                  # basis 0 = cos(pi x) (i=0, j=1 in generate_spatial_basis order)
else:
    u[1975 - 1902:1985 - 1902 + 1, 0] = 2.5        # the 70r pulse; the fitted structure detrends it identically
np.save(f"{out}/truth_w.npy", w); np.save(f"{out}/truth_u.npy", u)
PY
  AGE_MODEL_CONFIG="$OVERLAY" HOUFIN_MODEL_PRECISION=float64 python research/age_model/exp/simulate_observations.py \
    --run-dir "$HOUFIN_PROCESSED/model_results/age_map_float32_run_69a_quick90" \
    --out "$OUT_ROOT/$CODE/synthetic_observations.npz" --set "st_correction_w=@$OUT_ROOT/$CODE/truth_w.npy" \
    --set "process_noise_u=@$OUT_ROOT/$CODE/truth_u.npy" --seed 73 > "$OUT_ROOT/$CODE/simulate.log" 2>&1
  grep -a "simulate\]" "$OUT_ROOT/$CODE/simulate.log"
  export HOUFIN_SYNTHETIC_OBS="$OUT_ROOT/$CODE/synthetic_observations.npz"
fi
bash research/age_model/exp/run_e064_arm.sh
