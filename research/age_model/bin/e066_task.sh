#!/bin/bash
# Queue entry for E066 (DESK readout: split level/deviation x rank, on three bases). The filename selects
# the arm (66a..66l) or the input build ("build_vl_r64"); copy into queue2/pending under such a name.
set -euo pipefail
JOB=$(basename "$0" .sh)
REPO=/work/07980/bwa386/ls6/houfin/age-research
source "$REPO/research/age_model/bin/task_header.sh"
if [[ "$JOB" == *build_vl_r64* ]]; then
  # V&L inputs (adult shape .859) at DESK rank 64; the route ids / strata are cell- and obs-aligned, so copy them.
  AGE_MODEL_CONFIG="$TWT/config/overlays/map_build_vl_r64.json" python -m src.data.combine.model_inputs 2>&1 | tail -3
  P=$HOUFIN_PROCESSED/model_inputs
  cp -p "$P/numpyro_input_vl/obs_route_ids.npz" "$P/numpyro_input_vl/subcell_strata.npz" "$P/numpyro_input_vl_r64/"
  ls "$P/numpyro_input_vl_r64"
  exit 0
fi
CODE=$(printf '%s' "$JOB" | sed -E 's/.*(66[a-l]).*/\1/')
[[ "$CODE" =~ ^66[a-l]$ ]] || { echo "E066 arm not encoded in task filename: $JOB" >&2; exit 2; }
# wait for the rank-64 input build (arms that use it)
for i in $(seq 1 240); do [ -f "$HOUFIN_PROCESSED/model_inputs/numpyro_input_vl_r64/metadata.pkl" ] && break; sleep 60; done
export OVERLAY="$TWT/config/overlays/map_run${CODE}_readout.json"
export ARM="$CODE"
export OUT_ROOT="$WORK/houfin/research/age_model/E066"
# rank 32/64 float64 refines need float32 Z storage (lossless: Z is float32 on disk) to fit in A100 memory (E048)
export HOUFIN_Z_FLOAT32=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
bash research/age_model/exp/run_e064_arm.sh
