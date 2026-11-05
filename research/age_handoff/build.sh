#!/bin/bash
set -euo pipefail
FAMILY="$1"
ROOT=/work/07980/bwa386/ls6/houfin/research/final_age_handoff_20261025
source /work/07980/bwa386/ls6/houfin/age-research/scripts/tacc/env.sh
CODE="$ROOT/code"
cd "$CODE"
export PYTHONPATH="$CODE:$CODE/src" PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export AGE_MODEL_CONFIG="$CODE/config/overlays/map_build_fd_${FAMILY}.json"
ENC="$HOUFIN_PROCESSED/encoder"
case "$FAMILY" in
 v1) PREFIX=cube_desk_final_s; MEAN=cube_desk_final_mean4_diag; ARMS=fd_v1_r64;;
 v2) PREFIX=cube_desk_final_v2_s; MEAN=cube_desk_final_v2_mean4_diag; ARMS='fd_v2_r128 fd_v2_r64';;
 *) exit 2;;
esac
SEEDS=()
for seed in 1 2 3 4; do SEEDS+=("$ENC/${PREFIX}${seed}"); done
python research/exp/final_mean_cube.py --cubes "${SEEDS[@]}" --out "$ENC/$MEAN" --shared-coordinate-targets
python research/age_handoff/verify_mean.py --family "$FAMILY"
python src/processing/generate_all_path_features.py
python -m src.data.combine.model_inputs
P="$HOUFIN_PROCESSED/model_inputs"
cp -p "$P/numpyro_input_vl_r64/obs_route_ids.npz" "$P/numpyro_input_vl_r64/subcell_strata.npz" "$P/numpyro_input_fd_${FAMILY}/"
for arm in $ARMS; do python research/age_handoff/queue_task.py fit "$arm"; done
date -Is > "$ROOT/build_${FAMILY}_DONE"
