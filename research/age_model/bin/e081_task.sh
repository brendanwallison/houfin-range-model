#!/bin/bash
# Queue entry for E081 (age model on the NEW DESK cube, desk-temporal E086w cbc1902).
# "build_cbc1902" in the filename: path features (GPU) + model ingest at width 64 into numpyro_input_cbc1902_r64, copy the
# DESK-independent extras from vl_r64, then ENQUEUE the arms (81a/81b/81c) -- no worker sits polling for the build.
# 81a / 81b / 81c: the arms.
set -euo pipefail
JOB=$(basename "$0" .sh)
REPO=/work/07980/bwa386/ls6/houfin/age-research
source "$REPO/research/age_model/bin/task_header.sh"
Q=$WORK/houfin/research/age_model/queue2
P=$HOUFIN_PROCESSED/model_inputs
if [[ "$JOB" == *build_cbc1902* ]]; then
  export AGE_MODEL_CONFIG="$TWT/config/overlays/map_build_cbc1902_r64.json"
  python -c "from src.config_utils import load_age_model_config as l; c=l(); print('[build]', c['path_features'], c['raw_z_dir'], c['input_dir'], c['latent_dim'])"
  python src/processing/generate_all_path_features.py 2>&1 | tail -5
  python -m src.data.combine.model_inputs 2>&1 | tail -5
  cp -p "$P/numpyro_input_vl_r64/obs_route_ids.npz" "$P/numpyro_input_vl_r64/subcell_strata.npz" "$P/numpyro_input_cbc1902_r64/"
  ls -la "$P/numpyro_input_cbc1902_r64" | head
  for a in a b c; do cp "$TWT/research/age_model/bin/e081_task.sh" "$Q/pending/0329z${a}_81${a}_E081.sh"; done
  chmod +x "$Q"/pending/*.sh; echo "[build] enqueued 81a/81b/81c"
  exit 0
fi
CODE=$(printf '%s' "$JOB" | sed -E 's/.*(81[abc]).*/\1/')
[[ "$CODE" =~ ^81[abc]$ ]] || { echo "E081 arm not encoded in task filename: $JOB" >&2; exit 2; }
export HOUFIN_Z_FLOAT32=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export OVERLAY="$TWT/config/overlays/map_run${CODE}_desk_cbc1902.json" ARM="$CODE" OUT_ROOT="$WORK/houfin/research/age_model/E081"
unset HOUFIN_SYNTHETIC_OBS
bash research/age_model/exp/run_e064_arm.sh
