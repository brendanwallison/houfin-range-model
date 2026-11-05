#!/bin/bash
set -euo pipefail
ROOT=/work/07980/bwa386/ls6/houfin/research/final_age_handoff_20261025
[ -f "$ROOT/clean_v2_m2_BUILD_DONE" ] && exit 0
source /work/07980/bwa386/ls6/houfin/age-research/scripts/tacc/env.sh
cd "$ROOT/code_clean2"
export PYTHONPATH="$PWD:$PWD/src" PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export AGE_MODEL_CONFIG="$PWD/config/overlays/map_build_fd_clean_v2_m2.json"
ENC="$HOUFIN_PROCESSED/encoder"
# Training completion can queue this stage during cube export. Never read a
# partially written cube: its metadata is published only after all years exist.
for tick in $(seq 1 120); do
 if [ -f "$ENC/cube_desk_final_clean_v2_s1/cube_meta.json" ] && [ -f "$ENC/cube_desk_final_clean_v2_s2/cube_meta.json" ]; then break; fi
 sleep 5
done
test -f "$ENC/cube_desk_final_clean_v2_s1/cube_meta.json"
test -f "$ENC/cube_desk_final_clean_v2_s2/cube_meta.json"
python research/exp/final_mean_cube.py --cubes "$ENC/cube_desk_final_clean_v2_s1" "$ENC/cube_desk_final_clean_v2_s2" \
 --out "$ENC/cube_desk_final_clean_v2_mean2_diag" --shared-coordinate-targets
python src/processing/generate_all_path_features.py
python -m src.data.combine.model_inputs
P="$HOUFIN_PROCESSED/model_inputs"
cp -p "$P/numpyro_input_vl_r64/obs_route_ids.npz" "$P/numpyro_input_vl_r64/subcell_strata.npz" "$P/numpyro_input_fd_clean_v2_m2/"
python - <<'PY'
from pathlib import Path
r=Path('/work/07980/bwa386/ls6/houfin/research/final_age_handoff_20261025');q=r.parent/'age_model/queue2';name='9974_clean_v2_mean2_housefinch.sh'
if not any((q/s/name).exists() for s in ['pending','running','done','failed']):
 p=q/'pending'/(name+'.tmp');p.write_text(f'#!/bin/bash\nset -euo pipefail\nbash {r}/diagnostics/final_clean_age_fit.sh\n');p.replace(q/'pending'/name)
PY
date -Is > "$ROOT/clean_v2_m2_BUILD_DONE"
