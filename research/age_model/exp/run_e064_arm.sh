#!/bin/bash
# One cold E064 MAP -> float64 L-BFGS refine -> diagnostics -> western-sink audit.
# The queue task sources bin/task_header.sh first, then sets OVERLAY, ARM and OUT_ROOT.
set -euo pipefail
: "${OVERLAY:?}" "${ARM:?}" "${OUT_ROOT:?}"

OUT="$OUT_ROOT/$ARM"
mkdir -p "$OUT"
export AGE_MODEL_CONFIG="$OVERLAY"
export HOUFIN_MODEL_PRECISION=float32
export HOUFIN_MAP_LOG_EVERY=50
# E064 is a diagnostic round, directly comparable to E050--E063: a short
# continuation MAP supplies the basin, then float64 L-BFGS finds the optimum.
# Never silently inherit the 1,800-step production default.
export HOUFIN_MAP_PROFILE="${HOUFIN_MAP_PROFILE:-quick90}"

python - <<'PY' > "$OUT/resolved_config.json"
import json
from src.config_utils import load_age_model_config
print(json.dumps(load_age_model_config(), indent=2))
PY
git rev-parse HEAD > "$OUT/git_commit.txt"

MAP_DIR=$(python - <<'PY'
from src.config_utils import load_age_model_config
import os
c = load_age_model_config()
name = c["run_names"]["map"].format(precision="float32")
profile = os.environ["HOUFIN_MAP_PROFILE"]
if profile != "standard":
    name += "_" + profile
print(c["results_dir"] + "/" + name)
PY
)
echo "[$ARM] cold MAP -> $MAP_DIR"
python -m src.model.age_run_map > "$OUT/map.log" 2>&1

echo "[$ARM] float64 refinement"
export HOUFIN_MODEL_PRECISION=float64
export HOUFIN_HMC_MAP_PRECISION=float32
export HOUFIN_HMC_MAP_DIR="$MAP_DIR"
export HOUFIN_REFINE_MAX_ITER="${HOUFIN_REFINE_MAX_ITER:-8000}"
python -m src.model.age_refine_map > "$OUT/refine.log" 2>&1
REF_DIR=$(grep -Eo 'use HOUFIN_HMC_MAP_DIR=[^ ]+' "$OUT/refine.log" | tail -1 | cut -d= -f2)
test -n "$REF_DIR"
printf '%s\n' "$MAP_DIR" > "$OUT/map_dir.txt"
printf '%s\n' "$REF_DIR" > "$OUT/refine_dir.txt"

echo "[$ARM] diagnostics -> $OUT/viz_ref"
python scripts/viz/map_diagnostics.py --precision float64 --run-dir "$REF_DIR" \
  --out "$OUT/viz_ref" > "$OUT/viz_ref.log" 2>&1
python research/age_model/exp/west_sink_audit.py --viz-dir "$OUT/viz_ref" \
  --out "$OUT/viz_ref/west_sink_audit.json" > "$OUT/west_sink_audit.log" 2>&1
echo "[$ARM] complete: $REF_DIR"
