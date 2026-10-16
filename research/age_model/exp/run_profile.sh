#!/bin/bash
# Warm-started profile: for each "name=pins_json" in VARIANTS, refine (L-BFGS, float64) from
# WARM_DIR under BASE_OVERLAY + those pins, then decompose the likelihood change vs WARM_DIR.
# Runs ON a compute node (GPU), from the research worktree. Heavy outputs under $OUT.
#
#   BASE_OVERLAY=config/overlays/map_run23a_weak_allee_juv_ratio.json \
#   WARM_DIR=$HOUFIN_PROCESSED/model_results/refined_float64__age_map_float32_run_23a_new_z_quick90 \
#   OUT=$WORK/houfin/research/age_model/E001 \
#   VARIANTS='f3={"alpha_f":2.9486} f5={"alpha_f":4.9933}' bash research/age_model/exp/run_profile.sh
#
# HOUFIN_HMC_ALLOW_MODEL_DRIFT=1: the research branch adds code (pins) that leaves the unpinned
# model unchanged, but the drift check sees a different age_priors.py than the warm start's MAP.
set -uo pipefail
: "${BASE_OVERLAY:?}" "${WARM_DIR:?}" "${OUT:?}" "${VARIANTS:?}"
mkdir -p "$OUT/overlays"
for v in $VARIANTS; do
    name="${v%%=*}"; pins="${v#*=}"
    ov="$OUT/overlays/$name.json"
    python - "$BASE_OVERLAY" "$pins" "$ov" <<'PY'
import json, sys
base, pins, out = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3]
o = json.load(open(base))
o.setdefault("population_model", {})["pinned_sites"] = {**o["population_model"].get("pinned_sites", {}), **pins}
o["_comment"] = [f"profile variant: {base} + pinned_sites {pins}"]
json.dump(o, open(out, "w"), indent=2)
PY
    echo "=== $name: pins $pins ($(date))"
    t0=$(date +%s)
    env AGE_MODEL_CONFIG="$ov" HOUFIN_MODEL_PRECISION=float64 HOUFIN_HMC_MAP_PRECISION=float64 \
        HOUFIN_HMC_MAP_DIR="$WARM_DIR" HOUFIN_REFINE_TAG="$name" HOUFIN_HMC_ALLOW_MODEL_DRIFT=1 \
        HOUFIN_REFINE_MAX_ITER="${REFINE_MAX_ITER:-4000}" \
        python -m src.model.age_refine_map > "$OUT/$name.refine.log" 2>&1 || { echo "  refine FAILED (see $OUT/$name.refine.log)"; continue; }
    rdir=$(grep -Eo "use HOUFIN_HMC_MAP_DIR=[^ ]+" "$OUT/$name.refine.log" | tail -1 | cut -d= -f2)
    grep -E "\[stop\]|\[done\]" "$OUT/$name.refine.log" | sed 's/^/  /'
    env AGE_MODEL_CONFIG="$BASE_OVERLAY" HOUFIN_MODEL_PRECISION=float64 \
        python research/age_model/exp/loglik_decompose.py --run-dir "$WARM_DIR" \
        --compare "$rdir" --compare-config "$ov" --out "$OUT/$name.decomp" 2>&1 | grep -v -i "warn" | tail -3 | sed 's/^/  /'
    echo "  ($(( ($(date +%s) - t0) / 60 )) min)"
done
