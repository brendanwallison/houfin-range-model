#!/bin/bash
# One model variant end to end: MAP -> L-BFGS refine -> {visualization, probe}, as
# dependent jobs. Every comparison between variants should be made at the REFINED point:
# a quick90 MAP alone stops thousands of nats short of the optimum (run_18: 199,282 ->
# 191,961 under L-BFGS).
#
#   OVERLAY=config/overlays/map_run19_exchangeable.json bash scripts/tacc/submit_chain.sh
#   OVERLAY=... PROFILE=standard REFINE_MAX_ITER=12000 bash scripts/tacc/submit_chain.sh
#   OVERLAY=... SKIP_MAP=1 bash scripts/tacc/submit_chain.sh     # MAP already fitted
#
# Jobs: 30_model_map (float32) -> 34 as age_refine_map (float64, from the float32 MAP)
# -> 31_model_viz and 34 as age_hmc_probe, both on the refined checkpoint (float64).
set -euo pipefail
source "$(dirname "$0")/env.sh"

OVERLAY="${OVERLAY:?set OVERLAY=config/overlays/<variant>.json}"
PROFILE="${PROFILE:-quick90}"
QUEUE="${QUEUE:-gpu-a100-small}"
REFINE_MAX_ITER="${REFINE_MAX_ITER:-8000}"
SKIP_MAP="${SKIP_MAP:-0}"
A=""
[ -n "${TACC_ALLOCATION:-}" ] && [ "$TACC_ALLOCATION" != "REPLACE_WITH_PROJECT" ] && A="-A $TACC_ALLOCATION"
[ -f "$OVERLAY" ] || { echo "no such overlay: $OVERLAY" >&2; exit 1; }
export AGE_MODEL_CONFIG="$OVERLAY" HOUFIN_MAP_PROFILE="$PROFILE"

# Resolve the MAP and refined directories exactly as the Python stages name them.
# config_utils imports no JAX, so this is safe on a login node.
read -r MAP_DIR REFINED_DIR < <(python - <<'PY'
import os
from src.config_utils import load_age_model_config, load_config
cfg = load_age_model_config()
pcfg = load_config(default_name="age_posterior_config.json", env_var="AGE_POSTERIOR_CONFIG")
name = cfg["run_names"]["map"].format(precision="float32")
prof = os.environ["HOUFIN_MAP_PROFILE"]
if prof != "standard":
    name = f"{name}_{prof}"
refined = pcfg["run_names"]["refine"].format(precision="float64", map_run=name)
print(os.path.join(cfg["results_dir"], name), os.path.join(cfg["results_dir"], refined))
PY
)
echo "overlay $OVERLAY  profile $PROFILE"
echo "  MAP     -> $MAP_DIR"
echo "  refined -> $REFINED_DIR"

submit () {
    local out
    if ! out=$(sbatch "$@" 2>&1); then echo "sbatch rejected the job: $out" >&2; return 1; fi
    echo "$out" | grep -Eo '^[0-9]+' | tail -1
}
dep () { [ -n "$1" ] && echo "--dependency=afterok:$1" || true; }

map=""
if [ "$SKIP_MAP" != "1" ]; then
    map=$(submit $A -p "$QUEUE" -t 02:00:00 --parsable \
          --export=ALL,HOUFIN_MODEL_PRECISION=float32 scripts/tacc/30_model_map.slurm) || exit 1
    echo "MAP      $map"
fi
refine=$(submit $A -p "$QUEUE" -t 02:00:00 --parsable $(dep "$map") \
         --export=ALL,HOUFIN_PROBE_MODULE=src.model.age_refine_map,HOUFIN_MODEL_PRECISION=float64,HOUFIN_HMC_MAP_PRECISION=float32,HOUFIN_REFINE_MAX_ITER=$REFINE_MAX_ITER \
         scripts/tacc/34_model_hmc_probe.slurm) || exit 1
echo "refine   $refine  (log houfin_probe.o$refine)"
viz=$(submit $A -p "$QUEUE" -t 02:00:00 --parsable $(dep "$refine") \
      --export=ALL,HOUFIN_MODEL_PRECISION=float64,HOUFIN_VIZ_RUN_DIR=$REFINED_DIR \
      scripts/tacc/31_model_viz.slurm) || exit 1
echo "viz      $viz  (figures in $REFINED_DIR/map_diagnostics)"
probe=$(submit $A -p "$QUEUE" -t 02:00:00 --parsable $(dep "$refine") \
        --export=ALL,HOUFIN_PROBE_MODULE=src.model.age_hmc_probe,HOUFIN_MODEL_PRECISION=float64,HOUFIN_HMC_MAP_PRECISION=float64,HOUFIN_HMC_MAP_DIR=$REFINED_DIR \
        scripts/tacc/34_model_hmc_probe.slurm) || exit 1
echo "probe    $probe  (log houfin_probe.o$probe)"
