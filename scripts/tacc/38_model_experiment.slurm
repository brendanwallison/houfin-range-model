#!/bin/bash
#----------------------------------------------------------------------
# A whole model EXPERIMENT in one GPU allocation: for each overlay in
# EXPERIMENT_OVERLAYS (space-separated), run MAP (float32) -> L-BFGS refine (float64)
# -> visualization and probe on the refined point. One queue wait instead of four
# dependent jobs per variant (each variant takes ~1.5 h: MAP ~40 min, refine 10-30,
# viz ~7, probe up to ~1 h).
#
# Each stage is its own process (the model reads its config at import). A failing stage
# is logged and skips to the next variant; nothing else is lost. Resubmitting resumes:
# MAP and refine continue from their checkpoints (finished ones return at once), the
# probe's Hessian resumes, and visualization/probe are skipped if already current.
#
#   EXPERIMENT_OVERLAYS="config/overlays/a.json config/overlays/b.json" \
#     bash scripts/tacc/submit_experiment.sh
#----------------------------------------------------------------------
#SBATCH -J houfin_exp
#SBATCH -o houfin_exp.o%j
#SBATCH -e houfin_exp.e%j
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH --mail-type=all
set -uo pipefail

source "$SLURM_SUBMIT_DIR/scripts/tacc/env.sh"
source "$SLURM_SUBMIT_DIR/scripts/tacc/gpu_monitor.sh"
module load python/3.12.11 2>/dev/null || true
export PYTHONPATH="$SLURM_SUBMIT_DIR:$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
gpu_preflight "age-model experiment" || exit 1
start_gpu_monitor "$SLURM_SUBMIT_DIR/gpu_exp_${SLURM_JOB_ID}.csv" 30
trap stop_gpu_monitor EXIT

PROFILE="${HOUFIN_MAP_PROFILE:-quick90}"
STAGES="${EXPERIMENT_STAGES:-map refine viz probe}"
REFINE_MAX_ITER="${HOUFIN_REFINE_MAX_ITER:-8000}"
has () { case " $STAGES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
summary=()

for O in ${EXPERIMENT_OVERLAYS:?set EXPERIMENT_OVERLAYS}; do
    tag=$(basename "$O" .json)
    echo; echo "================ $tag ($(date)) ================"
    read -r MAP_DIR REFINED < <(AGE_MODEL_CONFIG="$O" HOUFIN_MAP_PROFILE="$PROFILE" python - <<'PY'
import os
from src.config_utils import load_age_model_config, load_config
cfg = load_age_model_config()
pcfg = load_config(default_name="age_posterior_config.json", env_var="AGE_POSTERIOR_CONFIG")
name = cfg["run_names"]["map"].format(precision="float32")
if os.environ["HOUFIN_MAP_PROFILE"] != "standard":
    name = f"{name}_{os.environ['HOUFIN_MAP_PROFILE']}"
print(os.path.join(cfg["results_dir"], name),
      os.path.join(cfg["results_dir"], pcfg["run_names"]["refine"].format(precision="float64", map_run=name)))
PY
)
    echo "MAP -> $MAP_DIR"; echo "refined -> $REFINED"
    common=(AGE_MODEL_CONFIG="$O" HOUFIN_MAP_PROFILE="$PROFILE")
    status="ok"
    if has map; then
        env -u HOUFIN_HMC_MAP_DIR "${common[@]}" HOUFIN_MODEL_PRECISION=float32 HOUFIN_MAP_FRESH=0 \
            /usr/bin/time -v python -m src.model.age_run_map || { status="map FAILED"; summary+=("$tag: $status"); continue; }
    fi
    if has refine; then
        # -u: a HOUFIN_HMC_MAP_DIR exported in the submitting shell would make refine read
        # some other checkpoint as its MAP.
        env -u HOUFIN_HMC_MAP_DIR "${common[@]}" HOUFIN_MODEL_PRECISION=float64 HOUFIN_HMC_MAP_PRECISION=float32 \
            HOUFIN_REFINE_MAX_ITER="$REFINE_MAX_ITER" \
            python -m src.model.age_refine_map || { status="refine FAILED"; summary+=("$tag: $status"); continue; }
    fi
    if has viz; then
        if [ -f "$REFINED/map_diagnostics/metrics.json" ] && [ "$REFINED/map_diagnostics/metrics.json" -nt "$REFINED/map_checkpoint.pkl" ]; then
            echo "[viz] current, skipped"
        else
            env -u HOUFIN_HMC_MAP_DIR "${common[@]}" HOUFIN_MODEL_PRECISION=float64 HOUFIN_VIZ_RUN_DIR="$REFINED" \
                python scripts/viz/map_diagnostics.py --profile "$PROFILE" --precision float64 || status="viz FAILED"
        fi
    fi
    if has probe; then
        env "${common[@]}" HOUFIN_MODEL_PRECISION=float64 HOUFIN_HMC_MAP_PRECISION=float64 \
            HOUFIN_HMC_MAP_DIR="$REFINED" HOUFIN_PROBE_MODULE=src.model.age_hmc_probe \
            python -m src.model.age_hmc_probe || status="$status; probe FAILED"
    fi
    summary+=("$tag: $status")
done

echo; echo "================ experiment summary ($(date)) ================"
printf '  %s\n' "${summary[@]}"
