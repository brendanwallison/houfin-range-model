#!/bin/bash
#----------------------------------------------------------------------
# One NUTS chain of the HMC trial, as a SLURM ARRAY task (task id = chain id).
#   HOUFIN_HMC_ARM=map     -> src/model/age_resume_hmc.py (NUTS from MAP;
#                             HOUFIN_HMC_METRIC=diag|laplace_fixed|laplace_adapt)
#   HOUFIN_HMC_ARM=neutra  -> src/model/age_run_hmc.py (NUTS in the VI guide's NeuTra space)
#
# Warmup runs in the first job (it cannot be split); sampling is chunked and each
# chunk checkpointed, so chained resubmits (submit_hmc.sh RESUBMITS=N) continue every
# chain where it stopped and finished chains exit immediately. A chain's
# output: results_dir/<run_names.hmc>/chain_XX/{warmup.pkl,mcmc_state.pkl,chunks/}.
#
#   STAGE=hmc ARM=map METRIC=laplace_fixed CHAINS=2 RESUBMITS=2 bash scripts/tacc/submit_hmc.sh
#----------------------------------------------------------------------
#SBATCH -J houfin_hmc
#SBATCH -o houfin_hmc.o%A_%a
#SBATCH -e houfin_hmc.e%A_%a
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH --mail-type=all
set -euo pipefail

source "$SLURM_SUBMIT_DIR/scripts/tacc/env.sh"
source "$SLURM_SUBMIT_DIR/scripts/tacc/gpu_monitor.sh"
module load python/3.12.11 2>/dev/null || true
export PYTHONPATH="$SLURM_SUBMIT_DIR:$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
export HOUFIN_HMC_CHAIN="${SLURM_ARRAY_TASK_ID:-${HOUFIN_HMC_CHAIN:-0}}"
ARM="${HOUFIN_HMC_ARM:-map}"
case "$ARM" in
    map)    MODULE=src.model.age_resume_hmc ;;
    neutra) MODULE=src.model.age_run_hmc ;;
    *) echo "HOUFIN_HMC_ARM must be map or neutra (got $ARM)"; exit 2 ;;
esac
gpu_preflight "age-model NUTS ($ARM chain $HOUFIN_HMC_CHAIN)"
start_gpu_monitor "$SLURM_SUBMIT_DIR/gpu_hmc_${SLURM_JOB_ID}.csv" 30
trap stop_gpu_monitor EXIT
date
/usr/bin/time -v python -m "$MODULE"
echo "=== 36_model_hmc complete (arm=$ARM metric=${HOUFIN_HMC_METRIC:-default} chain=$HOUFIN_HMC_CHAIN) ==="
date
