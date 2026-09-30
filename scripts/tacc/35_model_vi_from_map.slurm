#!/bin/bash
#----------------------------------------------------------------------
# Low-rank MVN VI initialized at the MAP (src/model/age_resume_svi_from_map.py), fit
# under nominal priors. Stage 1 of the NeuTra HMC arm. Checkpointed every vi.block
# steps; resumes automatically (HOUFIN_VI_FRESH=1 archives and restarts).
# Output: results_dir/<age_posterior_config run_names.vi>/vi_posterior_params.pkl.
#
#   STAGE=vi RESUBMITS=1 bash scripts/tacc/submit_hmc.sh
#   STAGE=vi HOUFIN_VI_RANK=20 bash scripts/tacc/submit_hmc.sh
#----------------------------------------------------------------------
#SBATCH -J houfin_vi
#SBATCH -o houfin_vi.o%j
#SBATCH -e houfin_vi.e%j
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
gpu_preflight "age-model VI from MAP"
start_gpu_monitor "$SLURM_SUBMIT_DIR/gpu_vi_${SLURM_JOB_ID}.csv" 30
trap stop_gpu_monitor EXIT
date
/usr/bin/time -v python -m src.model.age_resume_svi_from_map
echo "=== 35_model_vi_from_map complete ==="
date
