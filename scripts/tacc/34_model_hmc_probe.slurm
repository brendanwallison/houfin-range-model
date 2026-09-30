#!/bin/bash
#----------------------------------------------------------------------
# Posterior probe at the MAP point (src/model/age_hmc_probe.py): gradient wall time,
# |grad| in unconstrained space, dense Hessian + eigen-spectrum, and the floored
# Laplace inverse mass matrix (laplace.npz) that the HMC arms use. Run it FIRST: its
# gradient time sets max_tree_depth and draw counts for everything after it.
#
# The Hessian is checkpointed per column block, so a wall-clock kill resumes it on
# resubmit. Output: results_dir/<age_posterior_config run_names.probe>/.
#
#   STAGE=probe bash scripts/tacc/submit_hmc.sh
#   STAGE=probe HOUFIN_PROBE_HESSIAN=0 bash scripts/tacc/submit_hmc.sh   # timing only
#----------------------------------------------------------------------
#SBATCH -J houfin_probe
#SBATCH -o houfin_probe.o%j
#SBATCH -e houfin_probe.e%j
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
gpu_preflight "age-model posterior probe"
start_gpu_monitor "$SLURM_SUBMIT_DIR/gpu_probe_${SLURM_JOB_ID}.csv" 30
trap stop_gpu_monitor EXIT
date
/usr/bin/time -v python -m src.model.age_hmc_probe
echo "=== 34_model_hmc_probe complete ==="
date
