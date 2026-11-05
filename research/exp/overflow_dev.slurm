#!/bin/bash
#SBATCH -J overflow_dev3
#SBATCH -A DEB23008
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 3
#SBATCH --cpus-per-task=32
#SBATCH -t 02:00:00
#SBATCH -o overflow_dev3.o%j
#SBATCH -e overflow_dev3.e%j
set -euo pipefail
ROOT=/work/07980/bwa386/ls6/houfin
R=$ROOT/research/grid54
source "$ROOT/age-research/scripts/tacc/env.sh"
test -f "$R/monitor/overflow_early_VALIDATED.json"
export SLURM_JOB_END_TIME=$(squeue -h -j "$SLURM_JOB_ID" -o %e | xargs -I{} date -d '{}' +%s)
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
srun -n 3 --cpus-per-task=32 "$ROOT/venv/bin/python" "$R/monitor/overflow_dev_worker.py"
