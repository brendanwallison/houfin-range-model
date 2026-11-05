#!/bin/bash
#SBATCH -J desk_shared128
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu-a100-dev
#SBATCH -t 02:00:00
set -euo pipefail
bash /work/07980/bwa386/ls6/houfin/research/final_shared_round1_20261025/final_shared_fit.sh r128
