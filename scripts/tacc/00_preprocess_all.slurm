#!/bin/bash
#----------------------------------------------------------------------
# ALL preprocessing (CPU, torch-free) in one development-queue job:
#   preprocess -> climate -> climate_grid -> states -> ebird_cache -> bbs -> amplitude
# Produces every input the encoder needs (N-stream states, eBird cache, BBS
# community matrix, amplitude-modulated point set). Stage logic lives in
# pipeline.sh; select a subset via STAGES. Submit with:
#     bash scripts/tacc/submit_preprocess_all.sh
#     STAGES="climate_grid states ebird_cache bbs amplitude" sbatch --export=ALL scripts/tacc/00_preprocess_all.slurm
#
# This supersedes running 01_preprocess + 02_climate separately (those stay for
# normal-queue / partial runs). ESK/DESK are a SEPARATE (GPU) job — see
# 20_encoder.slurm. Cold climr is the only stage that can exceed 2h; if its cache
# isn't warm, run STAGES=climate once on `normal` first (QUEUE=normal TIME=06:00:00).
#----------------------------------------------------------------------
#SBATCH -J houfin_preall
#SBATCH -o houfin_preall.o%j
#SBATCH -e houfin_preall.e%j
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH --mail-type=all
##SBATCH -A REPLACE_WITH_PROJECT     # uncomment + set, or rely on env.sh default
set -euo pipefail

source "$SLURM_SUBMIT_DIR/scripts/tacc/env.sh"
module load python/3.12.11 2>/dev/null || true
module load remora 2>/dev/null || true
[ -n "${TACC_ALLOCATION:-}" ] && echo "allocation: $TACC_ALLOCATION"
date

# Default to the preprocessing chain unless the submitter overrode STAGES.
export STAGES="${STAGES:-preprocess climate climate_grid states ebird_cache bbs amplitude}"
bash "$SLURM_SUBMIT_DIR/scripts/tacc/pipeline.sh"

echo "=== 00_preprocess_all complete ==="
date
