#!/bin/bash
#----------------------------------------------------------------------
# Whole off/validate pipeline in one 2-hour development-queue job: preprocess ->
# climate -> climate_grid -> states -> ebird_cache -> bbs -> amplitude -> esk ->
# desk -> cube -> validate. Stage logic lives in pipeline.sh; select a subset via
# the STAGES env var (see that file). Submit with:
#     sbatch scripts/tacc/00_pipeline.slurm
#     STAGES="ebird_cache esk desk cube validate" sbatch --export=ALL scripts/tacc/00_pipeline.slurm
# The split 01/02/03 scripts remain for normal-queue / partial runs.
#----------------------------------------------------------------------
#SBATCH -J houfin_pipeline
#SBATCH -o houfin_pipeline.o%j
#SBATCH -e houfin_pipeline.e%j
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

bash "$SLURM_SUBMIT_DIR/scripts/tacc/pipeline.sh"

echo "=== 00_pipeline complete ==="
date
