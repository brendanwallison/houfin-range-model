#!/bin/bash
#----------------------------------------------------------------------
# Climate downscaling with climr (R). Depends on 01_preprocess (needs
# $HOUFIN_DATA/elevation/cell_centroids.csv) and on a climr reference cache
# already warmed on a login node (compute nodes have no internet). Submit with a
# dependency so it runs after preprocessing:
#     sbatch -d afterok:<prep_jobid> scripts/tacc/02_climate.slurm
#
# climr over ~21k centroids x 3 elevations x (1902..2025) monthly is heavy; this
# is the long/uncertain stage. See docs/TACC.md for the climr/R install (GitHub-only
# package whose dep tree needs R >= 4.1, so it runs from a micromamba conda-forge
# env, not TACC's Rstats/4.0.3) and masking centroids to land to cut the point count.
#----------------------------------------------------------------------
#SBATCH -J houfin_climate
#SBATCH -o houfin_climate.o%j
#SBATCH -e houfin_climate.e%j
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH --mail-type=all
##SBATCH -A REPLACE_WITH_PROJECT
set -euo pipefail

source "$SLURM_SUBMIT_DIR/scripts/tacc/env.sh"
module load python/3.12.11 2>/dev/null || true
module load remora 2>/dev/null || true
[ -n "${TACC_ALLOCATION:-}" ] && echo "allocation: $TACC_ALLOCATION"
echo "using Rscript: ${HOUFIN_RSCRIPT:-Rscript}"; date

# Thin wrapper over pipeline.sh. Runs the offline climate downscale (reads the warm
# climr cache; climate_climr.py fails fast if the cache is cold) + validation +
# climate_grid rasterization. Each geographic tile is a short-lived R process, so use
# many: HOUFIN_CLIMATE_WORKERS default 16 (of 128 cores; nthread auto = cores/workers),
# bump toward 32 with RAM headroom. This needs a WARM cache -- submit_climate.sh
# refuses on a cold one; a bare sbatch here hits the in-job preflight.
export HOUFIN_CLIMATE_WORKERS="${HOUFIN_CLIMATE_WORKERS:-16}"
export HOUFIN_VALIDATE=1
export STAGES="${STAGES:-climate climate_grid}"
bash "$SLURM_SUBMIT_DIR/scripts/tacc/pipeline.sh"
