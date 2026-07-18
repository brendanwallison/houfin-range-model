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
module load remora 2>/dev/null || true        # periodic CPU/mem/IO sampling over time
# R comes from $HOUFIN_RSCRIPT (set in env.sh: the micromamba conda-forge env if
# present, else PATH Rscript). climr is GitHub-only with heavy native deps; see docs/TACC.md.
echo "using Rscript: $HOUFIN_RSCRIPT"

# Monitor: remora samples resource use over time (live, in remora_<jobid>/). The
# parallel climr driver spawns many R chunk processes; remora samples node-wide, so
# it captures them all -- use it to confirm the cores are busy and gauge RAM headroom
# for HOUFIN_CLIMATE_WORKERS. Falls back to /usr/bin/time -v (end-only) without remora.
MON="/usr/bin/time -v"
command -v remora >/dev/null 2>&1 && MON="remora"
echo "resource monitor: $MON"
date

$MON python scripts/climate_climr.py \
    --centroids "$HOUFIN_DATA/elevation/cell_centroids.csv" \
    --out "$HOUFIN_DATA/climate" \
    --rscript "$HOUFIN_RSCRIPT"

python scripts/validate_products.py --stage climate --paths "$HOUFIN_DATA/climate"
echo "=== climate complete -> $HOUFIN_DATA/climate ; manifest in $HOUFIN_PROCESSED/validation/ ==="
date
