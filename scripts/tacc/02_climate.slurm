#!/bin/bash
#----------------------------------------------------------------------
# Climate downscaling with climr (R). Depends on 01_preprocess (needs
# $HOUFIN_DATA/elevation/cell_centroids.csv) and on a climr reference cache
# already warmed on a login node (compute nodes have no internet). Submit with a
# dependency so it runs after preprocessing:
#     sbatch -d afterok:<prep_jobid> scripts/tacc/02_climate.slurm
#
# climr over ~21k centroids x 3 elevations x (1902..2025) monthly is heavy; this
# is the long/uncertain stage. See docs/TACC.md for the R install, the R-version
# risk (TACC Rstats/4.0.3 vs a mamba newer-R fallback), and masking centroids to
# land to cut the point count.
#----------------------------------------------------------------------
#SBATCH -J houfin_climate
#SBATCH -o houfin_climate.o%j
#SBATCH -e houfin_climate.e%j
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH --mail-type=all
##SBATCH -A REPLACE_WITH_PROJECT
set -euo pipefail

source "$SLURM_SUBMIT_DIR/scripts/tacc/env.sh"
module load python/3.12.11 2>/dev/null || true
module load Rstats/4.0.3 RstatsPackages 2>/dev/null || true   # or a mamba R env; see docs/TACC.md
export R_LIBS_USER="${R_LIBS_USER:-$WORK/houfin/Rlib}"
date

/usr/bin/time -v python scripts/climate_climr.py \
    --centroids "$HOUFIN_DATA/elevation/cell_centroids.csv" \
    --out "$HOUFIN_DATA/climate"

python scripts/validate_products.py --stage climate --paths "$HOUFIN_DATA/climate"
echo "=== climate complete -> $HOUFIN_DATA/climate ; manifest in $HOUFIN_PROCESSED/validation/ ==="
date
