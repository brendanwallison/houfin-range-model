#!/bin/bash
#----------------------------------------------------------------------
# Visual-QC quicklooks: render the 25 km rasters + gridded climate covariates to
# thumbnail PNGs and tar them for scp. Fast + parallel, so it targets the
# development queue. Run AFTER preprocess + climate have produced their outputs.
# Extra flags pass through from sbatch, e.g.:
#     sbatch -A <alloc> scripts/tacc/03_visualize.slurm --all --climate-levels q10,q50,q90
#----------------------------------------------------------------------
#SBATCH -J houfin_viz
#SBATCH -o houfin_viz.o%j
#SBATCH -e houfin_viz.e%j
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH --mail-type=all
##SBATCH -A REPLACE_WITH_PROJECT
set -euo pipefail

source "$SLURM_SUBMIT_DIR/scripts/tacc/env.sh"
module load python/3.12.11 2>/dev/null || true
date

# --climate on by default here; forward any extra flags (--all, --climate-levels, ...).
python scripts/viz/quicklook_grids.py --climate "$@"

tar czf "$HOUFIN_PROCESSED/quicklooks.tgz" -C "$HOUFIN_PROCESSED" quicklooks
echo "=== quicklooks -> $HOUFIN_PROCESSED/quicklooks.tgz  (scp this back) ==="
date
