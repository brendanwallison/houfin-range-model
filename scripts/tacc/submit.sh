#!/bin/bash
# Submit the preprocessing pipeline: 02_climate runs only if 01_preprocess
# succeeds (Slurm dependency). Run from the repo root on a login node, AFTER
# download_all.sh has staged the raw data.
#     bash scripts/tacc/submit.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"

A=""
[ -n "${TACC_ALLOCATION:-}" ] && [ "$TACC_ALLOCATION" != "REPLACE_WITH_PROJECT" ] && A="-A $TACC_ALLOCATION"

prep=$(sbatch $A --parsable scripts/tacc/01_preprocess.slurm)
echo "submitted 01_preprocess: $prep"
clim=$(sbatch $A --parsable -d afterok:$prep scripts/tacc/02_climate.slurm)
echo "submitted 02_climate (after $prep): $clim"
echo "watch: squeue -u \$USER ; logs: houfin_prep.o$prep , houfin_climate.o$clim"
