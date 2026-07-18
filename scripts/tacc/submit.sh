#!/bin/bash
# Submit the preprocessing pipeline: 02_climate runs only if 01_preprocess
# succeeds (Slurm dependency). Run from the repo root on a login node, AFTER
# download_all.sh has staged the raw data.
#     bash scripts/tacc/submit.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"

A=""
[ -n "${TACC_ALLOCATION:-}" ] && [ "$TACC_ALLOCATION" != "REPLACE_WITH_PROJECT" ] && A="-A $TACC_ALLOCATION"

# TACC's sbatch wrapper prints a "Welcome to Lonestar6 / Verifying..." banner to
# stdout even with --parsable, which pollutes a naive $(sbatch ...) capture. Grab
# just the job id: the single all-digits line (--parsable emits "<jobid>").
submit () { sbatch "$@" 2>&1 | grep -Eo '^[0-9]+$' | tail -1; }

prep=$(submit $A --parsable scripts/tacc/01_preprocess.slurm)
[ -n "$prep" ] || { echo "01_preprocess submit failed (no job id captured)"; exit 1; }
echo "submitted 01_preprocess: $prep"

clim=$(submit $A --parsable -d afterok:$prep scripts/tacc/02_climate.slurm)
[ -n "$clim" ] || { echo "02_climate submit failed (no job id captured)"; exit 1; }
echo "submitted 02_climate (after $prep): $clim"

echo "watch: squeue -u \$USER ; logs: houfin_prep.o$prep , houfin_climate.o$clim"
