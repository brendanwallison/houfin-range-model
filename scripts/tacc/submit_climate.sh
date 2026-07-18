#!/bin/bash
# Submit ONLY the climate job (02_climate.slurm), standalone (no afterok on a
# preprocess job) -- assumes preprocess outputs already exist on disk, in
# particular $HOUFIN_DATA/elevation/cell_centroids.csv and a warmed climr cache.
# Defaults to the 2-hour `development` queue for a fast smoke test; override
# QUEUE/TIME for the full run. CLI -p/-t override the script's #SBATCH.
#     bash scripts/tacc/submit_climate.sh                          # development, 2h
#     QUEUE=normal TIME=12:00:00 bash scripts/tacc/submit_climate.sh   # full run
set -euo pipefail
source "$(dirname "$0")/env.sh"

QUEUE="${QUEUE:-development}"
TIME="${TIME:-02:00:00}"

A=""
[ -n "${TACC_ALLOCATION:-}" ] && [ "$TACC_ALLOCATION" != "REPLACE_WITH_PROJECT" ] && A="-A $TACC_ALLOCATION"

# TACC's sbatch wrapper prints a welcome/verify banner to stdout even with
# --parsable; grab just the numeric job id (see submit.sh).
submit () { sbatch "$@" 2>&1 | grep -Eo '^[0-9]+$' | tail -1; }

clim=$(submit $A -p "$QUEUE" -t "$TIME" --parsable scripts/tacc/02_climate.slurm)
[ -n "$clim" ] || { echo "02_climate submit failed (no job id captured)"; exit 1; }
echo "submitted 02_climate ($QUEUE, $TIME): $clim"
echo "watch: squeue -u \$USER ; log: houfin_climate.o$clim"
