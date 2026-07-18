#!/bin/bash
# Submit ONLY the preprocessing job (01_preprocess.slurm) -- no chained climate.
# Defaults to the 2-hour `development` queue for fast scheduling / a timed test;
# override QUEUE/TIME for a full run. CLI -p/-t override the script's #SBATCH.
#     bash scripts/tacc/submit_preprocess.sh                          # development, 2h
#     QUEUE=normal TIME=06:00:00 bash scripts/tacc/submit_preprocess.sh   # full run
set -euo pipefail
source "$(dirname "$0")/env.sh"

QUEUE="${QUEUE:-development}"
TIME="${TIME:-02:00:00}"

A=""
[ -n "${TACC_ALLOCATION:-}" ] && [ "$TACC_ALLOCATION" != "REPLACE_WITH_PROJECT" ] && A="-A $TACC_ALLOCATION"

# TACC's sbatch wrapper prints a welcome/verify banner to stdout even with
# --parsable; grab just the numeric job id (see submit.sh).
submit () { sbatch "$@" 2>&1 | grep -Eo '^[0-9]+$' | tail -1; }

prep=$(submit $A -p "$QUEUE" -t "$TIME" --parsable scripts/tacc/01_preprocess.slurm)
[ -n "$prep" ] || { echo "01_preprocess submit failed (no job id captured)"; exit 1; }
echo "submitted 01_preprocess ($QUEUE, $TIME): $prep"
echo "watch: squeue -u \$USER ; log: houfin_prep.o$prep"
