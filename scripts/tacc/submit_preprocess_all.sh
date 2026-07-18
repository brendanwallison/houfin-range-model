#!/bin/bash
# Submit the all-preprocessing job (00_preprocess_all.slurm, CPU) to development (2h).
# Pass a STAGES subset through the environment; override QUEUE/TIME for a longer run
# (e.g. cold climr on `normal`).
#     bash scripts/tacc/submit_preprocess_all.sh                                    # dev, 2h, full preprocessing
#     STAGES="climate_grid states ebird_cache bbs amplitude" bash scripts/tacc/submit_preprocess_all.sh
#     QUEUE=normal TIME=06:00:00 STAGES=climate bash scripts/tacc/submit_preprocess_all.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"

QUEUE="${QUEUE:-development}"
TIME="${TIME:-02:00:00}"
A=""
[ -n "${TACC_ALLOCATION:-}" ] && [ "$TACC_ALLOCATION" != "REPLACE_WITH_PROJECT" ] && A="-A $TACC_ALLOCATION"

submit () { sbatch "$@" 2>&1 | grep -Eo '^[0-9]+$' | tail -1; }
jid=$(submit $A -p "$QUEUE" -t "$TIME" --export=ALL --parsable scripts/tacc/00_preprocess_all.slurm)
[ -n "$jid" ] || { echo "submit failed (no job id captured)"; exit 1; }
echo "submitted 00_preprocess_all ($QUEUE, $TIME, STAGES='${STAGES:-<all preprocessing>}'): $jid"
echo "watch: squeue -u \$USER ; log: houfin_preall.o$jid"
