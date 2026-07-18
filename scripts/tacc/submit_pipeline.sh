#!/bin/bash
# Submit the whole-pipeline job (00_pipeline.slurm) to the development queue (2h).
# Pass a STAGES subset through the environment; override QUEUE/TIME for a longer run.
#     bash scripts/tacc/submit_pipeline.sh                                  # dev, 2h, all stages
#     STAGES="ebird_cache esk desk cube validate" bash scripts/tacc/submit_pipeline.sh
#     QUEUE=normal TIME=06:00:00 bash scripts/tacc/submit_pipeline.sh       # cold climate etc.
set -euo pipefail
source "$(dirname "$0")/env.sh"

QUEUE="${QUEUE:-development}"
TIME="${TIME:-02:00:00}"
A=""
[ -n "${TACC_ALLOCATION:-}" ] && [ "$TACC_ALLOCATION" != "REPLACE_WITH_PROJECT" ] && A="-A $TACC_ALLOCATION"

# --export=ALL so a STAGES override in this shell reaches the batch job.
submit () { sbatch "$@" 2>&1 | grep -Eo '^[0-9]+$' | tail -1; }
jid=$(submit $A -p "$QUEUE" -t "$TIME" --export=ALL --parsable scripts/tacc/00_pipeline.slurm)
[ -n "$jid" ] || { echo "submit failed (no job id captured)"; exit 1; }
echo "submitted 00_pipeline ($QUEUE, $TIME, STAGES='${STAGES:-<all>}'): $jid"
echo "watch: squeue -u \$USER ; log: houfin_pipeline.o$jid"
