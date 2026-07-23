#!/bin/bash
# Submit ecological post-MAP diagnostics. Examples:
#   bash scripts/tacc/submit_map_viz.sh
#   HOUFIN_MAP_PROFILE=quick90 bash scripts/tacc/submit_map_viz.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"

QUEUE="${QUEUE:-gpu-a100-dev}"
TIME="${TIME:-02:00:00}"
PROFILE="${HOUFIN_MAP_PROFILE:-standard}"
A=""
[ -n "${TACC_ALLOCATION:-}" ] && [ "$TACC_ALLOCATION" != "REPLACE_WITH_PROJECT" ] && A="-A $TACC_ALLOCATION"

submit () { sbatch "$@" 2>&1 | grep -Eo '^[0-9]+$' | tail -1; }
jid=$(submit $A -p "$QUEUE" -t "$TIME" --export=ALL,HOUFIN_MAP_PROFILE="$PROFILE" --parsable scripts/tacc/31_model_viz.slurm)
[ -n "$jid" ] || { echo "submit failed (no job id captured)"; exit 1; }
echo "submitted 31_model_viz ($QUEUE, $TIME, profile=$PROFILE): $jid"
echo "watch: squeue -u \$USER ; log: houfin_mapviz.o$jid ; GPU telemetry: gpu_mapviz_${jid}.csv"
