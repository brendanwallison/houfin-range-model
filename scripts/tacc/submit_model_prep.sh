#!/bin/bash
# Submit canonical post-DESK preparation:
#   bash scripts/tacc/submit_model_prep.sh
# To rerun just one guarded stage:
#   STAGES=path-features bash scripts/tacc/submit_model_prep.sh
#   STAGES=model-ingest  bash scripts/tacc/submit_model_prep.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"

QUEUE="${QUEUE:-gpu-a100}"
TIME="${TIME:-02:00:00}"
A=""
[ -n "${TACC_ALLOCATION:-}" ] && [ "$TACC_ALLOCATION" != "REPLACE_WITH_PROJECT" ] && A="-A $TACC_ALLOCATION"

submit () { sbatch "$@" 2>&1 | grep -Eo '^[0-9]+$' | tail -1; }
jid=$(submit $A -p "$QUEUE" -t "$TIME" --export=ALL --parsable scripts/tacc/25_model_prep.slurm)
[ -n "$jid" ] || { echo "submit failed (no job id captured)"; exit 1; }
echo "submitted 25_model_prep ($QUEUE, $TIME, STAGES='${STAGES:-path-features model-ingest}'): $jid"
echo "watch: squeue -u \$USER ; log: houfin_modelprep.o$jid ; GPU telemetry: gpu_modelprep_${jid}.csv"
