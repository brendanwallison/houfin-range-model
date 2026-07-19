#!/bin/bash
# Submit the CPU pre-encoder assembly (04_states.slurm: build_states + ebird_cache)
# on a compute node. Like the other submit_*.sh wrappers, injects the allocation
# (-A $TACC_ALLOCATION from env.sh) so it doesn't fail on multi-project accounts.
# Defaults to the `development` queue. Override QUEUE/TIME; set HOUFIN_STATES_WORKERS
# to change the parallel-compression worker count (default 16).
#     bash scripts/tacc/submit_states.sh
#     HOUFIN_STATES_WORKERS=32 QUEUE=normal TIME=02:00:00 bash scripts/tacc/submit_states.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"

QUEUE="${QUEUE:-development}"
TIME="${TIME:-01:00:00}"

A=""
[ -n "${TACC_ALLOCATION:-}" ] && [ "$TACC_ALLOCATION" != "REPLACE_WITH_PROJECT" ] && A="-A $TACC_ALLOCATION"

submit () { sbatch "$@" 2>&1 | grep -Eo '^[0-9]+$' | tail -1; }

st=$(submit $A -p "$QUEUE" -t "$TIME" --parsable scripts/tacc/04_states.slurm)
[ -n "$st" ] || { echo "04_states submit failed (no job id captured)"; exit 1; }
echo "submitted 04_states ($QUEUE, $TIME): $st"
echo "watch: squeue -u \$USER ; log: houfin_states.o$st"
