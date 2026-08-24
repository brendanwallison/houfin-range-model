#!/bin/bash
# Submit the CPU pre-encoder assembly (04_states.slurm: build_states + ebird_cache)
# on a compute node. Like the other submit_*.sh wrappers, injects the allocation
# (-A $TACC_ALLOCATION from env.sh) so it doesn't fail on multi-project accounts.
# Defaults to the `development` queue. Override QUEUE/TIME; set HOUFIN_STATES_WORKERS
# to change the parallel-compression worker count (default 16).
#     bash scripts/tacc/submit_states.sh
#     HOUFIN_STATES_WORKERS=32 QUEUE=normal TIME=02:00:00 bash scripts/tacc/submit_states.sh
#     ESK_DESK_CONFIG=<overlay> STAGES=states bash scripts/tacc/submit_states.sh   # one variant
# For the sweep's per-ema_tau builds use scripts/tacc/submit_tau_states.sh, which reads them
# from the sweep manifest instead of having the overlay path retyped three times.
set -euo pipefail
source "$(dirname "$0")/env.sh"

QUEUE="${QUEUE:-development}"
TIME="${TIME:-01:00:00}"

A=""
[ -n "${TACC_ALLOCATION:-}" ] && [ "$TACC_ALLOCATION" != "REPLACE_WITH_PROJECT" ] && A="-A $TACC_ALLOCATION"

submit () { sbatch "$@" 2>&1 | grep -Eo '^[0-9]+$' | tail -1; }

# --export=ALL, like every other submit_*.sh wrapper. Without it the job does not
# inherit the submitting shell's environment, so ESK_DESK_CONFIG and STAGES are
# dropped -- and build_states then calls load_config() with no overlay, resolving
# paths.hist_dir to the PRODUCTION states dir. A tau-variant build would silently
# rebuild production instead of its own directory. The stale-value class of bug this
# repo already documents for HOUFIN_PROCESSED, one level up.
st=$(submit $A -p "$QUEUE" -t "$TIME" --export=ALL --parsable scripts/tacc/04_states.slurm)
[ -n "$st" ] || { echo "04_states submit failed (no job id captured)"; exit 1; }
echo "submitted 04_states ($QUEUE, $TIME): $st"
echo "watch: squeue -u \$USER ; log: houfin_states.o$st"
