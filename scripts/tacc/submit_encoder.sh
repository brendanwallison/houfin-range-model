#!/bin/bash
# Submit the encoder job (20_encoder.slurm) to a GPU queue. Run AFTER preprocessing.
# Split ESK/DESK by submitting one STAGES at a time; override QUEUE/TIME as needed.
#     STAGES=esk  bash scripts/tacc/submit_encoder.sh
#     STAGES=desk TIME=04:00:00 bash scripts/tacc/submit_encoder.sh
#     bash scripts/tacc/submit_encoder.sh                     # all four (esk desk cube validate)
#     QUEUE=gpu-h100 bash scripts/tacc/submit_encoder.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"

QUEUE="${QUEUE:-gpu-a100}"
TIME="${TIME:-02:00:00}"
A=""
[ -n "${TACC_ALLOCATION:-}" ] && [ "$TACC_ALLOCATION" != "REPLACE_WITH_PROJECT" ] && A="-A $TACC_ALLOCATION"

submit () { sbatch "$@" 2>&1 | grep -Eo '^[0-9]+$' | tail -1; }
jid=$(submit $A -p "$QUEUE" -t "$TIME" --export=ALL --parsable scripts/tacc/20_encoder.slurm)
[ -n "$jid" ] || { echo "submit failed (no job id captured)"; exit 1; }
echo "submitted 20_encoder ($QUEUE, $TIME, STAGES='${STAGES:-esk desk cube validate}'): $jid"
echo "watch: squeue -u \$USER ; log: houfin_encoder.o$jid"
