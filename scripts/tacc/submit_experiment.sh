#!/bin/bash
# Submit one model experiment (38_model_experiment.slurm): several overlays run back to
# back in ONE GPU job, each MAP -> refine -> visualization + probe.
#
#   EXPERIMENT_OVERLAYS="config/overlays/map_run24a_pin_allee.json config/overlays/map_run24b_pin_demography.json" \
#     TIME=08:00:00 bash scripts/tacc/submit_experiment.sh
#
# Budget ~1.5-2 h per overlay. gpu-a100-small allows up to 48 h; dev only 2 h (one overlay).
# EXPERIMENT_STAGES (default "map refine viz probe") picks stages; AFTER=<jobid> chains.
set -euo pipefail
source "$(dirname "$0")/env.sh"
: "${EXPERIMENT_OVERLAYS:?set EXPERIMENT_OVERLAYS to one or more overlay paths}"
for o in $EXPERIMENT_OVERLAYS; do [ -f "$o" ] || { echo "no such overlay: $o" >&2; exit 1; }; done
QUEUE="${QUEUE:-gpu-a100-small}"
N=$(echo $EXPERIMENT_OVERLAYS | wc -w | tr -d ' ')
TIME="${TIME:-$(printf '%02d:00:00' $((N * 2 + 1)))}"
A=""
[ -n "${TACC_ALLOCATION:-}" ] && [ "$TACC_ALLOCATION" != "REPLACE_WITH_PROJECT" ] && A="-A $TACC_ALLOCATION"
DEP=()
[ -n "${AFTER:-}" ] && DEP=(--dependency=afterok:"$AFTER")
if ! out=$(sbatch $A -p "$QUEUE" -t "$TIME" ${DEP[@]+"${DEP[@]}"} --parsable --export=ALL scripts/tacc/38_model_experiment.slurm 2>&1); then
    echo "sbatch rejected the job: $out" >&2; exit 1
fi
jid=$(echo "$out" | grep -Eo '^[0-9]+' | tail -1)
echo "submitted experiment ($N overlays, $QUEUE, $TIME): $jid   log: houfin_exp.o$jid"
printf '  %s\n' $EXPERIMENT_OVERLAYS
