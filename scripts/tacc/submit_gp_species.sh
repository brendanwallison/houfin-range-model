#!/bin/bash
# Submit the GP species validation (37_gp_species.slurm).
#
#   bash scripts/tacc/submit_gp_species.sh
#   ESK_DESK_CONFIG=config/overlays/<other>.json bash scripts/tacc/submit_gp_species.sh
#   GPSP_ARGS="--no-baselines --thin none" bash scripts/tacc/submit_gp_species.sh   # slice-1 only
#   GPSP_ARGS="--shape-iters 200" bash scripts/tacc/submit_gp_species.sh         # if not converged
#
# The overlay must point paths.desk_output_dir at a run trained WITH a holdout; the script
# refuses an empty holdout_cells.npy. Output: <desk_output_dir>/gp_species/.
set -euo pipefail
source "$(dirname "$0")/env.sh"

export ESK_DESK_CONFIG="${ESK_DESK_CONFIG:-config/overlays/gp_species_base.json}"
export GPSP_N_BOOT="${GPSP_N_BOOT:-1000}"
QUEUE="${QUEUE:-gpu-a100-small}"
TIME="${TIME:-04:00:00}"
[ -f "$ESK_DESK_CONFIG" ] || { echo "ERROR: overlay not found: $ESK_DESK_CONFIG"; exit 1; }

A=""
[ -n "${TACC_ALLOCATION:-}" ] && [ "$TACC_ALLOCATION" != "REPLACE_WITH_PROJECT" ] && A="-A $TACC_ALLOCATION"
submit () { sbatch "$@" 2>&1 | grep -Eo '^[0-9]+$' | tail -1; }
jid=$(submit $A -p "$QUEUE" -t "$TIME" --export=ALL --parsable scripts/tacc/37_gp_species.slurm)
[ -n "$jid" ] || { echo "submit failed (no job id captured)"; exit 1; }
echo "submitted 37_gp_species ($QUEUE, $TIME, overlay=$ESK_DESK_CONFIG): $jid"
echo "watch: squeue -u \$USER ; log: houfin_gpsp.o$jid"
