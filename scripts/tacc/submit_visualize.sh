#!/bin/bash
# Submit the visual-QC quicklooks job (03_visualize.slurm). Like the other
# submit_*.sh wrappers, this injects the allocation (-A $TACC_ALLOCATION from
# env.sh) so a direct `sbatch 03_visualize.slurm` doesn't fail on multi-project
# accounts ("You have multiple projects to charge to"). Any extra flags are
# forwarded to quicklook_grids.py via the SLURM script.
# Defaults to the `development` queue. Override QUEUE/TIME; CLI -p/-t override the
# script's #SBATCH.
#     bash scripts/tacc/submit_visualize.sh --climate --climate-levels q10,q50,q90
#     bash scripts/tacc/submit_visualize.sh --all --include-ebird
set -euo pipefail
source "$(dirname "$0")/env.sh"

QUEUE="${QUEUE:-development}"
TIME="${TIME:-01:00:00}"

A=""
[ -n "${TACC_ALLOCATION:-}" ] && [ "$TACC_ALLOCATION" != "REPLACE_WITH_PROJECT" ] && A="-A $TACC_ALLOCATION"

# TACC's sbatch wrapper prints a welcome/verify banner to stdout even with
# --parsable; grab just the numeric job id (see submit_climate.sh).
submit () { sbatch "$@" 2>&1 | grep -Eo '^[0-9]+$' | tail -1; }

viz=$(submit $A -p "$QUEUE" -t "$TIME" --parsable scripts/tacc/03_visualize.slurm "$@")
[ -n "$viz" ] || { echo "03_visualize submit failed (no job id captured)"; exit 1; }
echo "submitted 03_visualize ($QUEUE, $TIME): $viz"
echo "watch: squeue -u \$USER ; log: houfin_viz.o$viz"
echo "output: \$HOUFIN_PROCESSED/quicklooks.tgz (scp this back)"
