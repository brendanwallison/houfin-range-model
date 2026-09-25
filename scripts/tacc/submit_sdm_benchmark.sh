#!/bin/bash
# Submit the correlative-SDM baselines. Examples:
#   bash scripts/tacc/submit_sdm_benchmark.sh
#   HOUFIN_SDM_TIERS=standard bash scripts/tacc/submit_sdm_benchmark.sh
#   HOUFIN_SDM_SPATIAL=1 bash scripts/tacc/submit_sdm_benchmark.sh   # + Test B
#   HOUFIN_SDM_NMIXTURE=0 bash scripts/tacc/submit_sdm_benchmark.sh  # occu only
#
# Needs no MAP run: these baselines read BBS + covariates only.
set -euo pipefail
source "$(dirname "$0")/env.sh"

QUEUE="${QUEUE:-gpu-a100-small}"
TIME="${TIME:-04:00:00}"
TIERS="${HOUFIN_SDM_TIERS:-standard full latent}"
SPATIAL="${HOUFIN_SDM_SPATIAL:-0}"
NMIXTURE="${HOUFIN_SDM_NMIXTURE:-1}"
A=""
[ -n "${TACC_ALLOCATION:-}" ] && [ "$TACC_ALLOCATION" != "REPLACE_WITH_PROJECT" ] && A="-A $TACC_ALLOCATION"

submit () { sbatch "$@" 2>&1 | grep -Eo '^[0-9]+$' | tail -1; }
jid=$(submit $A -p "$QUEUE" -t "$TIME" \
      --export=ALL,HOUFIN_SDM_TIERS="$TIERS",HOUFIN_SDM_SPATIAL="$SPATIAL",HOUFIN_SDM_NMIXTURE="$NMIXTURE" \
      --parsable scripts/tacc/33_sdm_benchmark.slurm)
[ -n "$jid" ] || { echo "submit failed (no job id captured)"; exit 1; }
echo "submitted 33_sdm_benchmark ($QUEUE, $TIME, tiers=[$TIERS], spatial=$SPATIAL, nmixture=$NMIXTURE): $jid"
echo "watch: squeue -u \$USER ; log: houfin_sdm.o$jid"
