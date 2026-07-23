#!/bin/bash
# Submit the age-model MAP job (30_model_map.slurm) to a GPU queue. Defaults to the
# 2 h dev queue. Because age_run_map checkpoints + resumes, you can either resubmit
# by hand after a wall-clock kill, or set RESUBMITS>0 to auto-chain dependent jobs
# that each resume from the last checkpoint until the fit finishes.
#
#     bash scripts/tacc/submit_map.sh                          # one job, gpu-a100-dev, 2 h
#     RESUBMITS=3 bash scripts/tacc/submit_map.sh              # 4 chained 2 h windows (resume each)
#     HOUFIN_MAP_STEPS=1800 bash scripts/tacc/submit_map.sh    # longer fit (needs >1 window)
#     QUEUE=gpu-a100 TIME=06:00:00 bash scripts/tacc/submit_map.sh   # normal GPU queue instead
set -euo pipefail
source "$(dirname "$0")/env.sh"

QUEUE="${QUEUE:-gpu-a100-dev}"        # 2 h dev queue
TIME="${TIME:-02:00:00}"
RESUBMITS="${RESUBMITS:-0}"           # extra jobs chained after the first (each resumes)
A=""
[ -n "${TACC_ALLOCATION:-}" ] && [ "$TACC_ALLOCATION" != "REPLACE_WITH_PROJECT" ] && A="-A $TACC_ALLOCATION"

submit () { sbatch "$@" 2>&1 | grep -Eo '^[0-9]+$' | tail -1; }

jid=$(submit $A -p "$QUEUE" -t "$TIME" --export=ALL --parsable scripts/tacc/30_model_map.slurm)
[ -n "$jid" ] || { echo "submit failed (no job id captured)"; exit 1; }
echo "submitted 30_model_map ($QUEUE, $TIME): $jid"

# Auto-chain: each subsequent job waits for the prior to end (ANY reason, incl. timeout)
# and resumes from the checkpoint. Gives you N+1 back-to-back 2 h windows unattended.
prev="$jid"
for _ in $(seq 1 "$RESUBMITS"); do
    # A deliberate HOUFIN_MAP_FRESH=1 applies only to the first job. Chained
    # windows must resume the checkpoint that first job creates.
    nxt=$(submit $A -p "$QUEUE" -t "$TIME" --export=ALL,HOUFIN_MAP_FRESH=0 --parsable \
                 --dependency=afterany:"$prev" scripts/tacc/30_model_map.slurm)
    [ -n "$nxt" ] || { echo "chained submit failed"; exit 1; }
    echo "  chained resume job (afterany:$prev): $nxt"
    prev="$nxt"
done

echo "watch: squeue -u \$USER ; log: houfin_map.o$jid ; checkpoint: <results_dir>/<map run>/map_checkpoint.pkl"
