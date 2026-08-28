#!/bin/bash
# Submit the per-component predictability diagnostic (23_predictability.slurm) to a CPU node.
#
#   bash scripts/tacc/submit_predictability.sh
#   PREDICT_MAX_ROWS=120000 bash scripts/tacc/submit_predictability.sh    # a faster first look
#   PREDICT_PAIRS=6000 TIME=04:00:00 QUEUE=normal bash scripts/tacc/submit_predictability.sh
#
# Runs on a COMPUTE node, not a login node. It needs no GPU, which is exactly why the mistake is
# tempting: the heavy steps are a sustained n*d^2 dgemm (~6,200 columns on the interaction rung)
# and the ESK projection of the sampled communities on CPU, plus ~1 GB resident. That profile is
# what a shared login node's process reaper kills. See the header of 23_predictability.slurm.
set -euo pipefail
source "$(dirname "$0")/env.sh"

export PREDICT_MAX_ROWS="${PREDICT_MAX_ROWS:-300000}"
export PREDICT_PAIRS="${PREDICT_PAIRS:-2000}"
export PREDICT_PCA_DIM="${PREDICT_PCA_DIM:-48}"
export PREDICT_RFF_WIDTH="${PREDICT_RFF_WIDTH:-2048}"
export PREDICT_OUT="${PREDICT_OUT:-$HOUFIN_PROCESSED/encoder/component_predictability.json}"
export PREDICT_THREADS="${PREDICT_THREADS:-}"
QUEUE="${QUEUE:-development}"
TIME="${TIME:-02:00:00}"

# Fail here, before the queue wait, on the three inputs whose absence the script can only report
# after it has already loaded the point set.
STATES="$HOUFIN_PROCESSED/encoder/states/yearly_states"
[ -d "$STATES" ] || { echo "ERROR: no states dir at $STATES (run 04_states first)"; exit 1; }
[ -f "$STATES/state_schema.json" ] || {
    echo "ERROR: $STATES has no state_schema.json -- rebuild states"; exit 1; }
ZDIR=$(python - <<'PY'
import json, os, sys
sys.path.insert(0, os.getcwd())
from src.community_encoder.train_DESK.config_utils import load_config
print(load_config(os.environ.get("ESK_DESK_CONFIG") or None)["desk"]["z_dir"])
PY
)
for f in esk_landmarks.npy esk_projmat.npy meta.json; do
    [ -f "$ZDIR/$f" ] || { echo "ERROR: $ZDIR/$f missing -- run spacetime-esk first"; exit 1; }
done
echo "states: $STATES"
echo "basis:  $ZDIR"
DD=$(python - <<'PY'
import os, sys
sys.path.insert(0, os.getcwd())
from src.community_encoder.train_DESK.config_utils import load_config
print(load_config(os.environ.get("ESK_DESK_CONFIG") or None)["paths"]["desk_output_dir"])
PY
)
if [ -f "$DD/holdout_cells.npy" ]; then
    echo "split:  reusing the trained run's masks in $DD (comparable to its val numbers)"
else
    echo "split:  NO holdout_cells.npy in $DD -- it will be redrawn from config. Same knobs and"
    echo "        seed, but if that run used a different block_cells or seed the R^2 is NOT"
    echo "        comparable to its reported val numbers."
fi
echo "max_rows=$PREDICT_MAX_ROWS pairs=$PREDICT_PAIRS -> $PREDICT_OUT"

A=""
[ -n "${TACC_ALLOCATION:-}" ] && [ "$TACC_ALLOCATION" != "REPLACE_WITH_PROJECT" ] && A="-A $TACC_ALLOCATION"
submit () { sbatch "$@" 2>&1 | grep -Eo '^[0-9]+$' | tail -1; }
jid=$(submit $A -p "$QUEUE" -t "$TIME" --export=ALL --parsable scripts/tacc/23_predictability.slurm)
[ -n "$jid" ] || { echo "submit failed (no job id captured)"; exit 1; }
echo "submitted 23_predictability ($QUEUE, $TIME): $jid"
echo "watch: squeue -u \$USER ; log: houfin_predict.o$jid"
