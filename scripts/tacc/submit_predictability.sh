#!/bin/bash
# Submit the per-component predictability diagnostic (23_predictability.slurm) to a CPU node.
#
#   bash scripts/tacc/submit_predictability.sh
#   PREDICT_MAX_ROWS=120000 bash scripts/tacc/submit_predictability.sh    # a faster first look
#   PREDICT_PAIRS=6000 TIME=04:00:00 QUEUE=normal bash scripts/tacc/submit_predictability.sh
#
# Runs on vm-small, the lightweight non-GPU queue 25_model_prep already uses for numpy/memmap
# work. Not `development`: at a 133x224 grid nothing here can use a 128-core node (the biggest
# BLAS step is ~10 s on 16 cores), and on LS6 `development` is the scarce short-iteration queue.
# Not a login node either -- the peak is ~2 GB, in the ESK projection rather than the BLAS. The
# full accounting is in the header of 23_predictability.slurm.
#
#   QUEUE=development TIME=02:00:00 bash scripts/tacc/submit_predictability.sh   # if VM is tight
set -euo pipefail
source "$(dirname "$0")/env.sh"

export PREDICT_MAX_ROWS="${PREDICT_MAX_ROWS:-300000}"
export PREDICT_PAIRS="${PREDICT_PAIRS:-2000}"
export PREDICT_PCA_DIM="${PREDICT_PCA_DIM:-48}"
export PREDICT_RFF_WIDTH="${PREDICT_RFF_WIDTH:-2048}"
export PREDICT_OUT="${PREDICT_OUT:-$HOUFIN_PROCESSED/encoder/component_predictability.json}"
export PREDICT_THREADS="${PREDICT_THREADS:-}"
QUEUE="${QUEUE:-vm-small}"
TIME="${TIME:-02:00:00}"

# Fail here, before the queue wait, on the inputs whose absence the script can only report after
# it has already loaded the point set.
#
# Every path below is resolved by the SAME loader the script uses, never by an assembled path.
# The first version of this check looked for state_schema.json inside yearly_states/ -- but
# run_states writes the sidecar into hist_dir while the npz files go into hist_dir/yearly_states,
# which is exactly why cio.load_schema searches the dir AND its parent. So it reported a perfectly
# good states tree as broken and told the user to rebuild ~130 years of states. A preflight that
# can be stricter than the thing it gates is worse than no preflight.
#
# env.sh already cd's to $HOUFIN_REPO, but these read through $HOUFIN_REPO explicitly rather than
# through the cwd, so they keep working if that ever stops being true.
_cfgq () { HOUFIN_QUERY="$1" python - <<'PYQ'
import os, sys
sys.path.insert(0, os.environ["HOUFIN_REPO"])
from src.community_encoder.train_DESK.config_utils import load_config
cfg = load_config(os.environ.get("ESK_DESK_CONFIG") or None)
sec, key = os.environ["HOUFIN_QUERY"].split(".", 1)
print(cfg[sec][key])
PYQ
}

_schema_ok () { STATES_DIR="$1" python - <<'PYS'
import os, sys
sys.path.insert(0, os.environ["HOUFIN_REPO"])
from src.community_encoder.train_DESK import covariate_io as cio
try:
    sc = cio.load_schema(os.environ["STATES_DIR"])
except FileNotFoundError as exc:
    print(f"{exc}")
    raise SystemExit(1)
print(f"{int(sc['streams'][-1]['end'])} channels in {len(sc['streams'])} streams ("
      + ", ".join(s["name"] for s in sc["streams"]) + ")")
PYS
}

STATES="$HOUFIN_PROCESSED/encoder/states/yearly_states"
[ -d "$STATES" ] || { echo "ERROR: no states dir at $STATES (run 04_states first)"; exit 1; }
ls "$STATES"/state_*.npz >/dev/null 2>&1 || {
    echo "ERROR: no state_*.npz in $STATES (run 04_states first)"; exit 1; }
SCHEMA=$(_schema_ok "$STATES") || {
    echo "ERROR: $SCHEMA"
    echo "       Searched $STATES and its parent, which is where run_states writes it."
    exit 1; }

ZDIR=$(_cfgq desk.z_dir)
for f in esk_landmarks.npy esk_projmat.npy meta.json; do
    [ -f "$ZDIR/$f" ] || { echo "ERROR: $ZDIR/$f missing -- run spacetime-esk first"; exit 1; }
done
echo "states: $STATES"
echo "schema: $SCHEMA"
echo "basis:  $ZDIR"

DD=$(_cfgq paths.desk_output_dir)
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
