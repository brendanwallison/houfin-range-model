#!/bin/bash
# Submit the eigenbasis rescore (22_rescore.slurm) to a GPU node.
#
#   bash scripts/tacc/submit_rescore.sh                       # all stage-1 runs of desk_hp
#   RESCORE_DRAWS=10 bash scripts/tacc/submit_rescore.sh
#   RESCORE_GLOB='.../sweep_t0_f100_mw*' bash scripts/tacc/submit_rescore.sh
#
# Runs on a COMPUTE node, not a login node: ~86 whole-grid forwards per run and ~2 GB resident for
# the z_ema window, times 19 runs, is what a login node's process reaper kills -- which is how the
# first attempt died mid-first-run with a reset connection.
set -euo pipefail
source "$(dirname "$0")/env.sh"

SWEEP_NAME="${SWEEP_NAME:-desk_hp}"
SWEEP_ROOT="${SWEEP_ROOT:-$HOUFIN_PROCESSED/sweeps/$SWEEP_NAME}"
export RESCORE_GLOB="${RESCORE_GLOB:-$SWEEP_ROOT/sweep_t0_f100_*}"
export RESCORE_DRAWS="${RESCORE_DRAWS:-6}"
export RESCORE_BATCH="${RESCORE_BATCH:-1024}"
export RESCORE_OUT="${RESCORE_OUT:-$SWEEP_ROOT/eigenbasis_rescore.json}"
QUEUE="${QUEUE:-gpu-a100-small}"
TIME="${TIME:-01:00:00}"

# Count what will be scored, here, so a mistyped glob fails now rather than after the queue wait.
N=$(ls -d $RESCORE_GLOB 2>/dev/null | wc -l | tr -d ' ')
[ "$N" -gt 0 ] || { echo "ERROR: RESCORE_GLOB matched nothing: $RESCORE_GLOB"; exit 1; }
NM=$(ls -d $RESCORE_GLOB 2>/dev/null | while read -r d; do
        [ -f "$d/desk_meta.npz" ] || echo "$d"; done | wc -l | tr -d ' ')
echo "$N run dir(s) matched; $NM without desk_meta.npz (those are skipped by the script)"
echo "draws=$RESCORE_DRAWS batch=$RESCORE_BATCH -> $RESCORE_OUT"

A=""
[ -n "${TACC_ALLOCATION:-}" ] && [ "$TACC_ALLOCATION" != "REPLACE_WITH_PROJECT" ] && A="-A $TACC_ALLOCATION"
submit () { sbatch "$@" 2>&1 | grep -Eo '^[0-9]+$' | tail -1; }
jid=$(submit $A -p "$QUEUE" -t "$TIME" --export=ALL --parsable scripts/tacc/22_rescore.slurm)
[ -n "$jid" ] || { echo "submit failed (no job id captured)"; exit 1; }
echo "submitted 22_rescore ($QUEUE, $TIME): $jid"
echo "watch: squeue -u \$USER ; log: houfin_rescore.o$jid"
