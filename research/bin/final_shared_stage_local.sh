#!/bin/bash
set -euo pipefail
ROOT=/home/breallis/houfin/work/houfin/research/FINAL_SHARED_ROUND1_20261025
CODE=/mnt/c/Dev/houfin-range-model
STORAGE="$CODE/.research_runtime/shared_round1_inputs"
REMOTE=bwa386@ls6.tacc.utexas.edu
SSH='ssh -S /home/breallis/.ssh/cm-tacc -o BatchMode=yes -o ConnectTimeout=15'
P=/work/07980/bwa386/ls6/houfin/processed
LOCAL=/home/breallis/houfin/work/houfin/processed/model_inputs
NAME=numpyro_input_fd_clean_v2_m2
mkdir -p "$ROOT" "$STORAGE/$NAME" "$LOCAL"
if [ ! -f "$ROOT/INPUTS_STAGED" ]; then
 $SSH "$REMOTE" "cd '$P/model_inputs/$NAME' && sha256sum *.dat *.pkl *.npz" > "$ROOT/source_inputs.sha256"
 rsync -a --no-perms --no-owner --no-group --partial -e "$SSH" "$REMOTE:$P/model_inputs/$NAME/" "$STORAGE/$NAME/"
 (cd "$STORAGE/$NAME" && sha256sum -c "$ROOT/source_inputs.sha256" > "$ROOT/source_inputs_verification.log")
 date -Is > "$ROOT/INPUTS_STAGED"
fi
for pair in 'w64 winter_half' 's64 summer'; do
 read -r arm season <<< "$pair"
 dest="${NAME}_shared_$arm"
 /home/breallis/houfin/venv/bin/python "$CODE/research/exp/final_seasonal_inputs.py" --source "$STORAGE/$NAME" --out "$STORAGE/$dest" --season "$season"
 if [ ! -e "$LOCAL/$dest" ]; then ln -s "$STORAGE/$dest" "$LOCAL/$dest"; fi
 [ "$(readlink -f "$LOCAL/$dest")" = "$STORAGE/$dest" ]
 marker="$ROOT/${arm}_QUEUED"
 if [ ! -f "$marker" ]; then
  /home/breallis/.local/bin/pueue add --group gpu -- env FINAL_SHARED_LOCAL=1 bash "$CODE/research/bin/final_shared_fit.sh" "$arm" > "$marker"
 fi
done
date -Is > "$ROOT/LOCAL_DISPATCH_DONE"
