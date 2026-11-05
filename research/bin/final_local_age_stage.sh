#!/bin/bash
# Stage immutable control inputs only; never run a competing refinement.
set -euo pipefail
ROOT=/home/breallis/houfin/work/houfin/research/FINAL_LOCAL_JAX_20261025
P=/work/07980/bwa386/ls6/houfin/processed
LOCAL=/home/breallis/houfin/work/houfin/processed
REMOTE=bwa386@ls6.tacc.utexas.edu
SSH='ssh -S /home/breallis/.ssh/cm-tacc -o BatchMode=yes -o ConnectTimeout=15'
INPUT=model_inputs/numpyro_input_cbc1902_r64
mkdir -p "$ROOT" "$LOCAL/$INPUT"
$SSH "$REMOTE" "cd '$P/$INPUT' && sha256sum *.dat *.pkl *.npz" > "$ROOT/control_inputs.sha256"
rsync -a --partial -e "$SSH" "$REMOTE:$P/$INPUT/" "$LOCAL/$INPUT/"
cd "$LOCAL/$INPUT"
sha256sum -c "$ROOT/control_inputs.sha256" > "$ROOT/control_inputs_verification.log"
date -Is > "$ROOT/INPUTS_STAGED"
