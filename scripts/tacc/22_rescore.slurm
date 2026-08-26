#!/bin/bash
#----------------------------------------------------------------------
# Recompute the eigenbasis diagnostics from saved DESK checkpoints, with an error bar
# (scripts/sweep/rescore_eigenbasis.py). SUBMIT VIA scripts/tacc/submit_rescore.sh, not bare
# sbatch: this account has multiple projects, the `#SBATCH -A` line below is deliberately
# disabled, and SLURM rejects a job with no allocation.
#
# NOT A LOGIN-NODE JOB, which is how this first failed. Per run it forwards the whole
# ema_warmup_start..label_year window -- ~86 whole-grid passes, because the output EMA is a causal
# scan and a per-year encode would give z_raw instead -- and holds z_raw plus z_ema for that window,
# roughly 2 GB at the 27 km grid. Nineteen runs of that on a shared login node is what the process
# reaper exists to stop, and it duly reset the connection mid-first-run.
#
# One A100 is plenty: the work is 19 x 86 forward passes with no gradients, so gpu-a100-small at
# 1.5 SU/hr is the right queue rather than a packed gpu-a100 node.
#----------------------------------------------------------------------
#SBATCH -J houfin_rescore
#SBATCH -o houfin_rescore.o%j
#SBATCH -e houfin_rescore.e%j
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH --mail-type=all
##SBATCH -A REPLACE_WITH_PROJECT
set -euo pipefail

source "$SLURM_SUBMIT_DIR/scripts/tacc/env.sh"
module load python/3.12.11 2>/dev/null || true
[ -n "${TACC_ALLOCATION:-}" ] && echo "allocation: $TACC_ALLOCATION"
date
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

: "${RESCORE_GLOB:?RESCORE_GLOB must name the run dirs (submit via submit_rescore.sh)}"
RESCORE_DRAWS="${RESCORE_DRAWS:-6}"
RESCORE_BATCH="${RESCORE_BATCH:-1024}"
RESCORE_OUT="${RESCORE_OUT:-$HOUFIN_PROCESSED/sweeps/eigenbasis_rescore.json}"

echo "glob: $RESCORE_GLOB"
echo "draws=$RESCORE_DRAWS batch=$RESCORE_BATCH out=$RESCORE_OUT"
# Unquoted on purpose: the wrapper passes a glob pattern and it must expand here, on the node.
python "$SLURM_SUBMIT_DIR/scripts/sweep/rescore_eigenbasis.py" $RESCORE_GLOB \
    --draws "$RESCORE_DRAWS" --batch "$RESCORE_BATCH" --out "$RESCORE_OUT"

echo "=== 22_rescore complete ==="
date
