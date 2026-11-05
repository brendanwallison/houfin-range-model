#!/bin/bash
# Exact original 500-epoch recipe; only CBC completeness/operator inputs change.
set -euo pipefail
FAMILY="$1" SEED="$2"
if [ "${FINAL_CLEAN_LOCAL:-0}" = 1 ]; then
 ROOT=/home/breallis/houfin/work/houfin/research/FINAL_CLEAN_20261025
 export HOUFIN_DATA=/home/breallis/houfin/scratch/houfin/data HOUFIN_PROCESSED=/home/breallis/houfin/work/houfin/processed
 export PATH=/home/breallis/houfin/venv/bin:$PATH
else
 ROOT=/work/07980/bwa386/ls6/houfin/research/final_clean_20261025
 source /work/07980/bwa386/ls6/houfin/age-research/scripts/tacc/env.sh
fi
test -f "$ROOT/PREPARED"
cd "$ROOT/code"
export ESK_DESK_CONFIG="config/overlays/desk_final_clean_${FAMILY}_s${SEED}.json"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(), flush=True)'
python scripts/run_encoder.py desk
python scripts/run_encoder.py cube
date -Is > "$ROOT/${FAMILY}_s${SEED}_DONE"
