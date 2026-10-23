#!/bin/bash
# E086aa on TACC: spacetime ESK basis, winter half, all 369 non-sealed species.
set -euo pipefail
export ESK_DESK_CONFIG=research/overlays/esk_all369_winter.json
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
git log --oneline -1; nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/run_encoder.py spacetime-esk
ls -la $WORK/houfin/processed/encoder/esk_all369_winter/spacetime
