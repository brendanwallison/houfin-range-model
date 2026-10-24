#!/bin/bash
# FINAL DESK candidate seed 1: DESK + cube (production paths, own output dirs).
set -euo pipefail
export ESK_DESK_CONFIG=config/overlays/desk_final_s1.json PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c "import torch; assert torch.cuda.is_available() and torch.cuda.device_count() > 0, 'no usable GPU on this node'"
git log --oneline -1; nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/run_encoder.py desk
python scripts/run_encoder.py cube
