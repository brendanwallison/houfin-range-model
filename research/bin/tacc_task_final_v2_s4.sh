#!/bin/bash
# FINAL DESK candidate v2 (equal summer / winter halves) seed 4: DESK + cube [z, winter half].
set -euo pipefail
export ESK_DESK_CONFIG=config/overlays/desk_final_v2_s4.json PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c "import torch; assert torch.cuda.is_available() and torch.cuda.device_count() > 0, 'no usable GPU on this node'"
git log --oneline -1; nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/run_encoder.py desk
python scripts/run_encoder.py cube
