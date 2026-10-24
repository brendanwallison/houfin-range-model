#!/bin/bash
# E086ac fold 2: all-sedentary summer ESK basis without the fold's species.
set -euo pipefail
export ESK_DESK_CONFIG=research/overlays/esk_ac_f2.json PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
git log --oneline -1
python scripts/run_encoder.py spacetime-esk
