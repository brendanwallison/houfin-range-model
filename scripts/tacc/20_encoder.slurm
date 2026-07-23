#!/bin/bash
#----------------------------------------------------------------------
# Community-encoder stages (SEPARATE from preprocessing; run AFTER
# 00_preprocess_all has produced states + eBird cache + amplitude points):
#     esk -> desk -> cube -> validate -> encoder-viz (optional diagnostics)
# ESK (Nystrom kernel-PCA) and DESK (autoencoder training) both use torch and are
# the heavy stages, hence a GPU queue. They are often best submitted ONE AT A TIME
# so each can be sized/queued on its own — pass STAGES:
#     STAGES=esk      sbatch --export=ALL scripts/tacc/20_encoder.slurm
#     STAGES=desk     sbatch --export=ALL scripts/tacc/20_encoder.slurm
#     STAGES="cube validate" sbatch --export=ALL scripts/tacc/20_encoder.slurm   # light; CPU is fine
#     STAGES=encoder-viz sbatch --export=ALL scripts/tacc/20_encoder.slurm
# or all four at once (the default).
#
# ENV NOTE: this needs torch importable (GPU: a CUDA torch build; CPU also works,
# just slower — cube/validate are light). The repo's `gpu` extra pins jax for the
# population model, NOT torch, so confirm torch is installed in the venv first.
#----------------------------------------------------------------------
#SBATCH -J houfin_encoder
#SBATCH -o houfin_encoder.o%j
#SBATCH -e houfin_encoder.e%j
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH --mail-type=all
##SBATCH -A REPLACE_WITH_PROJECT
set -euo pipefail

source "$SLURM_SUBMIT_DIR/scripts/tacc/env.sh"
source "$SLURM_SUBMIT_DIR/scripts/tacc/gpu_monitor.sh"
module load python/3.12.11 2>/dev/null || true
module load remora 2>/dev/null || true
[ -n "${TACC_ALLOCATION:-}" ] && echo "allocation: $TACC_ALLOCATION"
export STAGES="${STAGES:-esk desk cube validate}"

# torch preflight. ESK/DESK are heavy and need a GPU; cube/validate/encoder-viz are
# CPU-fine (they just project/plot), so only REQUIRE a CUDA device for the heavy stages.
# This lets the light stages run on shared/CPU queues (development, gpu-a100-small) where
# no GPU is bound. Import is always required; the misleading "not importable" once masked a
# failing `assert cuda.is_available()` on GPU-less nodes.
_needs_gpu=0
case " $STAGES " in
    *" esk "*|*" desk "*|*" spacetime-esk "*|*" spacetime_esk "*) _needs_gpu=1 ;;
esac
# Import check must NOT touch CUDA (torch.cuda.is_available() can hard-fail on nodes with
# no/partial driver). Print the real traceback on failure instead of a generic message.
if ! python -c "import torch; print('torch', torch.__version__)"; then
    echo "torch import failed on $(hostname); real error below:"
    python -c "import torch" || true
    exit 1
fi
if [ "$_needs_gpu" = 1 ]; then
    python -c "import torch; print('cuda', torch.cuda.is_available()); assert torch.cuda.is_available(), 'no PyTorch CUDA device'" || \
        { echo "ESK/DESK need a GPU but none is visible — use a full gpu-a100 node"; exit 1; }
    start_gpu_monitor "$SLURM_SUBMIT_DIR/gpu_encoder_${SLURM_JOB_ID}.csv" 30
    trap stop_gpu_monitor EXIT
else
    echo "[encoder] STAGES='$STAGES' are CPU-capable; not requiring (or touching) a CUDA device."
fi
date

bash "$SLURM_SUBMIT_DIR/scripts/tacc/pipeline.sh"

echo "=== 20_encoder complete ($STAGES) ==="
date
