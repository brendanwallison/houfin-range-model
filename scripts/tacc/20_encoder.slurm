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
#     STAGES=bbs-route-validate sbatch --export=ALL scripts/tacc/20_encoder.slurm
#     STAGES=encoder-viz sbatch --export=ALL scripts/tacc/20_encoder.slurm
#
# bbs-route-validate grades DESK against a no-change null on genuinely SURVEYED cell-years
# (raw BBS route counts), unlike `validate` which grades against the IDW-interpolated BBS trend
# surface. It needs the community matrix from the `bbs` stage of 04_states.slurm first.
# or all four at once (the default).
#
# QUEUE: gpu-a100-small (1 A100 40GB, 1.5 SU/hr, 48 h cap). Every stage here is
# -N 1 -n 1 and single-device (plain .to(device); no DataParallel/pmap), so the
# 3-A100 gpu-a100 node would bill 3 SU/hr to leave two GPUs idle. Per-GPU memory is
# the same 40 GB either way -- the figure desk.batch_years was tuned against. The
# -small nodes are virtual with fewer CORES and possibly less HOST RAM, which is the
# one real risk: DESK stages its whole year window through host memory before the
# device copy. A host (not CUDA) OOM is the signal to use QUEUE=gpu-a100.
#
# ENV NOTE: this needs torch importable (GPU: a CUDA torch build; CPU also works,
# just slower — cube/validate are light). The repo's `gpu` extra pins jax for the
# population model, NOT torch, so confirm torch is installed in the venv first.
#----------------------------------------------------------------------
#SBATCH -J houfin_encoder
#SBATCH -o houfin_encoder.o%j
#SBATCH -e houfin_encoder.e%j
#SBATCH -p gpu-a100-small
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
        { echo "ESK/DESK need a GPU but none is visible — use a GPU queue (gpu-a100-small)"; exit 1; }
else
    # cube and validate are not REQUIRED to have a GPU (they must still run on a login node for
    # debugging), but they do use one heavily: cube runs 124 whole-grid forwards and validate 60,
    # plus the Nystrom projection of ~890k points. So report what they got instead of staying
    # silent -- a silent CPU fallback here is a ~15x slowdown with no signal in the log.
    python -c "import torch; print('[encoder] cuda available:', torch.cuda.is_available())" || true
fi
# Telemetry for EVERY stage on a GPU queue, not just the heavy ones. Gating this on _needs_gpu
# meant STAGES="cube validate" produced no gpu_encoder_*.csv at all, so utilisation and VRAM for
# those stages could only be guessed at -- and remora's own GPU numbers are unreliable here
# (it reported 0.94 MB for cube and 15.46 MB for a desk run measured at 13.77 GB).
if command -v nvidia-smi >/dev/null 2>&1; then
    start_gpu_monitor "$SLURM_SUBMIT_DIR/gpu_encoder_${SLURM_JOB_ID}.csv" 30
    # TERM/INT as well as EXIT: an untrapped SIGTERM (which is how SLURM ends a job that hits its
    # wall clock) kills the shell via the default disposition WITHOUT running the EXIT trap, so
    # the monitor was only ever shut down cleanly on a normal exit.
    trap stop_gpu_monitor EXIT TERM INT
fi
date

bash "$SLURM_SUBMIT_DIR/scripts/tacc/pipeline.sh"

echo "=== 20_encoder complete ($STAGES) ==="
date
