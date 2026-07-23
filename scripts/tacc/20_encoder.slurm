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
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH --mail-type=all
##SBATCH -A REPLACE_WITH_PROJECT
set -euo pipefail

source "$SLURM_SUBMIT_DIR/scripts/tacc/env.sh"
module load python/3.12.11 2>/dev/null || true
module load remora 2>/dev/null || true
[ -n "${TACC_ALLOCATION:-}" ] && echo "allocation: $TACC_ALLOCATION"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || \
    { echo "torch not importable — install it in the venv before running the encoder"; exit 1; }
date

export STAGES="${STAGES:-esk desk cube validate}"
bash "$SLURM_SUBMIT_DIR/scripts/tacc/pipeline.sh"

echo "=== 20_encoder complete ($STAGES) ==="
date
