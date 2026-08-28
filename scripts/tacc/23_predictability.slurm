#!/bin/bash
#----------------------------------------------------------------------
# Per-ESK-component predictability from the existing covariates
# (scripts/diagnostics/component_predictability.py). SUBMIT VIA
# scripts/tacc/submit_predictability.sh, not bare sbatch: this account has multiple projects,
# the `#SBATCH -A` line below is deliberately disabled, and SLURM rejects a job with no
# allocation.
#
# NOT A LOGIN-NODE JOB, but not a compute-node job either. The first version of this header
# claimed "a sustained n*d^2 dgemm that saturates every core" and ~1 GB resident, and put it on
# `development`. Both numbers were wrong, and the queue followed from them:
#
#   * the model grid is 133 x 224 = 29,792 cells, so a per-year state is ~10 MB and the whole
#     load loop is seconds, not minutes;
#   * the interaction rung's Gram is n*d^2 ~= 4.3e12 flops at 150k rows and d~5,400 -- about
#     10 s on 16 cores and 3 s on 128. Nothing here can use a full node;
#   * the real peak is the ESK PROJECTION, not the BLAS. project_into_z caps its CPU batch at
#     5,000 rows and holds ~5 live (batch, M) tensors; at M=16,000 landmarks that is ~320 MB
#     each, so ~1.6 GB, and the feature matrix plus its standardized copy lands in the same
#     range. Budget ~2 GB.
#
# So this is the "numpy/memmap I/O plus a one-time build, fine on CPU" shape that 25_model_prep
# already sends to vm-small, and `development` was the wrong home: on LS6 it is the scarce
# short-iteration queue (one job per user, 2 h cap) and it bills a 128-core node this job cannot
# use. vm-small is the default here for that reason.
#
# If vm-small turns out to be memory-tight for the ~2 GB peak, the fallback needs no edit --
# the wrapper takes QUEUE/TIME from the environment:
#
#     QUEUE=development TIME=02:00:00 bash scripts/tacc/submit_predictability.sh
#
# A login node is still the wrong answer: ~2 GB resident and a multi-minute CPU projection is
# what a shared login node's process reaper kills, and the bare invocation this script first
# carried was copied from basis_domain_gap.py, whose 0.04 GiB peak is what justifies its.
#
#SBATCH -J houfin_predict
#SBATCH -o houfin_predict.o%j
#SBATCH -e houfin_predict.e%j
#SBATCH -p vm-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH --mail-type=all
##SBATCH -A REPLACE_WITH_PROJECT
set -euo pipefail

source "$SLURM_SUBMIT_DIR/scripts/tacc/env.sh"
module load python/3.12.11 2>/dev/null || true
[ -n "${TACC_ALLOCATION:-}" ] && echo "allocation: $TACC_ALLOCATION"
date

# Set the native thread pools explicitly. Left unset, OpenBLAS/MKL pick their own default,
# which has been either the whole machine or a single core -- and on a shared VM partition the
# first is the antisocial one. Take the node's actual allocation, and fall back LOW rather than
# to a compute-node figure: this used to default to 32, which oversubscribes a vm-small slot.
# Not pinned to 1 as in 04_states, which is I/O-bound and parallelizes reads and writes itself;
# here the projection and the ridge are single BLAS/torch calls that do want the cores present.
NTHREADS="${PREDICT_THREADS:-${SLURM_CPUS_ON_NODE:-8}}"
export OMP_NUM_THREADS="$NTHREADS" OPENBLAS_NUM_THREADS="$NTHREADS" \
       MKL_NUM_THREADS="$NTHREADS" VECLIB_MAXIMUM_THREADS="$NTHREADS" \
       NUMEXPR_NUM_THREADS="$NTHREADS"
echo "threads: $NTHREADS (SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-unset})"

PREDICT_MAX_ROWS="${PREDICT_MAX_ROWS:-300000}"
PREDICT_PAIRS="${PREDICT_PAIRS:-2000}"
PREDICT_PCA_DIM="${PREDICT_PCA_DIM:-48}"
PREDICT_RFF_WIDTH="${PREDICT_RFF_WIDTH:-2048}"
PREDICT_OUT="${PREDICT_OUT:-$HOUFIN_PROCESSED/encoder/component_predictability.json}"
mkdir -p "$(dirname "$PREDICT_OUT")"

echo "max_rows=$PREDICT_MAX_ROWS pairs=$PREDICT_PAIRS pca_dim=$PREDICT_PCA_DIM"
echo "rff_width=$PREDICT_RFF_WIDTH out=$PREDICT_OUT"

python "$SLURM_SUBMIT_DIR/scripts/diagnostics/component_predictability.py" \
    --max-rows "$PREDICT_MAX_ROWS" \
    --pairs "$PREDICT_PAIRS" \
    --pca-dim "$PREDICT_PCA_DIM" \
    --rff-width "$PREDICT_RFF_WIDTH" \
    --out "$PREDICT_OUT"

echo "=== 23_predictability complete ==="
date
