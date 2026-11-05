#!/bin/bash
set -euo pipefail
ROOT=/home/breallis/houfin/work/houfin/research/FINAL_CLEAN_20261025
cd "$ROOT/analysis_code"
export HOUFIN_DATA=/home/breallis/houfin/scratch/houfin/data HOUFIN_PROCESSED=/work/07980/bwa386/ls6/houfin/processed
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=
export RESEARCH_SEALED_READ_LOG="$ROOT/reused_holdout_reads.csv"
/home/breallis/houfin/venv/bin/python research/exp/final_winter_score.py \
 --root "$ROOT/evaluation/original2" \
 --cbc /home/breallis/houfin/work/houfin/research/FINAL_HOLDOUT_20261025/sealed/cbc_circles.npz \
 --reason 'FINAL_CLEAN: original two-seed winter sensitivity; established relocated-circle exclusion; previously opened holdouts'
date -Is > "$ROOT/evaluation/original2/EVALUATION_DONE"
