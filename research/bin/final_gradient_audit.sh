#!/bin/bash
set -euo pipefail
ARM="$1"
ROOT=/work/07980/bwa386/ls6/houfin/research/final_age_handoff_20261025
source /work/07980/bwa386/ls6/houfin/age-research/scripts/tacc/env.sh
export PYTHONPATH="$ROOT/code_memoryfix1:$ROOT/code_memoryfix1/src" PYTHONUNBUFFERED=1
export AGE_MODEL_CONFIG="$ROOT/code_memoryfix1/config/overlays/map_run_${ARM}.json"
export HOUFIN_MODEL_PRECISION=float64 HOUFIN_HMC_MAP_PRECISION=float64 HOUFIN_Z_FLOAT32=1 HOUFIN_HMC_ALLOW_MODEL_DRIFT=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export HOUFIN_HMC_MAP_DIR="/work/07980/bwa386/ls6/houfin/processed/model_results/refined_float64__age_map_float32_${ARM}_quick90__finite_restart1"
python "$ROOT/diagnostics/final_gradient_audit.py" --out "$ROOT/gradient_audit/${ARM}.json"
