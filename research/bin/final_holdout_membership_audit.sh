#!/bin/bash
set -euo pipefail
ROOT=/home/breallis/houfin/work/houfin/research/FINAL_HOLDOUT_20261025
cd "$ROOT/code_membership_audit"
export HOUFIN_DATA=/home/breallis/houfin/scratch/houfin/data OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
PY=/home/breallis/houfin/venv/bin/python
"$PY" -c 'from research.adopt import record_read; record_read("raw CBC benchmark membership", "FINAL_HELDOUT: quantify recognized sealed and focal species contributions to the unchanged training completeness")'
"$PY" research/exp/cbc_completeness.py \
  --raw-dir /mnt/e/Datasets/cbc_raw/20250205_FullSet_AllCountries_AllYears_CBC_Data_Request \
  --prepared /mnt/e/Datasets/cbc_raw/prepared/cbc_circles_all.npz \
  --out "$ROOT/audit/completeness_membership.npz" --bench-from 1966 \
  --taxonomy "$HOUFIN_DATA/avonet/eBird_taxonomy.csv" \
  --audit-sealed-cache /home/breallis/houfin/work/houfin/research_cache/desk_tempho_1995_s1/meta.json
