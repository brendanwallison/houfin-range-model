#!/bin/bash
# Diagnose which parameter block carries corrected 64b's high-gradient plateau.
set -euo pipefail
REPO=/work/07980/bwa386/ls6/houfin/age-research
source "$REPO/research/age_model/bin/task_header.sh"
OUT="$WORK/houfin/research/age_model/E064/64b"
REF=$(cat "$OUT/refine_dir.txt")
test -f "$REF/map_checkpoint.pkl"
export AGE_MODEL_CONFIG="$TWT/config/overlays/map_run64b_subcell_route.json"
export HOUFIN_MODEL_PRECISION=float64
export HOUFIN_HMC_MAP_PRECISION=float64
export HOUFIN_HMC_MAP_DIR="$REF"
export HOUFIN_HMC_ALLOW_MODEL_DRIFT=1
export HOUFIN_PROBE_HESSIAN=0
export HOUFIN_PROBE_FRESH=1
python -m src.model.age_hmc_probe > "$OUT/gradient_anatomy.log" 2>&1
PROBE=$(grep -Eo '/[^ ]+/probe.json' "$OUT/gradient_anatomy.log" | tail -1)
test -f "$PROBE"
cp "$PROBE" "$OUT/gradient_anatomy.json"
