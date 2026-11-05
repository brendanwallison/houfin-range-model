#!/bin/bash
# Fixed matched two-seed sensitivity; reuse existing observation-aware scoring.
set -euo pipefail
ROOT=/home/breallis/houfin/work/houfin/research/FINAL_CLEAN_20261025
OLD=/home/breallis/houfin/work/houfin/research/FINAL_HOLDOUT_20261025
REMOTE=/work/07980/bwa386/ls6/houfin/research/final_clean_20261025
PY=/home/breallis/houfin/venv/bin/python
SSH=(ssh -S /home/breallis/.ssh/cm-tacc -o BatchMode=yes -o ConnectTimeout=15 bwa386@ls6.tacc.utexas.edu)
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONUNBUFFERED=1
export HOUFIN_DATA=/home/breallis/houfin/scratch/houfin/data HOUFIN_PROCESSED=/work/07980/bwa386/ls6/houfin/processed
export RESEARCH_SEALED_READ_LOG="$ROOT/reused_holdout_reads.csv"
for tick in $(seq 1 120); do [ -f "$ROOT/DISPATCHED" ] && break; sleep 30; done
test -f "$ROOT/DISPATCHED"
"${SSH[@]}" "mkdir -p $REMOTE/analysis"
scp -q -o ControlPath=/home/breallis/.ssh/cm-tacc "$ROOT/analysis_code/research/exp/final_holdout_extract.py" \
 "bwa386@ls6.tacc.utexas.edu:$REMOTE/analysis/extract.py"
for mode in ${FINAL_CLEAN_EVAL_MODES:-original2 clean2}; do
 case "$mode" in original2|clean2) ;; *) exit 2;; esac
 if [ "$mode" = clean2 ]; then
  for tick in $(seq 1 720); do
   if "${SSH[@]}" 'test -f /work/07980/bwa386/ls6/houfin/processed/encoder/cube_desk_final_clean_v1_s1/cube_meta.json && test -f /work/07980/bwa386/ls6/houfin/processed/encoder/cube_desk_final_clean_v1_s2/cube_meta.json && test -f /work/07980/bwa386/ls6/houfin/processed/encoder/cube_desk_final_clean_v2_s1/cube_meta.json && test -f /work/07980/bwa386/ls6/houfin/processed/encoder/cube_desk_final_clean_v2_s2/cube_meta.json'; then break; fi
   sleep 60
  done
 fi
 OUT="$ROOT/evaluation/$mode"
 mkdir -p "$OUT/features"
 for pair in 'final v1' 'final_v2 v2'; do
  read -r alias family <<< "$pair"
  source="$alias"; [ "$mode" = clean2 ] && source="final_clean_$family"
  "${SSH[@]}" "set -e; export PYTHONPATH=$REMOTE/code:$REMOTE/code/src; /work/07980/bwa386/ls6/houfin/venv/bin/python $REMOTE/analysis/extract.py --encoder-root /work/07980/bwa386/ls6/houfin/processed/encoder --cells /work/07980/bwa386/ls6/houfin/research/final_holdout_20261025/cells.npy --family $source --seeds 1 2 --out $REMOTE/evaluation/$mode/features/$alias"
  rsync -a -e 'ssh -S /home/breallis/.ssh/cm-tacc' "bwa386@ls6.tacc.utexas.edu:$REMOTE/evaluation/$mode/features/$alias/" "$OUT/features/$alias/"
 done
 cd "$ROOT/analysis_code"
 "$PY" research/exp/final_holdout_prepare.py adapters --base /home/breallis/houfin/work/houfin/research_cache/desk_tempho_1995_s1 \
  --features "$OUT/features" --out "$OUT/adapters"
 export BLR_DEVICE=cuda BLR_MAX_ITER=100 BLR_STARTS=0.5 FINAL_FIT_CACHE="$ROOT/evaluation/fit_cache"
 for spec in '1 1' '0.03 1' '0.03 2'; do
  read -r thin seed <<< "$spec"
  for arm in v1_r64 v2_r64 v2_r128; do
   rank=64;split=0
   case "$arm" in v2_r64) split=32;; v2_r128) rank=128;split=64;; esac
   dest="$OUT/summer/${arm}_f${thin}_s${seed}"; mkdir -p "$dest"
   [ -f "$dest/DONE" ] && continue
   "$PY" research/exp/final_cached_fair.py --cache "$OUT/adapters/$arm" --out "$dest" --target sealed \
    --confirmation-reason 'FINAL_CLEAN: fixed completeness-only sensitivity, matched seeds1/2; previously opened holdouts, not fresh confirmation' \
    --rank "$rank" --desk-split "$split" --early 1966 1986 --modern 2005 2025 \
    --incremental --n-pos 96 --pos-seeds 1 2 --strong-base --lean --save-species --tag _final --n-boot 1000 \
    --z-keys-only --thin "$thin" --thin-unit row --thin-seed "$seed"
   date -Is > "$dest/DONE"
  done
 done
 "$PY" research/exp/final_winter_score.py --root "$OUT" --cbc "$OLD/sealed/cbc_circles.npz" \
  --reason 'FINAL_CLEAN: fixed reused-heldout seasonal sensitivity; matched two-seed ensembles'
 date -Is > "$OUT/EVALUATION_DONE"
done
date -Is > "$ROOT/EVALUATION_DONE"
