#!/bin/bash
set -euo pipefail
ROOT=/home/breallis/houfin/work/houfin/research/FINAL_CLEAN_20261025
CODE=/mnt/c/Dev/houfin-range-model
export RESEARCH_SEALED_READ_LOG="$ROOT/reused_holdout_reads.csv" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=
for tick in $(seq 1 600); do
 if [ -f "$ROOT/evaluation/clean2/SEASONAL_SUMMER_DONE" ]; then
  cd "$CODE"
  /home/breallis/houfin/venv/bin/python research/exp/final_clean_candidates.py \
   --root "$ROOT/evaluation/clean2" --cbc /home/breallis/houfin/work/houfin/research/FINAL_HOLDOUT_20261025/sealed/cbc_circles.npz \
   --arms v2_s64 v2_w64 v2_r64 v2_r128 --out research/experiments/FINAL_CLEAN_SEASONAL_RESULTS.json > "$ROOT/seasonal_results.log"
  /home/breallis/houfin/venv/bin/python - <<'PY'
from pathlib import Path
import json
p=Path('research/experiments');d=json.loads((p/'FINAL_CLEAN_SEASONAL_RESULTS.json').read_text())
def fmt(m):
 if m.get('value') is None:return 'unsupported'
 lo,hi=m['ci95'];return f"{m['value']:+.3f} [{lo:+.3f},{hi:+.3f}]"
lines=['# Clean seasonal comparison','',d['timestamp'],'',d['interpretation'],'',
 'All summer rows compare each representation to clean v1. Pooled uses a spatially fitted readout applied to change; within uses temporal recalibration. Neither is the full population model.','',
 '| Calibration | Representation | Within difference [95%] | Pooled difference [95%] |','|---|---|---|---|']
for regime,arms in d['summer'].items():
 for arm,scores in arms.items():lines.append(f"|{regime}|{arm}|{fmt(scores['desk_within'])}|{fmt(scores['desk_pooled'])}|")
lines+=['','Winter temporal-gain differences relative to clean v1; conditional recorded list composition, not abundance truth.','',
 '| Era | Lists/circles | Summer64 | Winter64 | Full128 |','|---|---|---|---|---|']
for era,subsets in d['winter'].items():
 b=subsets['primary'];m=b['metrics'];v=[fmt(m[a+'/temporal_gain']) for a in ['v2_s64','v2_w64','v2_r128']]
 lines.append(f"|{era}|{b['lists']}/{b['circles']}|"+'|'.join(v)+'|')
lines+=['','JSON retains the paired-bootstrap comparisons and same-circle/low-counter winter sensitivities. These results reuse opened holdouts and are exploratory; no automatic promotion.','']
t=p/'FINAL_CLEAN_SEASONAL_RESULTS.tmp.md';t.write_text('\n'.join(lines));t.replace(p/'FINAL_CLEAN_SEASONAL_RESULTS.md')
PY
  exit 0
 fi
 sleep 60
done
exit 2
