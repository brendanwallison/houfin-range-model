#!/bin/bash
set -euo pipefail
ROOT=/mnt/c/Dev/houfin-range-model/.research_runtime/community_z_repair_20261026
CODE="$ROOT/observation_audit_code"
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 PYTHONUNBUFFERED=1
export PYTHONPATH="$CODE:$CODE/src"
cd "$CODE"
ROOT="$ROOT" /home/breallis/houfin/venv/bin/python - <<'PY'
from pathlib import Path
import json,os,shutil
r=Path(os.environ['ROOT']);out=r/'observation_comparison';out.mkdir(exist_ok=True)
for arm in ['control_s1','time_r2_s1','change100','continue100']:
    audit=r/('observation_'+arm)
    if not (audit/'DONE.json').exists():raise ValueError('Incomplete audit '+arm)
    src=audit/'evaluation/backcast40/control';dst=out/arm
    dst.mkdir(exist_ok=True)
    summary=json.loads((src/'SUMMER_DONE.json').read_text())
    for name in ['SUMMER_DONE.json','WINTER_DONE.json','winter_lists.npz',*[p+'.npz' for p in summary['panels']]]:
        shutil.copyfile(src/name,dst/name)
(out/'PROVENANCE.json').write_text(json.dumps({'reference_for_loss_test':'continue100',
    'budget':'change100/continue100 share warm start,100 epochs, optimizer reset and learning rate.',
    'scope':'Community-observation EMA; instantaneous population cubes remain separate.'},indent=2))
PY
PY=/home/breallis/houfin/venv/bin/python
"$PY" research/exp/z_repair_compare.py --root "$ROOT" --evaluation-root "$ROOT/observation_comparison" --reference control_s1 --arms time_r2_s1 change100 continue100 --output-stem ALL_OBSERVATION_COMPARISON
"$PY" research/exp/z_repair_compare.py --root "$ROOT" --evaluation-root "$ROOT/observation_comparison" --reference continue100 --arms change100 --output-stem MATCHED_CHANGE_LOSS_COMPARISON
