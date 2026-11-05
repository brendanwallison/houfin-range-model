#!/bin/bash
# Run after proxy scoring on the local GPU; compare one immutable TACC point.
set -euo pipefail
ROOT=/home/breallis/houfin/work/houfin/research/FINAL_LOCAL_JAX_20261025
CODE=/mnt/c/Dev/houfin-range-model
P=/work/07980/bwa386/ls6/houfin/processed
test -f "$ROOT/INPUTS_STAGED"
test -f "$ROOT/BASELINE_STAGED"
export PYTHONPATH="$ROOT/runtime:$ROOT/code_precondition1:$ROOT/code_precondition1/src"
export AGE_MODEL_CONFIG="$ROOT/code_precondition1/config/overlays/map_run_fd_control64.json"
export HOUFIN_DATA=/home/breallis/houfin/scratch/houfin/data HOUFIN_PROCESSED="$P"
export HOUFIN_MODEL_PRECISION=float64 HOUFIN_HMC_MAP_PRECISION=float64 HOUFIN_Z_FLOAT32=1
export HOUFIN_HMC_ALLOW_MODEL_DRIFT=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.72
export HOUFIN_HMC_MAP_DIR="$P/model_results/refined_float64__age_map_float32_fd_control64_quick90__finite_restart1"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
cd "$ROOT/code_precondition1"
sha256sum -c SOURCE_MANIFEST.sha256 > "$ROOT/source_verification.log"
/usr/bin/time -v /home/breallis/houfin/venv/bin/python "$CODE/research/exp/final_gradient_audit.py" \
 --out "$ROOT/local_control_baseline.json" --steps 1e-7 > "$ROOT/local_control_baseline.log" 2>&1
/home/breallis/houfin/venv/bin/python - "$ROOT" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]); a=json.loads((r/'local_control_baseline.json').read_text()); b=json.loads((r/'tacc_control_baseline.json').read_text())
du=abs(a['U']-b['U']); dg=abs(a['grad_norm']-b['grad_norm'])/max(1.,b['grad_norm'])
result={'absolute_U_difference':du,'relative_gradient_norm_difference':dg,
        'U_tolerance':1e-3,'relative_gradient_norm_tolerance':1e-3,
        'passed':du<1e-3 and dg<1e-3,
        'interpretation':'Backend agreement at one immutable point; not an optimization or convergence result.'}
(r/'backend_agreement.json').write_text(json.dumps(result,indent=2))
print(result); assert result['passed'],result
PY
date -Is > "$ROOT/BASELINE_AUDIT_DONE"
