#!/bin/bash
# Reserve the two already-registered NestWatch bias guards for local float32 MAP.
# Return the original bundle to TACC for ALL float64 diagnostics/refinement.
set -euo pipefail
SOCKET=/home/breallis/.ssh/cm-tacc
HOST=bwa386@ls6.tacc.utexas.edu
REMOTE=/work/07980/bwa386/ls6/houfin
LOCAL=/mnt/e/Datasets/houfin/grid54
PY=/home/breallis/houfin/venv/bin/python
MODE=${1:-bias}
case "$MODE" in
 bias) BUNDLE=0402_bundle_nest_g3.sh; STEPS='step31_nestsub step31_nestdrop' ;;
 candidates) BUNDLE=0510_candidates_seed41.sh; STEPS='step33_nestcbc_s41 step33_nestcbc_natdep_s41' ;;
 *) echo 'Unknown registered local-MAP batch'; exit 2 ;;
esac
SOURCE=/mnt/c/Dev/houfin-range-model
CLAIMED=0
release() {
 if [ "$CLAIMED" = 1 ]; then
  ssh -n -S "$SOCKET" "$HOST" "test ! -f $REMOTE/research/grid54/queue/pending/$BUNDLE && mv $REMOTE/research/grid54/queue/local_running/$BUNDLE $REMOTE/research/grid54/queue/pending/$BUNDLE" || {
   echo "RELEASE FAILED: reservation retained; see local_running/$BUNDLE"; return 1;
  }
  echo "[offload] bundle returned to TACC $(date -Is)"
 fi
}
trap release EXIT
trap 'exit 130' INT TERM
# Exclusivity with other local offload processes; remote rename arbitrates with workers.
exec 9> "$SOURCE/.research_runtime/range_monitor/local_gpu.lock"
flock -n 9
ssh -S "$SOCKET" "$HOST" python3 - "$BUNDLE" <<'REMOTE_CLAIM'
import pathlib,json,sys,os,time
name=sys.argv[1]
assert name in ['0402_bundle_nest_g3.sh','0510_candidates_seed41.sh']
q=pathlib.Path('/work/07980/bwa386/ls6/houfin/research/grid54/queue'); (q/'local_running').mkdir(exist_ok=True)
dest=q/'local_running'/name
if dest.exists():
 info_path=dest.with_suffix('.reservation.json'); info=json.loads(info_path.read_text())
 assert name=='0510_candidates_seed41.sh' and info['state']=='reserved_for_local'
 info.update(state='local_running',started_epoch=time.time()); info_path.write_text(json.dumps(info,indent=2)+'\n')
else:
 os.rename(q/'pending'/name,dest)
REMOTE_CLAIM
CLAIMED=1
echo "[offload] claimed $BUNDLE $(date -Is)"
export HOUFIN_DATA=/home/breallis/houfin/scratch/houfin/data
export HOUFIN_PROCESSED=/work/07980/bwa386/ls6/houfin/processed
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HOUFIN_MODEL_PRECISION=float32 HOUFIN_Z_FLOAT32=1 HOUFIN_MAP_PROFILE=quick90 HOUFIN_MAP_SEED=41
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 PYTHONUNBUFFERED=1
export PATH=/home/breallis/houfin/venv/lib/python3.12/site-packages/triton/backends/nvidia/bin:$PATH
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
if [ "$MODE" = bias ]; then export HOUFIN_SYNTHETIC_OBS="$LOCAL/synthetic/yb_j1_null_h0.npz"; else unset HOUFIN_SYNTHETIC_OBS; fi
unset HOUFIN_MAP_REUSE HOUFIN_MODEL_INPUT_RESIDENCY JAX_PLATFORMS
for STEP in $STEPS; do
 R="$LOCAL/runs/$STEP"; CODE="$R/code"; OUT="$R/out"
 mkdir -p "$OUT"
 cd "$CODE"
 sha256sum --quiet -c ../CODE_MANIFEST.sha256
 export PYTHONPATH="/home/breallis/houfin/work/houfin/research/FINAL_LOCAL_JAX_20261025/runtime:$CODE:$CODE/src"
 export AGE_MODEL_CONFIG="$CODE/config/overlays/grid54_$STEP.json"
 export DATA_CONFIG="$CODE/config/overlays/grid54_data.json"
 echo "[offload] float32 MAP $STEP $(date -Is)"
 timeout 3600 taskset -c 0-7 "$PY" -m src.model.age_run_map > "$OUT/map.log" 2>&1
 MAP="$HOUFIN_PROCESSED/model_results/age_map_float32_grid54_${STEP}_quick90"
 export LOCAL_TRANSFER_MAP="$MAP"
 "$PY" - <<'PY'
import hashlib,json,os,pathlib
from src.config_utils import load_age_model_config
c=load_age_model_config(); code=pathlib.Path(os.environ['AGE_MODEL_CONFIG']).parents[2]
paths={'nests':pathlib.Path(c['population_model']['nest_likelihood']['path']),
       'code_manifest':code.parent/'CODE_MANIFEST.sha256'}
if os.environ.get('HOUFIN_SYNTHETIC_OBS'): paths['synthetic']=pathlib.Path(os.environ['HOUFIN_SYNTHETIC_OBS'])
d={}
for k,p in paths.items():
 with p.open('rb') as f: d[k]=hashlib.file_digest(f,'sha256').hexdigest()
d.update(device='RTX 3090',precision='float32',seed=41)
pathlib.Path(os.environ['LOCAL_TRANSFER_MAP'],'local_map_transfer.json').write_text(json.dumps(d,indent=2)+'\n')
PY
 UPLOAD="$REMOTE/research/grid54/monitor/upload_${STEP}_$(date +%s)_$$"
 ssh -n -S "$SOCKET" "$HOST" "mkdir $UPLOAD"
 scp -r -o ControlPath="$SOCKET" "$MAP/." "$HOST:$UPLOAD/"
 ssh -n -S "$SOCKET" "$HOST" "env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 $REMOTE/venv/bin/python $REMOTE/research/grid54/monitor/accept_local_map.py $UPLOAD $STEP"
 date -Is > "$OUT/LOCAL_MAP_TRANSFERRED"
done
echo "[offload] float32 stages transferred; refinement remains on TACC $(date -Is)"
