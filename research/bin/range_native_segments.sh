#!/bin/bash
# Prepare native-habitat segments locally in float32; publish verified new files.
set -euo pipefail
REPO=/mnt/c/Dev/houfin-range-model
CODE=/mnt/e/Datasets/houfin/grid54/runs/step33_nestcbc_s41/code
PY=/home/breallis/houfin/venv/bin/python
ROOT=/work/07980/bwa386/ls6/houfin
INPUT=$ROOT/processed/field54/stageF/model_inputs/numpyro_input_field54prod_ema_s64_grid54_obsroute
SOCKET=/home/breallis/.ssh/cm-tacc
HOST=bwa386@ls6.tacc.utexas.edu
exec 9> "$REPO/.research_runtime/range_monitor/local_gpu.lock"
flock -n 9
cd "$CODE"
sha256sum --quiet -c ../CODE_MANIFEST.sha256
export HOUFIN_DATA=/home/breallis/houfin/scratch/houfin/data HOUFIN_PROCESSED=$ROOT/processed
export HOUFIN_MODEL_PRECISION=float32 HOUFIN_Z_FLOAT32=1 PYTHONUNBUFFERED=1
export AGE_MODEL_CONFIG=$CODE/config/overlays/grid54_step33_nestcbc_s41.json
export DATA_CONFIG=$CODE/config/overlays/grid54_data.json
export PYTHONPATH=/home/breallis/houfin/work/houfin/research/FINAL_LOCAL_JAX_20261025/runtime:$CODE:$CODE/src
export PATH=/home/breallis/houfin/venv/lib/python3.12/site-packages/triton/backends/nvidia/bin:$PATH
export XLA_FLAGS=--xla_gpu_enable_command_buffer= XLA_PYTHON_CLIENT_MEM_FRACTION=.55
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
unset JAX_PLATFORMS HOUFIN_SYNTHETIC_OBS
if [ ! -e "$INPUT/z_disp_seg4_meta.json" ]; then
  timeout --kill-after=30 3600 taskset -c 0-7 "$PY" -m src.data.preprocess.segmented_path_features --n-seg 4
fi
"$PY" - "$INPUT" <<'PY'
import hashlib,json,pathlib,sys,math
p=pathlib.Path(sys.argv[1]); m=json.load(open(p/'z_disp_seg4_meta.json'))
assert m['n_seg']==4 and m['reproduction_vs_stored_Zdisp']['years']==m['shape'][0]
assert m['reproduction_vs_stored_Zdisp']['max_median_rel']<=1e-5
assert (p/'z_disp_seg4_gathered.dat').stat().st_size==math.prod(m['shape'])*4
files=['metadata.pkl','z_disp_seg4_gathered.dat','z_disp_seg4_meta.json']
receipt={'files':{},'precision':'float32','source_step':'step33_nestcbc_s41'}
for f in files:
 with (p/f).open('rb') as s: receipt['files'][f]=hashlib.file_digest(s,'sha256').hexdigest()
(p/'native_segments_transfer.json').write_text(json.dumps(receipt,indent=2)+'\n')
PY
UPLOAD=$ROOT/research/grid54/monitor/native_segments_upload
ssh -n -S "$SOCKET" "$HOST" "mkdir -p $UPLOAD"
for file in z_disp_seg4_gathered.dat z_disp_seg4_meta.json native_segments_transfer.json; do
  scp -o ControlPath="$SOCKET" "$INPUT/$file" "$HOST:$UPLOAD/$file"
done
ssh -S "$SOCKET" "$HOST" "$ROOT/venv/bin/python -" <<'PY'
import hashlib,json,pathlib,os
root=pathlib.Path('/work/07980/bwa386/ls6/houfin')
upload=root/'research/grid54/monitor/native_segments_upload'
target=root/'processed/field54/stageF/model_inputs/numpyro_input_field54prod_ema_s64_grid54_obsroute'
receipt=json.load(open(upload/'native_segments_transfer.json'))
def sha(p):
 with p.open('rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()
assert sha(target/'metadata.pkl')==receipt['files']['metadata.pkl'],'input metadata differs'
for name in ['z_disp_seg4_gathered.dat','z_disp_seg4_meta.json']:
 assert sha(upload/name)==receipt['files'][name],f'transfer mismatch: {name}'
 if (target/name).exists():assert sha(target/name)==receipt['files'][name],f'existing input differs: {name}'
for name in ['z_disp_seg4_gathered.dat','z_disp_seg4_meta.json']:
 if not (target/name).exists():os.rename(upload/name,target/name)
(root/'research/grid54/overnight/native_segments_transfer.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('Native segments published; existing input files preserved')
PY
