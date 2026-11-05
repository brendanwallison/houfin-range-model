#!/bin/bash
set -uo pipefail
ROOT=/work/07980/bwa386/ls6/houfin
OUT=$ROOT/research/grid54/overnight
mkdir -p "$OUT"
source "$ROOT/age-research/scripts/tacc/env.sh"
PY=$ROOT/venv/bin/python
export HOUFIN_PROCESSED=$ROOT/processed
export HOUFIN_MODEL_PRECISION=float64 HOUFIN_HMC_PRECISION=float64 HOUFIN_Z_FLOAT32=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=.85 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
overall=0
for pair in 'step31_nestnull null_h0' 'step31_nestrec rec_h03'; do
 read -r STEP SUFFIX <<< "$pair"
 CODE=$ROOT/research/grid54/runs/$STEP/code
 export AGE_MODEL_CONFIG=$CODE/config/overlays/grid54_$STEP.json
 export DATA_CONFIG=$CODE/config/overlays/grid54_data.json
 export PYTHONPATH=$CODE:$CODE/src
 if [ "$SUFFIX" = null_h0 ]; then BBS=yb_j1_null_h0; else BBS=yb_j1_rec_h03; fi
 export HOUFIN_SYNTHETIC_OBS=$ROOT/research/grid54/synthetic/$BBS.npz
 cd "$CODE" || exit 1
 (sha256sum --quiet -c ../CODE_MANIFEST.sha256) || exit 1
 REPORT=$OUT/${STEP}_objective.json
 if [ -f "$REPORT" ]; then
  echo "[audit] existing receipt retained: $REPORT"; continue
 fi
 echo "[audit] START $STEP $(date -Is)"
 if timeout 1800 "$PY" "$ROOT/research/grid54/monitor/nest_truth_audit.py" \
   --run "$ROOT/processed/model_results/refined_float64__age_map_float32_grid54_${STEP}_quick90" \
   --truth "$ROOT/processed/grid54/nestwatch/nests_syn_$SUFFIX.json" \
   --out "$REPORT" > "$OUT/${STEP}_objective.log" 2>&1; then
  echo "[audit] DONE $STEP $(date -Is)"
 else
  rc=$?; overall=1
  "$PY" - "$REPORT" "$STEP" "$rc" <<'PY'
import json,pathlib,sys,datetime
p=pathlib.Path(sys.argv[1]); tmp=p.with_suffix('.tmp.json')
tmp.write_text(json.dumps({'status':'failed','step':sys.argv[2],'exit_code':int(sys.argv[3]),
                          'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
tmp.replace(p)
PY
  echo "[audit] FAILED $STEP exit $rc $(date -Is)"
 fi
done
exit "$overall"
