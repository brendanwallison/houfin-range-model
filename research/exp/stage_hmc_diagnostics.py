"""Queue bounded HMC geometry and actual NUTS screens on existing workers."""
import argparse,fcntl,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/work/07980/bwa386/ls6/houfin/research/grid54')
GEOMETRY=['step33_nestcbc_s41','step33_nestcbc_abs_s41','step34_abs_natdep_s41','step34_abs_v2_natdep_s41']
SAMPLING=['step33_nestcbc_abs_s41','step34_abs_natdep_s41']


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute');p.add_argument('--block-only',action='store_true');a=p.parse_args()
 if a.execute:
  mode,step=a.execute.split('/')
  if mode not in ['geometry','nuts','nuts_block'] or step not in GEOMETRY:raise ValueError('invalid diagnostic')
  out=ROOT/'runs'/step/'out'/('hmc_geometry' if mode=='geometry' else 'hmc_screen_block' if mode=='nuts_block' else 'hmc_screen');out.mkdir(parents=True,exist_ok=True)
  with (out/'attempt.lock').open('w') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   if (out/'ATTEMPT_STARTED.json').exists():
    if mode=='geometry' or not list(out.glob('chain*/checkpoint.pkl')) or (out/'READY.json').exists():
     raise RuntimeError('existing diagnostic retained')
    (out/f'RESUME_STARTED_{time.time_ns()}.json').write_text(json.dumps({'epoch':time.time()})+'\n')
   start=time.time()
   if not (out/'ATTEMPT_STARTED.json').exists():(out/'ATTEMPT_STARTED.json').write_text(json.dumps({'epoch':start})+'\n')
   script=ROOT/'monitor'/f'hmc_{mode}_{step}.sh'
   rc=subprocess.run(['timeout','--kill-after=30','7200' if mode=='geometry' else '21600','bash',str(script)]).returncode
   (out/'RUN_RECEIPT.json').write_text(json.dumps({'exit_code':rc,'elapsed_seconds':time.time()-start})+'\n')
  raise SystemExit(rc)
 specs=[]
 for mode,steps in ([('nuts_block',SAMPLING)] if a.block_only else [('geometry',GEOMETRY),('nuts',SAMPLING)]):
  for step in steps:
   code=ROOT/'runs'/step/'code';ref=ROOT.parents[1]/f'processed/model_results/refined_float64__age_map_float32_grid54_{step}_quick90'
   report=json.load(open(ref/'refine_report.json'))
   if not report.get('valid'):raise ValueError('unaccepted reference '+step)
   destination=ROOT/'runs'/step/'out'/('hmc_geometry' if mode=='geometry' else 'hmc_screen_block' if mode=='nuts_block' else 'hmc_screen')
   if destination.exists():raise FileExistsError('preserve existing diagnostic '+str(destination))
   geometry=ROOT/'runs'/step/'out/hmc_geometry'
   command=(f'"$PY" {ROOT}/monitor/run_hmc_geometry_audit.py --run "{ref}" --out "{destination}" --device gpu --steps 48 --prior-draws 256 --rough-directions 4' if mode=='geometry' else
      f'"$PY" {ROOT}/monitor/run_hmc_screen.py --run "{ref}" --geometry "{geometry}" --out "{destination}" --warmup {512 if mode=="nuts_block" else 128} --samples 128 --max-depth 6 --metric {"block" if mode=="nuts_block" else "diag"}')
   script=f'''#!/bin/bash
set -euo pipefail
source {ROOT.parents[1]}/age-research/scripts/tacc/env.sh
CODE={code}
cd "$CODE"
sha256sum --quiet -c ../CODE_MANIFEST.sha256
export AGE_MODEL_CONFIG=$CODE/config/overlays/grid54_{step}.json DATA_CONFIG=$CODE/config/overlays/grid54_data.json
export HOUFIN_MODEL_PRECISION=float64 JAX_ENABLE_X64=True HOUFIN_Z_FLOAT32=1 JAX_PLATFORMS=cuda
export PYTHONPATH=$CODE:$CODE/src:{ROOT}/monitor
export XLA_PYTHON_CLIENT_MEM_FRACTION=.85 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HOUFIN_SYNTHETIC_OBS
PY={ROOT.parents[1]}/venv/bin/python
{command}
'''
   (ROOT/'monitor'/f'hmc_{mode}_{step}.sh').write_text(script)
   priority='0630' if mode=='geometry' else '0640';q=ROOT/'queue/pending'/f'{priority}_hmc_{mode}_{step}.sh'
   if q.exists():raise FileExistsError(q)
   temp=q.with_suffix('.tmp');temp.write_text(f'#!/bin/bash\nset -euo pipefail\nexec {ROOT.parents[1]}/venv/bin/python {ROOT}/monitor/stage_hmc_diagnostics.py --execute {mode}/{step}\n');os.rename(temp,q)
   spec={'mode':mode,'step':step,'target':'full unconstrained nominal posterior including Jacobians and observer nuisance parameters',
     'geometry_steps':48 if mode=='geometry' else None,'prior_scale_draws':256 if mode=='geometry' else None,
     'time_limit_seconds':7200 if mode=='geometry' else 21600,
     'sampling':{'chains':2,'warmup_per_chain':512 if mode=='nuts_block' else 128,'draws_per_chain':128,'max_depth':6,'target_accept':.9,'metric':'block' if mode=='nuts_block' else 'diag'} if mode!='geometry' else None,
     'scope':'comparison/screening, not certification of inference'}
   specs.append(spec);print('QUEUED',mode,step,flush=True)
 specfile=ROOT/'monitor/hmc_diagnostic_specs.json'
 if a.block_only:specs=json.loads(specfile.read_text())+specs
 specfile.write_text(json.dumps(specs,indent=2)+'\n')


if __name__=='__main__':main()
