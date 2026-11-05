"""Stage the five registered configuration candidates from frozen source d4435fb."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

LOCAL=Path('/mnt/e/Datasets/houfin/grid54/runs')
LIVE=Path('/home/breallis/houfin/work/houfin/research/GRID54_LADDER_20261029/code')
REPO=Path('/mnt/c/Dev/houfin-range-model')
STEPS={
 'step33_nestcbc_s41':(41,False,False),
 'step33_nestcbc_s42':(42,False,False),
 'step33_nestcbc_natdep_s41':(41,True,False),
 'step33_nestcbc_natdep_s42':(42,True,False),
 'step33_nestcbc_abs_s41':(41,False,True)}


def main():
    archive=REPO/'.research_runtime/pathology_source_d4435fb.tar'
    with archive.open('wb') as stream:
        subprocess.run(['git','-C',str(LIVE),'archive','d4435fb'],stdout=stream,check=True)
    def cfg(step,filename=None):
        return json.loads((LOCAL/step/'code/config/overlays'/(filename or f'grid54_{step}.json')).read_text())
    base=cfg('step31_nestonly')
    cbc=cfg('step32_cbc_yb')['population_model']['invasion_pulse_prior']
    departure=cfg('step25_natdep')['population_model']['native_departure']
    absolute=cfg('step0_s64','grid54_step0.json')['population_model']
    stage=REPO/'.research_runtime/pathology_stage'; stage.mkdir(exist_ok=True)
    specs={}
    for name,(seed,native,abs_allee) in STEPS.items():
        run=LOCAL/name
        if run.exists(): raise FileExistsError(run)
        code=run/'code'; code.mkdir(parents=True)
        with tarfile.open(archive) as tar: tar.extractall(code,filter='data')
        config=copy.deepcopy(base)
        pop=config['population_model']
        pop['pinned_sites']['log_inv_total']=-.283
        pop['invasion_pulse_prior']=copy.deepcopy(cbc)
        if native: pop['native_departure']=copy.deepcopy(departure)
        if abs_allee:
            pop['allee_prior']=copy.deepcopy(absolute['allee_prior'])
            pop['allee_prior']['form']='absolute'
            pop['pinned_sites']['n50_raw']=absolute['pinned_sites']['n50_raw']
        for key in ['map','vis_map']:
            config['run_names'][key]='age_map_{precision}_grid54_'+name
        config['_pathology_candidate']={'registration':'PARSIMONIOUS_CANDIDATES_20261031.md',
            'source_commit':'d4435fb','seed':seed,'cbc_founder_pin':-.283,
            'native_departure':native,'absolute_allee_sensitivity':abs_allee}
        target=code/'config/overlays'/f'grid54_{name}.json'
        target.write_text(json.dumps(config,indent=2)+'\n')
        (code/'config/overlays/grid54_data.json').write_bytes((LOCAL/'step31_nestonly/code/config/overlays/grid54_data.json').read_bytes())
        (run/'grid54_fit.sh').write_bytes((LOCAL/'step31_nestonly/grid54_fit.sh').read_bytes())
        manifest=[]
        for path in sorted(code.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc':
                manifest.append(hashlib.sha256(path.read_bytes()).hexdigest()+'  ./'+path.relative_to(code).as_posix())
        (run/'CODE_MANIFEST.sha256').write_text('\n'.join(manifest)+'\n')
        spec={'step':name,'seed':seed,'source_commit':'d4435fb',
              'config_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
              'changes':config['_pathology_candidate']}
        (run/'RUN_SPEC.json').write_text(json.dumps(spec,indent=2)+'\n'); specs[name]=spec
        # The entire fit+readout is limited by the queue wrapper to two hours.
        script=f'''#!/bin/bash
set -euo pipefail
ROOT=/work/07980/bwa386/ls6/houfin
R=$ROOT/research/grid54/runs/{name}
CODE=$R/code
source "$ROOT/age-research/scripts/tacc/env.sh"
export HOUFIN_MAP_SEED={seed} HOUFIN_Z_FLOAT32=1 PYTHONUNBUFFERED=1
unset HOUFIN_SYNTHETIC_OBS
bash "$R/grid54_fit.sh" {name} grid54_{name}.json
cd "$CODE"
export AGE_MODEL_CONFIG=$CODE/config/overlays/grid54_{name}.json DATA_CONFIG=$CODE/config/overlays/grid54_data.json
export PYTHONPATH=$CODE:$CODE/src HOUFIN_MODEL_PRECISION=float64 HOUFIN_PROCESSED=$ROOT/processed
export XLA_PYTHON_CLIENT_MEM_FRACTION=${{GRID54_MEM_FRACTION:-.85}}
PY=$ROOT/venv/bin/python
REF=$ROOT/processed/model_results/refined_float64__age_map_float32_grid54_{name}_quick90
"$PY" "$ROOT/research/grid54/monitor/range_pathology_readout.py" --run "$REF" --out "$R/out/pathology.json" > "$R/out/pathology.log" 2>&1
"$PY" "$ROOT/research/grid54/monitor/grid54_front_speed.py" "$REF" "$R/out/front_speed.json" > "$R/out/front_speed.log" 2>&1
'''
        (run/'run_candidate.sh').write_text(script)
    batches={'0500_candidates_seed42.sh':['step33_nestcbc_s42','step33_nestcbc_natdep_s42'],
             '0505_candidate_absolute.sh':['step33_nestcbc_abs_s41'],
             '0510_candidates_seed41.sh':['step33_nestcbc_s41','step33_nestcbc_natdep_s41']}
    for filename,names in batches.items():
        fraction='.42' if len(names)==2 else '.85'
        script='#!/bin/bash\nset -uo pipefail\nexport OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4\npids=()\n'
        for name in names:
            script+=f'env GRID54_MEM_FRACTION={fraction} timeout 7200 bash /work/07980/bwa386/ls6/houfin/research/grid54/runs/{name}/run_candidate.sh > "$0.{name}.log" 2>&1 &\npids+=($!)\n'
        script+='rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done; exit "$rc"\n'
        (stage/filename).write_text(script)
    (REPO/'research/experiments/PARSIMONIOUS_CANDIDATES_20261031.json').write_text(json.dumps(specs,indent=2)+'\n')
    print('Staged',len(specs),'configurations; no tasks published yet')


if __name__=='__main__': main()
