"""Immutable matched movement/K-floor and historical-CBC experiments (2026-11-02)."""
import argparse,fcntl,hashlib,json,math,os,shutil,subprocess,time
from pathlib import Path
R=Path('/work/07980/bwa386/ls6/houfin/research/grid54')
PARENT='step38_cbc1951_noncbc_free_s41'

def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def replace(p,a,b):
    text=p.read_text()
    if text.count(a)!=1:raise ValueError(f'{p}: ambiguous patch {a[:80]}')
    p.write_text(text.replace(a,b))

def patch(code):
    for name in ['priority_overflow.py','cbc_early.py']:
        shutil.copy2(R/'monitor'/name,code/'src/model'/name)
    af=code/'src/model/age_forward.py'
    replace(af,'import jax.scipy.ndimage','''import jax.scipy.ndimage
from src.config_utils import load_age_model_config
from src.model.priority_overflow import departure_probabilities
_OVERFLOW = dict(load_age_model_config()['population_model'].get('retention_overflow') or {})
OVERFLOW_ENABLED = _OVERFLOW.get('mode','veit') != 'veit'
def overflow_probabilities(a,j,k):
    return departure_probabilities(a,j,k,_OVERFLOW['mode'],float(_OVERFLOW.get('width_fraction',.05)))''')
    replace(af,'    departure_density=None,','    departure_density=None,\n    departure_probabilities_override=None,')
    replace(af,'    if departure_scale is not None:     # population_model.native_departure',
      '''    if departure_probabilities_override is not None:
        p_adult,p_total=departure_probabilities_override
    elif OVERFLOW_ENABLED:
        p_adult,p_total=overflow_probabilities(N_a,N_j,K_safe)
    if departure_scale is not None:     # population_model.native_departure''')
    fields=code/'src/model/age_fields.py'
    replace(fields,'from jax import lax, checkpoint','''from jax import lax, checkpoint
from src.config_utils import load_age_model_config
_FLOOR = float(load_age_model_config()['population_model'].get('capacity_floor_internal',0.))''')
    replace(fields,'        # 5. Map path habitat to Journey Survival',
      '''        # Additive positive link floor after disease; raw baseline/native seed unchanged.
        K_val = K_val + _FLOOR

        # 5. Map path habitat to Journey Survival''')
    model=code/'src/model/cbc_joint.py'
    replace(model,'    rows,cols=data[\'land_rows\'],data[\'land_cols\']',
      '''    if spec.get('family')=='early_detection':
        from src.model.cbc_early import early_observe
        return early_observe(data,Na_grid,Nj_grid,pop_scalar,spec)
    rows,cols=data['land_rows'],data['land_cols']''')
    for p in [af,fields,model,code/'src/model/priority_overflow.py',code/'src/model/cbc_early.py']:compile(p.read_text(),str(p),'exec')

def cases():
    # Fixed nominal established-state abundance isolates movement rules first.
    for seed in [41,42]:
        for mode in ['veit','shared','adult_priority']:
            yield f'{mode}_fixed_s{seed}',seed,mode,0.,False,False,False
    for seed in [41,42]:
        for env in [False,True]:
            yield f'early_{"envelope" if env else "survey"}_s{seed}',seed,'veit',0.,True,True,env
    for seed in [41,42]:
        for multiple in [1.25,4.]:
            yield f'kfloor{str(multiple).replace(".","p")}_s{seed}',seed,'veit',multiple,False,False,False
    for seed in [41,42]:
        yield f'priority_floor4_s{seed}',seed,'adult_priority',4.,False,False,False

def stage():
    source=R/'runs'/PARENT
    for line in (source/'CODE_MANIFEST.sha256').read_text().splitlines():
        digest,rel=line.split('  ',1)
        if sha(source/'code'/rel)!=digest:raise ValueError('parent source changed')
    specs=[]
    for order,(suffix,seed,mode,floor,free,cbc,envelope) in enumerate(cases()):
        step='step39_'+suffix;dest=R/'runs'/step
        if dest.exists():raise FileExistsError(dest)
        dest.mkdir();shutil.copytree(source/'code',dest/'code',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        code=dest/'code';cfg=json.loads((code/f'config/overlays/grid54_{PARENT}.json').read_text());pop=cfg['population_model']
        if pop.get('native_departure',{}).get('enabled'):raise ValueError('unexpected native departure')
        if free:pop['pinned_sites'].pop('log_inv_total',None)
        else:pop['pinned_sites']['log_inv_total']=0. # inherited nominal 102.8 bird-equivalent convention
        n50=math.log1p(math.exp(pop['pinned_sites']['n50_raw']))
        pop['retention_overflow']={'mode':mode,'width_fraction':.05,'capacity_census':'pre-departure total A+J; inherited K units'}
        pop['capacity_floor_internal']=floor*n50/20.
        pop['cbc_likelihood'].update(enabled=cbc,family='early_detection',
            file=str(R/'inputs/cbc_early_20261102/cbc_early.npz'),minimum_state_year=1902,
            fixed_migrant_fraction=.35,fixed_young_availability=1.,envelope_enabled=envelope,envelope_weight=.1)
        for k in ['map','vis_map']:cfg['run_names'][k]='age_map_{precision}_grid54_'+step
        spec=dict(step=step,parent=PARENT,reference=PARENT,seed=seed,mode=mode,queue_priority=f'{605+order:04d}',
            initial_abundance='free' if free else 'fixed nominal multiplier 1 (102.8 bird-equivalent convention)',
            initialization_year=1951,cbc_enabled=cbc,envelope_enabled=envelope,capacity_floor_n50_multiple=floor,
            capacity_floor_internal=pop['capacity_floor_internal'],n50_route_scale=n50,
            native_departure=False,native_seed='Unfloored K_base inherited; additive floor changes final K only',
            floor_semantics='Small positive additive capacity floor after disease; does not force demographic viability',
            registered='OVERFLOW_EARLY_EXECUTION_20261102.md')
        cfg['_overflow_early']=spec;write(code/f'config/overlays/grid54_{step}.json',cfg);patch(code)
        shutil.copy2(source/'grid54_fit.sh',dest/'grid54_fit.sh')
        script=(source/'run_candidate.sh').read_text().replace(PARENT,step).replace('HOUFIN_MAP_SEED=41',f'HOUFIN_MAP_SEED={seed}')
        # All-years NB count scoring is invalid for the new binary family.
        script='\n'.join(l for l in script.splitlines() if 'score_cbc_joint.py' not in l)+'\n'
        (dest/'run_candidate.sh').write_text(script)
        write(dest/'RUN_SPEC.json',spec)
        (dest/'CODE_MANIFEST.sha256').write_text('\n'.join(sha(p)+'  ./'+p.relative_to(code).as_posix() for p in sorted(code.rglob('*')) if p.is_file())+'\n')
        specs.append(spec)
    write(R/'monitor/overflow_early_specs.json',specs)
    print('STAGED',len(specs),'not published until validation passes',flush=True)

def publish():
    if not (R/'monitor/overflow_early_VALIDATED.json').exists():raise ValueError('validation receipt required')
    for s in json.loads((R/'monitor/overflow_early_specs.json').read_text()):
        name=f"{s['queue_priority']}_{s['step']}.sh"
        if any((R/'queue'/state/name).exists() for state in ['pending','running','done','failed','held_early_design']):continue
        p=R/'queue/pending'/name;tmp=p.with_suffix('.tmp')
        tmp.write_text(f'#!/bin/bash\nset -euo pipefail\nexport GRID54_MEM_FRACTION=.85 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4\nexec {R.parents[1]}/venv/bin/python {R}/monitor/stage_overflow_early.py --execute {s["step"]}\n');os.rename(tmp,p)
    print('PUBLISHED',flush=True)

def execute(step):
    if step not in [s['step'] for s in json.loads((R/'monitor/overflow_early_specs.json').read_text())]:raise ValueError(step)
    run=R/'runs'/step
    with (run/'attempt.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        start=time.time();write(run/f'ATTEMPT_{time.time_ns()}.json',{'epoch':start,'slurm_job':os.environ.get('SLURM_JOB_ID'),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')})
        rc=subprocess.run(['timeout','--kill-after=30','10800','bash',str(run/'run_candidate.sh')]).returncode
        write(run/'RUN_RECEIPT.json',dict(exit_code=rc,elapsed_seconds=time.time()-start,started=start))
    raise SystemExit(rc)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute');p.add_argument('--publish',action='store_true');a=p.parse_args()
    execute(a.execute) if a.execute else publish() if a.publish else stage()
