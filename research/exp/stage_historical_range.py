"""Freeze and queue replacement historical range constraints (no early effort model)."""
import argparse,fcntl,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
from stage_overflow_early import replace
R=Path('/work/07980/bwa386/ls6/houfin/research/grid54');PARENT='step38_cbc1951_noncbc_free_s41'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')

def stage():
    source=R/'runs'/PARENT
    for line in (source/'CODE_MANIFEST.sha256').read_text().splitlines():
        h,rel=line.split('  ',1)
        if sha(source/'code'/rel)!=h:raise ValueError('parent source changed')
    specs=[]
    for seed in [41,42]:
        for mode in ['historical','invasion']:
            step=f'step40_rangev2_{mode}_s{seed}';dest=R/'runs'/step
            if dest.exists():raise FileExistsError(dest)
            dest.mkdir();shutil.copytree(source/'code',dest/'code',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            code=dest/'code';cfg=json.loads((code/f'config/overlays/grid54_{PARENT}.json').read_text());pop=cfg['population_model']
            pop['cbc_likelihood']['enabled']=False
            pop['pinned_sites'].pop('log_inv_total',None)
            receipt=json.loads((R/f'inputs/historical_range_v2_20261102/{mode}_RECEIPT.json').read_text())
            path=R/f'inputs/historical_range_v2_20261102/{mode}.npz'
            if sha(path)!=receipt['sha256']:raise ValueError('range input hash mismatch')
            pop['historical_range']=dict(enabled=True,file=str(path),fixed_concentration=receipt['fixed_concentration'],
                mode=mode,calibration='Fixed reference NB2; no learned CBC scale/effort/circle effects',input_sha256=sha(path))
            spec=dict(step=step,seed=seed,parent=PARENT,reference=PARENT,mode=mode,queue_priority=str(611+len(specs)).zfill(4),
                cbc_enabled=False,initial_abundance='free',initialization_year=1951,
                removed_old_pseudozero_rows=receipt['old_pseudozeros_removed'],range_input_sha256=sha(path),
                fixed_concentration=receipt['fixed_concentration'],registered='HISTORICAL_RANGE_IMPLEMENTATION_20261102.md')
            cfg['_historical_range_experiment']=spec
            for key in ['map','vis_map']:cfg['run_names'][key]='age_map_{precision}_grid54_'+step
            write(code/f'config/overlays/grid54_{step}.json',cfg)
            shutil.copy2(R/'monitor/historical_range.py',code/'src/model/historical_range.py')
            loader=code/'src/model/data_loading.py'
            replace(loader,'    residency = os.environ.get("HOUFIN_MODEL_INPUT_RESIDENCY", "device").lower()',
              '''    _hr=dict((_lamc().get('population_model',{}) or {}).get('historical_range') or {})
    if _hr.get('enabled'):
        with np.load(_hr['file']) as _z:
            meta.update({k:np.asarray(_z[k]) for k in _z.files})
        _old=meta['hr_old_zero_mask'].astype(bool)
        if _old.shape!=meta['observed_results'].shape:raise ValueError('old pseudozero mask shape mismatch')
        if np.any(_old & ((meta['obs_route_idx']>=0) | (meta['observed_results']!=0))):raise ValueError('range replacement would mask real BBS observations')
        meta['obs_weight']=np.asarray(meta.get('obs_weight',np.ones_like(meta['observed_results'])),float)*(~_old)
    residency = os.environ.get("HOUFIN_MODEL_INPUT_RESIDENCY", "device").lower()''')
            model=code/'src/model/age_priors.py'
            model.write_text(model.read_text()+'''
    _hr=dict(_POP_SPEC.get('historical_range') or {})
    if _hr.get('enabled'):
        from src.model.historical_range import range_observe
        range_observe(data,Na_grid,Nj_grid,data['pop_scalar'],_hr)
''')
            for p in [loader,model,code/'src/model/historical_range.py']:compile(p.read_text(),str(p),'exec')
            shutil.copy2(source/'grid54_fit.sh',dest/'grid54_fit.sh')
            script=(source/'run_candidate.sh').read_text().replace(PARENT,step).replace('HOUFIN_MAP_SEED=41',f'HOUFIN_MAP_SEED={seed}')
            script='\n'.join(l for l in script.splitlines() if 'score_cbc_joint.py' not in l)+'\n'
            (dest/'run_candidate.sh').write_text(script);write(dest/'RUN_SPEC.json',spec)
            (dest/'CODE_MANIFEST.sha256').write_text('\n'.join(sha(p)+'  ./'+p.relative_to(code).as_posix() for p in sorted(code.rglob('*')) if p.is_file())+'\n')
            specs.append(spec)
    write(R/'monitor/historical_range_specs.json',specs);print('STAGED',len(specs),flush=True)

def publish():
    if not (R/'monitor/historical_range_VALIDATED.json').exists():raise ValueError('validation required')
    for s in json.loads((R/'monitor/historical_range_specs.json').read_text()):
        name=f"{s['queue_priority']}_{s['step']}.sh"
        if any((R/'queue'/state/name).exists() for state in ['pending','running','done','failed']):continue
        p=R/'queue/pending'/name;tmp=p.with_suffix('.tmp')
        tmp.write_text(f'#!/bin/bash\nset -euo pipefail\nexport GRID54_MEM_FRACTION=.85 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4\nexec {R.parents[1]}/venv/bin/python {R}/monitor/stage_historical_range.py --execute {s["step"]}\n');os.rename(tmp,p)
    print('PUBLISHED',flush=True)

def execute(step):
    if step not in [s['step'] for s in json.loads((R/'monitor/historical_range_specs.json').read_text())]:raise ValueError(step)
    run=R/'runs'/step
    with (run/'attempt.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        start=time.time();write(run/f'ATTEMPT_{time.time_ns()}.json',dict(epoch=start,slurm_job=os.environ.get('SLURM_JOB_ID'),gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))
        rc=subprocess.run(['timeout','--kill-after=30','10800','bash',str(run/'run_candidate.sh')]).returncode
        write(run/'RUN_RECEIPT.json',dict(exit_code=rc,elapsed_seconds=time.time()-start))
    raise SystemExit(rc)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute');p.add_argument('--publish',action='store_true');a=p.parse_args()
    execute(a.execute) if a.execute else publish() if a.publish else stage()
