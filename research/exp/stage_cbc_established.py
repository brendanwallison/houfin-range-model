"""Immutable later-state/CBC refits for existing shared A100 workers."""
import argparse,copy,fcntl,hashlib,json,math,os,shutil,subprocess,time
from pathlib import Path

R=Path('/work/07980/bwa386/ls6/houfin/research/grid54');PARENT='step35_est1951_ctl_mid_s41'
CASES=[
 ('free_s41',41,'free',300,1.,False,True),('mid_s41',41,560,300,1.,False,True),
 ('free_s42',42,'free',300,1.,False,True),('mid_s42',42,560,300,1.,False,True),
 ('noncbc_free_s41',41,'free',300,1.,False,False),('noncbc_free_s42',42,'free',300,1.,False,False),
 ('lo_s41',41,280,300,1.,False,True),('hi_s41',41,1120,300,1.,False,True),
 ('sw600_free_s41',41,'free',600,1.,False,True),('circle2_free_s41',41,'free',300,2.,False,True),
 ('broods_free_s41',41,'free',300,1.,True,True),('free_s43',43,'free',300,1.,False,True),
 ('lo_s42',42,280,300,1.,False,True),('hi_s42',42,1120,300,1.,False,True),
 ('sw600_free_s42',42,'free',600,1.,False,True),('circle2_free_s42',42,'free',300,2.,False,True),
 ('broods_free_s42',42,'free',300,1.,True,True),('sw600_mid_s41',41,560,600,1.,False,True),
 ('sw600_mid_s42',42,560,600,1.,False,True),('mid_s43',43,560,300,1.,False,True)]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')

def stage():
    source=R/'runs'/PARENT
    for line in (source/'CODE_MANIFEST.sha256').read_text().splitlines():
        digest,rel=line.split('  ',1)
        if sha(source/'code'/rel)!=digest:raise ValueError('parent changed '+rel)
    module=R/'monitor/cbc_joint.py';specs=[]
    for order,(suffix,seed,abundance,distance,circle_sd,broods,cbc) in enumerate(CASES):
        step='step38_cbc1951_'+suffix;dest=R/'runs'/step
        if dest.exists():raise FileExistsError(dest)
        dest.mkdir();shutil.copytree(source/'code',dest/'code',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        code=dest/'code';cfg=json.loads((code/f'config/overlays/grid54_{PARENT}.json').read_text());pop=cfg['population_model']
        if pop.get('native_departure',{}).get('enabled'):raise ValueError('native departure is not a primary branch')
        if abundance=='free':pop['pinned_sites'].pop('log_inv_total',None)
        else:pop['pinned_sites']['log_inv_total']=math.log(abundance/102.8)
        pop['invasion_pulse_prior'].update(total_median=560/102.8,total_log_sd=2.)
        pop['release_allee_exemption_years']=0
        if broods:pop['nest_likelihood']['broods'].update(form='lognormal',log_sd=.3)
        spec={'step':step,'parent':PARENT,'seed':seed,'initialization_year':1951,'initial_abundance':abundance,
          'cbc_enabled':cbc,'cbc_distance_scenario_km':distance,'cbc_circle_sd':circle_sd,
          'nest_annual_conversion_uncertain':broods,'native_departure':False,'queue_priority':f'{615+order:04d}',
          'order':order,'cbc_input_sha256':sha(R/f'inputs/cbc_joint_20261101_v2/cbc_joint_{distance}km.npz'),
          'cbc_module_sha256':sha(module),'no_historical_backcast_pin':True,'release_allee_exemption_years':0,
          'prior_status':'Weak reference for free log abundance; no independent census evidence or historical CBC backcast reused',
          'seasonal_status':'Observation-only conservative SW mixture with fixed 300/600km sensitivity; estimated migrant fraction and young availability',
          'age_split':'Inherited 50/50 adult/young; no subcell establishment state inferred',
          'reference':PARENT,'registered':'CBC_ESTABLISHED_OVERNIGHT_20261101.md'}
        pop['cbc_likelihood']={'enabled':cbc,'file':str(R/f'inputs/cbc_joint_20261101_v2/cbc_joint_{distance}km.npz'),
          'circle_sd':circle_sd,'minimum_state_year':1951,'distance_scenario_km':distance}
        cfg['_later_established_state']=copy.deepcopy(spec);cfg['_cbc_established']=copy.deepcopy(spec)
        for k in ['map','vis_map']:cfg['run_names'][k]='age_map_{precision}_grid54_'+step
        write(code/f'config/overlays/grid54_{step}.json',cfg)
        shutil.copy2(module,code/'src/model/cbc_joint.py')
        loader=code/'src/model/data_loading.py';text=loader.read_text();anchor='    residency = os.environ.get("HOUFIN_MODEL_INPUT_RESIDENCY", "device").lower()'
        block='''    # Isolated joint-CBC branch; host arrays go through the standard dtype/device conversion.
    _cbc = dict((_lamc().get("population_model", {}) or {}).get("cbc_likelihood") or {})
    if _cbc.get("enabled"):
        with np.load(_cbc["file"]) as _z:
            meta.update({k: np.asarray(_z[k]) for k in _z.files})
        meta["cbc_n_circles"] = int(meta["cbc_w_local"].shape[0])
        _cbc_years = np.asarray(meta["years"])[meta["cbc_t"]]
        if (_cbc_years < int(_cbc["minimum_state_year"])).any():
            raise ValueError("pre-initialization CBC likelihood would be invalid")
'''
        if text.count(anchor)!=1:raise ValueError('loader patch anchor ambiguous')
        loader.write_text(text.replace(anchor,block+anchor))
        model=code/'src/model/age_priors.py';text=model.read_text()
        if not text.endswith('            numpyro.sample("obs", obs_dist, obs=data["observed_results"])\n'):raise ValueError('model tail changed')
        model.write_text(text+'''\n    # Source-specific winter count likelihood. Seasonal operator has no demographic feedback.
    _cbc = dict(_POP_SPEC.get("cbc_likelihood") or {})
    if _cbc.get("enabled"):
        from src.model.cbc_joint import cbc_observe
        cbc_observe(data, Na_grid, Nj_grid, data["pop_scalar"], _cbc)
''')
        for p in [loader,model,code/'src/model/cbc_joint.py']:compile(p.read_text(),str(p),'exec')
        shutil.copy2(source/'grid54_fit.sh',dest/'grid54_fit.sh')
        script=f'''#!/bin/bash
set -euo pipefail
ROOT={R.parents[1]}
R={dest}
CODE=$R/code
source "$ROOT/age-research/scripts/tacc/env.sh"
export HOUFIN_MAP_SEED={seed} HOUFIN_Z_FLOAT32=1 PYTHONUNBUFFERED=1
unset HOUFIN_SYNTHETIC_OBS
mkdir -p "$R/out"
if [ ! -f "$R/out/FIT_DONE" ]; then bash "$R/grid54_fit.sh" {step} grid54_{step}.json; fi
cd "$CODE"
export AGE_MODEL_CONFIG=$CODE/config/overlays/grid54_{step}.json DATA_CONFIG=$CODE/config/overlays/grid54_data.json
export PYTHONPATH=$CODE:$CODE/src:{R}/monitor HOUFIN_MODEL_PRECISION=float64 HOUFIN_PROCESSED=$ROOT/processed
PY=$ROOT/venv/bin/python
REF=$ROOT/processed/model_results/refined_float64__age_map_float32_grid54_{step}_quick90
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
if [ ! -f "$R/out/pathology.json" ]; then "$PY" {R}/monitor/range_pathology_readout.py --device cpu --run "$REF" --out "$R/out/pathology.json" > "$R/out/pathology.log" 2>&1; fi
if [ ! -f "$R/out/front_speed.json" ]; then "$PY" {R}/monitor/grid54_front_speed.py "$REF" "$R/out/front_speed.json" > "$R/out/front_speed.log" 2>&1; fi
if [ ! -f "$R/out/contribution/READY.json" ]; then "$PY" {R}/monitor/run_contribution_audit.py --run "$REF" --out "$R/out/contribution" > "$R/out/contribution.log" 2>&1; fi
if [ ! -f "$R/out/cbc_score.json" ]; then "$PY" {R}/monitor/score_cbc_joint.py --run "$REF" --budget "$R/out/contribution/annual_budget.npz" --out "$R/out/cbc_score.json" > "$R/out/cbc_score.log" 2>&1; fi
'''
        (dest/'run_candidate.sh').write_text(script);write(dest/'RUN_SPEC.json',spec)
        files=[p for p in sorted(code.rglob('*')) if p.is_file()]
        (dest/'CODE_MANIFEST.sha256').write_text('\n'.join(sha(p)+'  ./'+p.relative_to(code).as_posix() for p in files)+'\n')
        specs.append(spec)
    # Publish only after the entire batch is frozen and source patches compile.
    for s in specs:
        p=R/'queue/pending'/f"{s['queue_priority']}_{s['step']}.sh";temp=p.with_suffix('.tmp')
        temp.write_text(f'#!/bin/bash\nset -euo pipefail\nexport GRID54_MEM_FRACTION=.85 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4\nexec {R.parents[1]}/venv/bin/python {R}/monitor/stage_cbc_established.py --execute {s["step"]}\n')
        os.rename(temp,p)
    write(R/'monitor/cbc_established_specs.json',specs)
    print('CBC RUNS QUEUED',len(specs),flush=True)

def execute(step):
    if step not in ['step38_cbc1951_'+x[0] for x in CASES]:raise ValueError(step)
    run=R/'runs'/step
    with (run/'attempt.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # MAP and refinement checkpoint themselves. Re-entry after allocation loss is allowed,
        # using the same immutable source and continuing the same optimizer schedule.
        write(run/f'ATTEMPT_{time.time_ns()}.json',{'epoch':time.time(),'slurm_job':os.environ.get('SLURM_JOB_ID')})
        start=time.time();rc=subprocess.run(['timeout','--kill-after=30','10800','bash',str(run/'run_candidate.sh')]).returncode
        write(run/'RUN_RECEIPT.json',{'exit_code':rc,'elapsed_seconds':time.time()-start,'started':start})
    raise SystemExit(rc)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute');a=p.parse_args()
    execute(a.execute) if a.execute else stage()
