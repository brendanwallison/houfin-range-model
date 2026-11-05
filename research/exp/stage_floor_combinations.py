"""Floored factorial plus matched reproductive-variation, settlement and native tests."""
import argparse,fcntl,json,math,os,shutil,subprocess,time
from pathlib import Path
from stage_overflow_early import replace,sha,write
R=Path('/work/07980/bwa386/ls6/houfin/research/grid54')


def patch_settlement(code):
    shutil.copy2(R/'monitor/local_arrival_settlement.py',code/'src/model/local_arrival_settlement.py')
    af=code/'src/model/age_forward.py'
    replace(af,'import jax.scipy.ndimage',"""import jax.scipy.ndimage
from src.config_utils import load_age_model_config as _local_config
from src.model.local_arrival_settlement import redistribute as local_redistribute
_LOCAL_SPEC=dict(_local_config()['population_model'].get('local_arrival_settlement') or {})
LOCAL_SETTLEMENT_FRACTION=float(_LOCAL_SPEC.get('fraction',0.))
LOCAL_SETTLEMENT=LOCAL_SETTLEMENT_FRACTION>0.
if not 0.<=LOCAL_SETTLEMENT_FRACTION<=1.:raise ValueError('invalid local placement fraction')""")
    replace(af,'    departure_probabilities_override=None,','    departure_probabilities_override=None,\n    local_settlement=None,')
    replace(af,'    N_a_post = adult_stayers + adult_arriving',
      """    _local = LOCAL_SETTLEMENT if local_settlement is None else local_settlement
    if _local:
        adult_arriving=local_redistribute(adult_arriving,K,LOCAL_SETTLEMENT_FRACTION)
    N_a_post = adult_stayers + adult_arriving""")
    replace(af,'    if return_adult_arrivals:       # settlement.apply_to = dispersers needs arrivals apart',
      """    if _local:
        juvenile_arriving=local_redistribute(juvenile_arriving,K,LOCAL_SETTLEMENT_FRACTION)
    if return_adult_arrivals:       # settlement.apply_to = dispersers needs arrivals apart""")


def cases():
    for seed in (41,42):
        for power,movement in [(1,'rational_adult20'),(2,'veit'),(2,'rational_adult20')]:
            parent=f'step41_rational_adult20_s{seed}' if power==1 else f'step42_power2_{movement}_s{seed}'
            yield seed,f'factor_p{power}_{movement}',parent,4.,'factorial'
        for family in ('reprovar','settlement','native'):
            for floor in (0.,4.):yield seed,f'{family}_floor{int(floor)}',f'step39_veit_fixed_s{seed}',floor,family


def stage():
    specs=[]
    for seed,label,parent,floor,family in cases():
        source=R/'runs'/parent
        for line in (source/'CODE_MANIFEST.sha256').read_text().splitlines():
            h,rel=line.split('  ',1)
            if sha(source/'code'/rel)!=h:raise ValueError('parent manifest mismatch')
        step=f'step43_{label}_s{seed}';run=R/'runs'/step
        if run.exists():raise FileExistsError(run)
        run.mkdir();shutil.copytree(source/'code',run/'code',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        code=run/'code';cfg=json.loads((code/f'config/overlays/grid54_{parent}.json').read_text());pop=cfg['population_model']
        assert pop['reproduction']['yearling_breeding'] and not pop['cbc_likelihood']['enabled']
        assert pop['pinned_sites']['log_inv_total']==0. and pop.get('capacity_floor_internal',0.)==0.
        assert not pop.get('native_departure',{}).get('enabled') and not pop.get('process_noise',{}).get('enabled')
        assert pop.get('settlement',{}).get('form','none')=='none'
        n50=math.log1p(math.exp(pop['pinned_sites']['n50_raw']))
        pop['capacity_floor_internal']=floor*n50/20.
        if family=='reprovar':
            pop['process_noise']=dict(enabled=True,target='fecundity',sigma=.1,n_freq=2,window_start=1966,window_end=2020,detrend_degree=1)
        elif family=='settlement':
            pop['local_arrival_settlement']=dict(fraction=.25,neighborhood='3x3 cell centers',max_search_km=math.sqrt(2)*54)
            patch_settlement(code)
        elif family=='native':pop['native_departure']=dict(enabled=True)
        reference=(f'step39_kfloor4p0_s{seed}' if floor else f'step39_veit_fixed_s{seed}')
        spec=dict(step=step,parent=parent,reference=reference,seed=seed,family=family,
                  queue_priority='0614' if family=='factorial' else '0615' if family=='reprovar' else '0616' if family=='settlement' else '0619',
                  capacity_floor_n50_multiple=floor,capacity_floor_internal=pop['capacity_floor_internal'],
                  reproduction=pop['reproduction'],gradual_departure=pop.get('gradual_departure'),
                  process_noise=pop.get('process_noise'),native_departure=pop.get('native_departure'),
                  local_arrival_settlement=pop.get('local_arrival_settlement'),initialization_year=1951,
                  registered='KFLOOR_COMBINATION_EXECUTION_20261102.md')
        cfg['_floor_combination']=spec
        for key in ('map','vis_map'):cfg['run_names'][key]='age_map_{precision}_grid54_'+step
        write(code/f'config/overlays/grid54_{step}.json',cfg)
        for p in [code/'src/model/age_forward.py']:compile(p.read_text(),str(p),'exec')
        shutil.copy2(source/'grid54_fit.sh',run/'grid54_fit.sh')
        (run/'run_candidate.sh').write_text((source/'run_candidate.sh').read_text().replace(parent,step))
        write(run/'RUN_SPEC.json',spec)
        (run/'CODE_MANIFEST.sha256').write_text('\n'.join(sha(p)+'  ./'+p.relative_to(code).as_posix() for p in sorted(code.rglob('*')) if p.is_file())+'\n')
        specs.append(spec)
    write(R/'monitor/floor_combination_specs.json',specs);print('STAGED',len(specs),flush=True)


def publish():
    v=json.loads((R/'monitor/floor_combination_VALIDATED.json').read_text());assert v['passed']
    specs=json.loads((R/'monitor/floor_combination_specs.json').read_text())
    for s in specs:
        run=R/'runs'/s['step'];assert sha(run/'CODE_MANIFEST.sha256')==v['manifests'][s['step']]
        for line in (run/'CODE_MANIFEST.sha256').read_text().splitlines():
            h,rel=line.split('  ',1)
            if sha(run/'code'/rel)!=h:raise ValueError('source drift')
        name=f"{s['queue_priority']}_{s['step']}.sh"
        if any((R/'queue'/state/name).exists() for state in ('pending','running','done','failed')):continue
        p=R/'queue/pending'/name;tmp=p.with_suffix('.tmp')
        tmp.write_text(f'#!/bin/bash\nset -euo pipefail\nexport GRID54_MEM_FRACTION=.85 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4\nexec {R.parents[1]}/venv/bin/python {R}/monitor/stage_floor_combinations.py --execute {s["step"]}\n')
        os.rename(tmp,p)
    print('PUBLISHED',len(specs),flush=True)


def finalize_staging():
    for s in json.loads((R/'monitor/floor_combination_specs.json').read_text()):
        run=R/'runs'/s['step'];code=run/'code'
        if any((R/'queue'/state/f"{s['queue_priority']}_{s['step']}.sh").exists() for state in ('pending','running','done','failed')):
            raise RuntimeError('Cannot edit a published run')
        if (run/'out/MAP_DONE').exists():raise RuntimeError('Run has already started')
        if s['family']=='settlement':shutil.copy2(R/'monitor/local_arrival_settlement.py',code/'src/model/local_arrival_settlement.py')
        audit=code/'research/floor_combination_audit';audit.mkdir(parents=True,exist_ok=True)
        for name in ('contribution_budget.py','run_contribution_audit.py'):
            shutil.copy2(R/'monitor/floor_combination_audit'/name,audit/name)
        script=run/'run_candidate.sh'
        script.write_text(script.read_text().replace(str(R/'monitor/run_contribution_audit.py'),str(audit/'run_contribution_audit.py')))
        (run/'CODE_MANIFEST.sha256').write_text('\n'.join(sha(p)+'  ./'+p.relative_to(code).as_posix() for p in sorted(code.rglob('*')) if p.is_file())+'\n')
    print('FINALIZED_STAGING',flush=True)


def execute(step):
    assert step in [s['step'] for s in json.loads((R/'monitor/floor_combination_specs.json').read_text())]
    run=R/'runs'/step
    with (run/'attempt.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.time()
        write(run/f'ATTEMPT_{time.time_ns()}.json',dict(epoch=start,slurm_job=os.environ.get('SLURM_JOB_ID'),gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))
        rc=subprocess.run(['timeout','--kill-after=30','10800','bash',str(run/'run_candidate.sh')]).returncode
        write(run/'RUN_RECEIPT.json',dict(exit_code=rc,elapsed_seconds=time.time()-start))
    raise SystemExit(rc)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--publish',action='store_true');p.add_argument('--execute');p.add_argument('--finalize-staging',action='store_true');a=p.parse_args()
    execute(a.execute) if a.execute else publish() if a.publish else finalize_staging() if a.finalize_staging else stage()
