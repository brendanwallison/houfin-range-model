"""Immutable two-seed crowding-power x movement test; reuse power-one controls."""
import argparse,fcntl,json,os,shutil,subprocess,time
from pathlib import Path
from stage_overflow_early import replace,sha,write
R=Path('/work/07980/bwa386/ls6/houfin/research/grid54')


def stage():
    specs=[]
    for seed in (41,42):
        for movement in ('veit','rational_adult20'):
            parent=f'step39_veit_fixed_s{seed}' if movement=='veit' else f'step41_rational_adult20_s{seed}'
            source=R/'runs'/parent
            for line in (source/'CODE_MANIFEST.sha256').read_text().splitlines():
                h,rel=line.split('  ',1)
                if sha(source/'code'/rel)!=h:raise ValueError('parent source changed')
            step=f'step42_power2_{movement}_s{seed}';dest=R/'runs'/step
            if dest.exists():raise FileExistsError(dest)
            dest.mkdir();shutil.copytree(source/'code',dest/'code',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            code=dest/'code';cfg=json.loads((code/f'config/overlays/grid54_{parent}.json').read_text());pop=cfg['population_model']
            assert pop['reproduction'].get('yearling_breeding') and pop['reproduction'].get('form','standard')=='standard'
            assert not pop.get('subcell_strata',{}).get('enabled') and not pop.get('occupancy',{}).get('enabled')
            assert pop.get('settlement',{}).get('form','none') in ('none','off')
            assert not pop['cbc_likelihood']['enabled'] and not pop.get('native_departure',{}).get('enabled')
            assert pop.get('capacity_floor_internal',0.)==0. and pop['pinned_sites']['log_inv_total']==0.
            assert pop['retention_overflow']['mode']=='veit'
            pop['reproduction']['crowding_power']=2.
            spec=dict(step=step,parent=parent,reference=parent,seed=seed,movement=movement,crowding_power=2.,
                      queue_priority='0614',initialization_year=1951,initial_abundance='fixed nominal multiplier 1',
                      cbc_enabled=False,historical_supervision='inherited original pseudozeros',
                      capacity_floor_internal=0.,native_departure=False,
                      registered='CROWDING_CURVATURE_EXECUTION_20261102.md')
            cfg['_crowding_curvature_experiment']=spec
            for key in ('map','vis_map'):cfg['run_names'][key]='age_map_{precision}_grid54_'+step
            write(code/f'config/overlays/grid54_{step}.json',cfg)
            shutil.copy2(R/'monitor/density_regulation.py',code/'src/model/density_regulation.py')
            af=code/'src/model/age_forward.py';vm=code/'src/vis/age_model_math.py'
            replace(af,'import jax.scipy.ndimage','import jax.scipy.ndimage\nfrom src.model.density_regulation import crowding_argument')
            replace(af,'    F_eff = F_max / (1.0 + c * (N_total_post / K_safe))',
                       '    F_eff = F_max / (1.0 + c * crowding_argument(N_total_post / K_safe))')
            replace(vm,'from src.model.age_priors import equilibrium_age_quantities',
                       'from src.model.age_priors import equilibrium_age_quantities\nfrom src.model.density_regulation import crowding_argument')
            replace(vm,'            F = Fmax / (1.0 + c * u) * _mated(allee_gamma * N)',
                       '            F = Fmax / (1.0 + c * crowding_argument(u)) * _mated(allee_gamma * N)')
            for p in (af,vm,code/'src/model/density_regulation.py'):compile(p.read_text(),str(p),'exec')
            shutil.copy2(source/'grid54_fit.sh',dest/'grid54_fit.sh')
            # Parent runner already carries the proper cold-start seed. No warm checkpoint.
            (dest/'run_candidate.sh').write_text((source/'run_candidate.sh').read_text().replace(parent,step))
            write(dest/'RUN_SPEC.json',spec)
            (dest/'CODE_MANIFEST.sha256').write_text('\n'.join(sha(p)+'  ./'+p.relative_to(code).as_posix() for p in sorted(code.rglob('*')) if p.is_file())+'\n')
            specs.append(spec)
    write(R/'monitor/crowding_curvature_specs.json',specs);print('STAGED',len(specs))


def publish():
    validation=json.loads((R/'monitor/crowding_curvature_VALIDATED.json').read_text())
    specs=json.loads((R/'monitor/crowding_curvature_specs.json').read_text())
    assert validation['passed']
    for s in specs:
        run=R/'runs'/s['step']
        assert sha(run/'CODE_MANIFEST.sha256')==validation['manifests'][s['step']]
        for line in (run/'CODE_MANIFEST.sha256').read_text().splitlines():
            h,rel=line.split('  ',1)
            if sha(run/'code'/rel)!=h:raise ValueError('validated source changed')
        name=f"{s['queue_priority']}_{s['step']}.sh"
        if any((R/'queue'/state/name).exists() for state in ('pending','running','done','failed')):continue
        p=R/'queue/pending'/name;tmp=p.with_suffix('.tmp')
        tmp.write_text(f'#!/bin/bash\nset -euo pipefail\nexport GRID54_MEM_FRACTION=.85 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4\nexec {R.parents[1]}/venv/bin/python {R}/monitor/stage_crowding_curvature.py --execute {s["step"]}\n')
        os.rename(tmp,p)
    print('PUBLISHED',len(specs))


def execute(step):
    specs=json.loads((R/'monitor/crowding_curvature_specs.json').read_text())
    if step not in [s['step'] for s in specs]:raise ValueError(step)
    run=R/'runs'/step
    with (run/'attempt.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.time()
        write(run/f'ATTEMPT_{time.time_ns()}.json',dict(epoch=start,slurm_job=os.environ.get('SLURM_JOB_ID'),gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))
        rc=subprocess.run(['timeout','--kill-after=30','10800','bash',str(run/'run_candidate.sh')]).returncode
        write(run/'RUN_RECEIPT.json',dict(exit_code=rc,elapsed_seconds=time.time()-start))
    raise SystemExit(rc)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute');p.add_argument('--publish',action='store_true');a=p.parse_args()
    execute(a.execute) if a.execute else publish() if a.publish else stage()
