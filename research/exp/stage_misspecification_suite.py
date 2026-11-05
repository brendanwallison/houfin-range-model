"""Stage, validate and publish a varied TACC-only misspecification suite."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from stage_overflow_early import sha, write, replace

R=Path('/work/07980/bwa386/ls6/houfin/research/grid54')
TAG='misspecification_20261104'


def designs():
    # Paired optimizer seeds retain the two known demographic basins as controls.
    arms=[
      ('ctl_full',dict(kind='control',growth_fraction=1.)),
      ('ctl_weak',dict(kind='control',growth_fraction=.25)),
      ('ctl_frozen',dict(kind='control',growth_fraction=0.)),
      ('ab_ar05_weak',dict(kind='abundance',mode='regional_ar',rho=.9,sigma=.05,growth_fraction=.25)),
      ('ab_ar15_weak',dict(kind='abundance',mode='regional_ar',rho=.9,sigma=.15,growth_fraction=.25)),
      ('ab_ar15_frozen',dict(kind='abundance',mode='regional_ar',rho=.9,sigma=.15,growth_fraction=0.)),
      ('ab_blocks15_weak',dict(kind='abundance',mode='cell_blocks',sigma=.15,growth_fraction=.25)),
      ('ab_ar15_full',dict(kind='abundance',mode='regional_ar',rho=.9,sigma=.15,growth_fraction=1.)),
      ('est_static05_full',dict(kind='establishment',mode='cell_static',sigma=.5,growth_fraction=1.,clock=True)),
      ('est_static10_full',dict(kind='establishment',mode='cell_static',sigma=1.,growth_fraction=1.,clock=True)),
      ('est_static10_frozen',dict(kind='establishment',mode='cell_static',sigma=1.,growth_fraction=0.,clock=True)),
      ('arrival10_full',dict(kind='arrival',mode='cell_static',sigma=1.,growth_fraction=1.)),
      ('shoot10_weak',dict(kind='shooting',spacing=10,sigma=.15,growth_fraction=.25)),
      ('shoot10_wide_weak',dict(kind='shooting',spacing=10,sigma=.5,growth_fraction=.25)),
      ('shoot5_weak',dict(kind='shooting',spacing=5,sigma=.15,growth_fraction=.25)),
      ('hurdle_ctl',dict(kind='control',growth_fraction=1.,positive_count_weight=.25)),
      ('hurdle_ab',dict(kind='abundance',mode='regional_ar',rho=.9,sigma=.15,growth_fraction=.25,positive_count_weight=.25)),
      ('hurdle_est',dict(kind='establishment',mode='cell_static',sigma=1.,growth_fraction=1.,clock=True,positive_count_weight=.25)),
    ]
    for seed in [41,42]:
        for rank,(label,spec) in enumerate(arms):
            yield seed,rank,label,dict(spec,established_only=True)


def patch_audit(code):
    dest=code/'research/misspecification_audit';dest.mkdir(parents=True)
    for name in ['contribution_budget.py','run_contribution_audit.py']:
        shutil.copy2(R/'monitor'/name,dest/name)
    p=dest/'contribution_budget.py'
    replace(p,"        amp=grid(x['disease_m_amp'][t]);n=ap+js+ja", """        amp=grid(x['disease_m_amp'][t])
        # Settlement reweighting occurs AFTER density-dependent disease in the
        # actual forward. Undo only its frozen arrival weight for that exposure.
        ja_exposure = ja/jnp.maximum(grid(x['mm_arrival_weight'][t]),1e-30) if 'mm_arrival_weight' in x else ja
        n=ap+js+ja_exposure""")
    replace(p,'    def disease(t,sa,sj,ap,js,ja):', '''    original_prev_a, original_prev_j = prev_a, prev_j
    if 'mm_input_adult' in x:
        prev_a = jnp.zeros_like(prev_a).at[:,lr,lc].set(x['mm_input_adult'])
        prev_j = jnp.zeros_like(prev_j).at[:,lr,lc].set(x['mm_input_juvenile'])

    def disease(t,sa,sj,ap,js,ja):''')
    replace(p,'            return tuple(v*land for v in out)', '''            out = tuple(v*land for v in out)
            if 'mm_arrival_weight' in x:
                out = (out[0],out[1],out[2]*grid(x['mm_arrival_weight'][t]))
            return out''')
    replace(p,'        diag=(da*land,ds*land,dj*land)', '''        if 'mm_arrival_weight' in x:dj = dj*grid(x['mm_arrival_weight'][t])
        diag=(da*land,ds*land,dj*land)''')
    replace(p,'        observed_delta=an+jn-prev_a[t]-prev_j[t]', '''        node_adjustment = (prev_a[t]+prev_j[t])-(original_prev_a[t]+original_prev_j[t])
        rhs = rhs+node_adjustment
        observed_delta=an+jn-original_prev_a[t]-original_prev_j[t]''')
    replace(p,'            process_adjustment=process_adjustment,','            process_adjustment=process_adjustment,node_adjustment=node_adjustment,')
    p=dest/'run_contribution_audit.py'
    replace(p,'    x,states=captured[0];states=jax.tree.map', '''    x,states=captured[0]
    from src.model.misspecification_suite import audit_arguments
    base_x = dict(x)
    x = audit_arguments(ap._MM_LAST)
    states=jax.tree.map''')
    replace(p,'    arrays,setup=audit_forward(af,x,states)', '''    arrays,setup=audit_forward(af,x,states)
    arrays['potential_lambda_effective_establishment'] = arrays['potential_lambda'].copy()
    arrays['potential_lambda'] = np.asarray(base_x['Sa_flat']+base_x['Sj_flat']*base_x['Fmax_flat'])
    arrays['node_adjustment'] = np.asarray(arrays['node_adjustment'])''')
    replace(p,"    for reg in ['west','plains','broad_plains','east']:", '''    summary['misspecification'] = dict(spec=ap._MM_LAST['spec'],
        node_net_adjustment=float(arrays['node_adjustment'].sum()),
        node_gross_adjustment=float(np.abs(arrays['node_adjustment']).sum()),
        attribution='Ambient fitted gates and settlement weights frozen for linear origin attribution.',
        missing=['Dynamic-gate no-immigration replay', 'New-fit HMC geometry', 'Finite-propagule establishment'])
    for reg in []: # Dynamic discrepancy gates require a dedicated isolation replay.''')
    return dest


def stage():
    sys.path.insert(0,str(R/'monitor'))
    from range_watch import HostUnpickler
    with (R/'inputs/established1951/metadata.pkl').open('rb') as f:meta=HostUnpickler(f).load()
    years=np.asarray(meta['years']);rr=np.asarray(meta['land_rows']);cc=np.asarray(meta['land_cols'])
    # Eight smooth coordinate modes, normalized to unit marginal cell variance.
    x=(cc-cc.min())/max(np.ptp(cc),1);y=(rr-rr.min())/max(np.ptp(rr),1)
    basis=np.stack([fn(2*np.pi*k*z) for z in [x,y] for k in [1,2] for fn in [np.sin,np.cos]],axis=1)
    basis/=np.sqrt((basis*basis).sum(1,keepdims=True))
    active=((years>=1951)&(years<=2020)).astype(float)
    source=R/'runs/step39_veit_fixed_s41'
    for line in (source/'CODE_MANIFEST.sha256').read_text().splitlines():
        h,rel=line.split('  ',1);assert sha(source/'code'/rel)==h,'reference source drift'
    specs=[]
    for seed,rank,label,spec in designs():
        step=f'step48_{label}_s{seed}';run=R/'runs'/step
        if run.exists():raise FileExistsError(run)
        run.mkdir();shutil.copytree(source/'code',run/'code',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        code=run/'code';cfg=json.loads((code/'config/overlays/grid54_step39_veit_fixed_s41.json').read_text());pop=cfg['population_model']
        assert pop['reproduction']['yearling_breeding'] and not pop['cbc_likelihood']['enabled']
        assert not pop.get('process_noise',{}).get('enabled') and not pop.get('native_departure',{}).get('enabled')
        pop['capacity_floor_internal']=4.*math.log1p(math.exp(pop['pinned_sites']['n50_raw']))/20.
        spec.update(step=step,seed=seed,rank=rank,reference=f'step39_kfloor4p0_s{seed}',
                    calendar_start=1951,calendar_end=2020,growth_reference_year=1966,
                    log_bound=2.,centering='median-one; zero is exactly identity')
        arrays=dict(years=years,basis=basis,active=active)
        blocks=np.maximum(years-1951,0)//5;spec['n_blocks']=int(blocks.max())+1
        arrays['block_index']=blocks.astype(np.int32)
        if spec['kind']=='shooting':
            node_times=np.flatnonzero(np.isin(years,np.arange(1956,2021,spec['spacing']))).astype(np.int32)
            idx=np.full(len(years),-1,np.int32);idx[node_times]=np.arange(len(node_times))
            with np.load(R/f"runs/{spec['reference']}/out/contribution/annual_budget.npz") as z:
                center=np.stack([z['next_adult'][node_times-1],z['next_juvenile'][node_times-1]],axis=1)
            arrays.update(node_center=np.log(np.maximum(center,1e-20)),node_times=node_times,node_index=idx)
            spec['node_years']=years[node_times].tolist()
        np.savez_compressed(code/'src/model/misspecification_assets.npz',**arrays)
        shutil.copy2(R/'monitor/misspecification_suite.py',code/'src/model/misspecification_suite.py')
        p=code/'src/model/age_priors.py'
        alpha=spec['growth_fraction'];ref=int(np.flatnonzero(years==1966)[0])
        replace(p,'    # Save fields for viz',f'''    # Deliberately restricted historical growth, with K and Z_disp still real.
    _gfrac={alpha!r}
    if _gfrac!=1.:
        _logit=lambda v:jnp.log(jnp.maximum(v,1e-12))-jnp.log1p(-jnp.minimum(v,1.-1e-12))
        _mix=lambda v:(1.-_gfrac)*v[{ref}][None,:]+_gfrac*v
        _r=Sj_flat/jnp.maximum(Sa_flat,1e-12)
        Sa_flat=jnn.sigmoid(_mix(_logit(Sa_flat)))
        Sj_flat=Sa_flat*jnn.sigmoid(_mix(_logit(_r)))
        Fmax_flat=jnn.softplus(_mix(Fmax_flat+jnp.log(-jnp.expm1(-Fmax_flat))))

    # Save fields for viz''')
        p.write_text(p.read_text()+'''\n\n# Experiment-local wrapper, covered by the complete code manifest.
from src.model.misspecification_suite import install as _mm_install
_MM_SPEC = load_age_model_config()['_misspecification']
build_model_2d = _mm_install(build_model_2d, globals(), _MM_SPEC,
    __import__('pathlib').Path(__file__).with_name('misspecification_assets.npz'))
''')
        cfg['_misspecification']=spec
        for key in ['map','vis_map']:cfg['run_names'][key]='age_map_{precision}_grid54_'+step
        write(code/f'config/overlays/grid54_{step}.json',cfg)
        audit=patch_audit(code)
        shutil.copy2(source/'grid54_fit.sh',run/'grid54_fit.sh')
        runner=(source/'run_candidate.sh').read_text().replace(source.name,step).replace('HOUFIN_MAP_SEED=41',f'HOUFIN_MAP_SEED={seed}')
        runner=runner.replace(str(R/'monitor/run_contribution_audit.py'),str(audit/'run_contribution_audit.py'))
        # Dedicated full-data, nonzero-discrepancy gradient smoke precedes fitting.
        runner=runner.replace('if [ ! -f "$R/out/FIT_DONE" ]; then', '''if [ ! -f "$R/out/MISSPEC_SMOKE_DONE" ]; then
  export AGE_MODEL_CONFIG=$CODE/config/overlays/grid54_'''+step+'''.json DATA_CONFIG=$CODE/config/overlays/grid54_data.json
  export PYTHONPATH=$CODE:$CODE/src:'''+str(R/'monitor')+''' HOUFIN_MODEL_PRECISION=float64 HOUFIN_PROCESSED=$ROOT/processed
  "$ROOT/venv/bin/python" '''+str(R/'monitor/smoke_misspecification_suite.py')+''' > "$R/out/misspec_smoke.log" 2>&1
  date -Is > "$R/out/MISSPEC_SMOKE_DONE"
fi
if [ ! -f "$R/out/FIT_DONE" ]; then''')
        # Existing front summaries alone do not distinguish shooting anchors from dynamics.
        runner+='\nif [ ! -f "$R/out/misspecification_followup.json" ]; then "$PY" '+str(R/'monitor/misspecification_followup.py')+' "$REF" "$R/out" > "$R/out/misspecification_followup.log" 2>&1; fi\n'
        (run/'run_candidate.sh').write_text(runner);write(run/'RUN_SPEC.json',spec)
        for p in code.rglob('*.py'):compile(p.read_text(),str(p),'exec')
        (run/'CODE_MANIFEST.sha256').write_text('\n'.join(sha(p)+'  ./'+p.relative_to(code).as_posix() for p in sorted(code.rglob('*')) if p.is_file())+'\n')
        specs.append(spec)
    write(R/f'monitor/{TAG}_specs.json',specs)
    print('STAGED',len(specs),flush=True)


def publish():
    proof=json.loads((R/f'monitor/{TAG}_VALIDATED.json').read_text());assert proof['passed']
    specs=json.loads((R/f'monitor/{TAG}_specs.json').read_text())
    pilot_order=['ab_ar15_weak','est_static10_full','shoot10_weak','ctl_weak','ctl_full',
      'hurdle_ab','hurdle_est','hurdle_ctl','arrival10_full','ab_ar05_weak',
      'shoot5_weak','shoot10_wide_weak','est_static05_full','ab_blocks15_weak',
      'ctl_frozen','ab_ar15_frozen','est_static10_frozen','ab_ar15_full']
    for spec in specs:
        run=R/'runs'/spec['step'];assert sha(run/'CODE_MANIFEST.sha256')==proof['manifests'][spec['step']]
        # Smoke gate is part of each queue task, not a claim of full-data validation yet.
        label=spec['step'].removeprefix('step48_').rsplit('_s',1)[0]
        priority=pilot_order.index(label)+(0 if spec['seed']==41 else len(pilot_order))
        spec['queue_priority']=f'0601{priority:02d}'
        name=f"{spec['queue_priority']}_{spec['step']}.sh"
        if any((R/'queue'/s/name).exists() for s in ['pending','running','done','failed']):continue
        target=R/'queue/pending'/name;tmp=target.with_suffix('.tmp')
        tmp.write_text('#!/bin/bash\nset -euo pipefail\nexport GRID54_MEM_FRACTION=.85 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4\nexec '+str(R.parents[1]/'venv/bin/python')+' '+str(R/'monitor/stage_misspecification_suite.py')+' --execute '+spec['step']+'\n')
        tmp.replace(target)
    write(R/f'monitor/{TAG}_specs.json',specs)
    print('PUBLISHED',len(specs),flush=True)


def execute(step):
    assert step in [s['step'] for s in json.loads((R/f'monitor/{TAG}_specs.json').read_text())]
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('TACC allocation required')
    run=R/'runs'/step
    with (run/'attempt.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(run/f'ATTEMPT_{time.time_ns()}.json',dict(job=os.environ['SLURM_JOB_ID'],gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),epoch=time.time()))
        start=time.time();rc=subprocess.run(['bash',str(run/'run_candidate.sh')]).returncode
        write(run/'RUN_RECEIPT.json',dict(exit_code=rc,elapsed_seconds=time.time()-start))
    raise SystemExit(rc)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--publish',action='store_true');p.add_argument('--execute');a=p.parse_args()
    execute(a.execute) if a.execute else publish() if a.publish else stage()
