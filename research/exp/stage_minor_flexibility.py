"""Immutable field-decoupling and cell-year abundance tests; validate before publish."""
import argparse,fcntl,json,math,os,shutil,subprocess,time
from pathlib import Path
from stage_overflow_early import replace,sha,write
R=Path('/work/07980/bwa386/ls6/houfin/research/grid54')


def patch_fields(code,tau):
    p=code/'src/model/age_priors.py'
    old='''    w_env = numpyro.deterministic(
        "w_env", w_scale[None, :] * (jnp.sqrt(rho_c) * f_shared + jnp.sqrt(1.0 - rho_c) * eps_idio))'''
    new=f'''    _tau_private={tau!r}
    if fixed is None or float(fixed)!=1.:raise ValueError('private K requires shared baseline')
    with numpyro.plate("capacity_private_features", M_features, dim=-2):
        _k_private=numpyro.sample("capacity_private",dist.Normal(0.,1.))
    _offset=jnp.zeros((M_features,4)).at[:,2].set(_tau_private*_k_private[:,0])
    w_env = numpyro.deterministic(
        "w_env",w_scale[None,:]*(f_shared+_offset))
    numpyro.deterministic("capacity_private_effective_scale",w_scale[2]*math.sqrt(1.+_tau_private**2))'''
    replace(p,old,new)
    replace(p,'    r = jnp.sqrt(rho_c) * jnp.ones(4)',
              f'    r = jnp.ones(4).at[2].set(1./math.sqrt(1.+{tau!r}**2))')


def patch_noise(code):
    p=code/'src/model/data_loading.py'
    replace(p,'    from src.data.combine.model_inputs import generate_spatial_basis\n    S = generate_spatial_basis',
      '''    if spec.get('form')=='cell_iid':
        from src.model.minor_flexibility import active_indices
        _, active=active_indices(meta['years'],int(spec['window_start']),int(spec['window_end']),
            (_observation_mask_spec() or {}).get('holdout_years') or ())
        return np.zeros((len(meta['land_rows']),0)),active
    from src.data.combine.model_inputs import generate_spatial_basis
    S = generate_spatial_basis''')
    replace(p,'        meta["pn_S"], meta["pn_active"] = S_pn.astype(f_type_cpu), active.astype(f_type_cpu)',
      '''        if pspec.get('unit_cell_variance') and pspec.get('form')!='cell_iid':
            from src.model.minor_flexibility import normalize_cell_variance
            S_pn=normalize_cell_variance(S_pn)
        meta["pn_S"], meta["pn_active"] = S_pn.astype(f_type_cpu), active.astype(f_type_cpu)
        if pspec.get('form')=='cell_iid':meta['pn_time_indices']=np.flatnonzero(active).astype(np.int32)''')
    p=code/'src/model/age_priors.py'
    old='''        u_raw = numpyro.sample("process_noise_u", dist.Normal(0.0, 1.0).expand(
            [data['pn_active'].shape[0], data['pn_S'].shape[-1]]))                          # (time, nb)
        u_act = u_raw * data['pn_active'][:, None]
        if "pn_P" in data:      # detrend_degree: no sustained regional trend in eps (stage 2: that is lambda(t)'s job)
            u_act = data['pn_P'] @ u_act
        process_noise = (float(_PN.get("sigma", 0.05)) * u_act) @ data['pn_S'].T'''
    new='''        from src.model.minor_flexibility import iid_log_innovations,regional_log_innovations
        if _PN.get('form')=='cell_iid':
            u_raw=numpyro.sample('process_noise_u',dist.Normal(0.,1.).expand(
                [data['pn_time_indices'].shape[0],data['land_rows'].shape[0]]))
            process_noise=iid_log_innovations(u_raw,data['pn_time_indices'],data['pn_active'].shape[0],float(_PN['sigma']),jnp)
        else:
            u_raw=numpyro.sample('process_noise_u',dist.Normal(0.,1.).expand(
                [data['pn_active'].shape[0],data['pn_S'].shape[-1]]))
            process_noise=regional_log_innovations(u_raw,data['pn_active'],data['pn_S'],float(_PN['sigma']))'''
    replace(p,old,new)
    shutil.copy2(R/'monitor/minor_flexibility.py',code/'src/model/minor_flexibility.py')


def cases():
    yield 43,'step44_shared_s43','control',None
    for seed in (41,42,43):
        for label,tau in [('kprivate10',.10),('kprivate25',.25),('allrho99',.99)]:
            yield seed,f'step44_{label}_s{seed}',label,tau
        for form in ['cell_iid','regional']:
            for label,sigma in [('s005',.005),('s010',.01),('s020',.02)]:
                yield seed,f'step45_{form}_{label}_s{seed}',form,sigma


def stage():
    source=R/'runs/step39_veit_fixed_s41'
    for line in (source/'CODE_MANIFEST.sha256').read_text().splitlines():
        h,rel=line.split('  ',1)
        if sha(source/'code'/rel)!=h:raise ValueError('parent source drift')
    specs=[]
    for seed,step,family,value in cases():
        run=R/'runs'/step
        if run.exists():raise FileExistsError(run)
        run.mkdir();shutil.copytree(source/'code',run/'code',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        code=run/'code';cfg=json.loads((code/'config/overlays/grid54_step39_veit_fixed_s41.json').read_text());pop=cfg['population_model']
        assert pop['reproduction']['yearling_breeding'] and not pop['cbc_likelihood']['enabled']
        assert not pop.get('process_noise',{}).get('enabled') and not pop.get('native_departure',{}).get('enabled')
        assert pop['manifold_prior']['form']=='exchangeable' and pop['manifold_prior']['fixed_coupling']==1.
        pop['capacity_floor_internal']=4.*math.log1p(math.exp(pop['pinned_sites']['n50_raw']))/20.
        if family.startswith('kprivate'):
            pop['capacity_private_relative_sd']=value
            patch_fields(code,value)
        elif family=='allrho99':pop['manifold_prior']['fixed_coupling']=value
        elif family in ['cell_iid','regional']:
            pop['process_noise']=dict(enabled=True,form=family,target='abundance',sigma=value,n_freq=2,
                                     window_start=1966,window_end=2020,unit_cell_variance=True)
            patch_noise(code)
            from noise_diagnostic_patch import patch as patch_noise_diagnostics
            diagnostic=code/'scripts/viz/map_diagnostics.py'
            diagnostic.write_text(patch_noise_diagnostics(diagnostic.read_text()))
        for key in ['map','vis_map']:cfg['run_names'][key]='age_map_{precision}_grid54_'+step
        reference=f'step39_kfloor4p0_s{seed}' if seed in [41,42] else 'step44_shared_s43'
        if family=='control':reference='step39_kfloor4p0_s41'
        spec=dict(step=step,seed=seed,parent=source.name,reference=reference,family=family,value=value,
                  queue_priority='0608' if family=='control' else '0609',registered='FLEXIBILITY_EXECUTION_20261103.md',
                  capacity_floor_internal=pop['capacity_floor_internal'],process_noise=pop.get('process_noise'),
                  field_prior_correlation=pop['manifold_prior']['fixed_coupling'],private_K_relative_SD=value if family.startswith('kprivate') else 0.)
        cfg['_minor_flexibility']=spec;write(code/f'config/overlays/grid54_{step}.json',cfg)
        # Freeze the process-aware attribution implementation with each new run.
        audit=code/'research/minor_flexibility_audit';audit.mkdir(parents=True,exist_ok=True)
        for name in ['contribution_budget.py','run_contribution_audit.py']:
            shutil.copy2(R/'monitor'/name,audit/name)
        shutil.copy2(source/'grid54_fit.sh',run/'grid54_fit.sh')
        runner=(source/'run_candidate.sh').read_text().replace(source.name,step)
        runner=runner.replace('HOUFIN_MAP_SEED=41',f'HOUFIN_MAP_SEED={seed}')
        runner=runner.replace(str(R/'monitor/run_contribution_audit.py'),str(audit/'run_contribution_audit.py'))
        (run/'run_candidate.sh').write_text(runner)
        write(run/'RUN_SPEC.json',spec)
        for p in [code/'src/model/age_priors.py',code/'src/model/data_loading.py',*audit.glob('*.py')]:
            compile(p.read_text(),str(p),'exec')
        (run/'CODE_MANIFEST.sha256').write_text('\n'.join(sha(p)+'  ./'+p.relative_to(code).as_posix() for p in sorted(code.rglob('*')) if p.is_file())+'\n')
        specs.append(spec)
    write(R/'monitor/minor_flexibility_specs.json',specs);print('STAGED',len(specs),flush=True)


def publish():
    v=json.loads((R/'monitor/minor_flexibility_VALIDATED.json').read_text());assert v['passed']
    specs=json.loads((R/'monitor/minor_flexibility_specs.json').read_text())
    for s in specs:
        run=R/'runs'/s['step'];assert sha(run/'CODE_MANIFEST.sha256')==v['manifests'][s['step']]
        for line in (run/'CODE_MANIFEST.sha256').read_text().splitlines():
            h,rel=line.split('  ',1)
            if sha(run/'code'/rel)!=h:raise ValueError('validated source drift')
        name=f"{s['queue_priority']}_{s['step']}.sh"
        if any((R/'queue'/state/name).exists() for state in ['pending','running','local_running','done','failed']):continue
        target=R/'queue/pending'/name;tmp=target.with_suffix('.tmp')
        tmp.write_text(f'#!/bin/bash\nset -euo pipefail\nexport GRID54_MEM_FRACTION=.85 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4\nexec {R.parents[1]}/venv/bin/python {R}/monitor/stage_minor_flexibility.py --execute {s["step"]}\n')
        tmp.replace(target)
    print('PUBLISHED',len(specs),flush=True)


def execute(step):
    assert step in [s['step'] for s in json.loads((R/'monitor/minor_flexibility_specs.json').read_text())]
    run=R/'runs'/step
    with (run/'attempt.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.time()
        write(run/f'ATTEMPT_{time.time_ns()}.json',dict(epoch=start,slurm_job=os.environ.get('SLURM_JOB_ID'),gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))
        rc=subprocess.run(['timeout','--kill-after=30','14400','bash',str(run/'run_candidate.sh')]).returncode
        write(run/'RUN_RECEIPT.json',dict(exit_code=rc,elapsed_seconds=time.time()-start))
    raise SystemExit(rc)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--publish',action='store_true');p.add_argument('--execute');a=p.parse_args()
    execute(a.execute) if a.execute else publish() if a.publish else stage()
