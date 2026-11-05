"""Freeze six additional fits for the matched 2x2 departure comparison."""
import argparse, fcntl, json, os, shutil, subprocess, time
from pathlib import Path
from stage_overflow_early import replace, sha, write

R = Path('/work/07980/bwa386/ls6/houfin/research/grid54')
PARENT = 'step39_veit_fixed_s41'


def stage():
    source = R/'runs'/PARENT
    for line in (source/'CODE_MANIFEST.sha256').read_text().splitlines():
        h, rel = line.split('  ', 1)
        if sha(source/'code'/rel) != h:
            raise ValueError('parent manifest mismatch')
    specs = []
    for seed in (41, 42):
        for curve, rho in [('veit', .2), ('rational', 1.), ('rational', .2)]:
            step = f'step41_{curve}_{"shared" if rho == 1 else "adult20"}_s{seed}'
            dest = R/'runs'/step
            if dest.exists():
                raise FileExistsError(dest)
            dest.mkdir()
            shutil.copytree(source/'code', dest/'code', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            code = dest/'code'
            cfg = json.loads((code/f'config/overlays/grid54_{PARENT}.json').read_text())
            pop = cfg['population_model']
            assert not pop['cbc_likelihood']['enabled']
            assert pop['retention_overflow']['mode'] == 'veit'
            assert pop.get('capacity_floor_internal', 0.) == 0.
            assert pop['pinned_sites']['log_inv_total'] == 0.
            assert not pop.get('native_departure', {}).get('enabled')
            pop['gradual_departure'] = dict(enabled=True, curve=curve, adult_multiplier=rho)
            spec = dict(step=step, parent=PARENT, seed=seed, curve=curve, adult_multiplier=rho,
                        queue_priority='0614',
                        reference=f'step39_veit_fixed_s{seed}', initialization_year=1951,
                        initial_abundance='fixed nominal multiplier 1, inherited ~102.8 bird-equivalents',
                        historical_supervision='inherited pseudozeros; no CBC likelihood',
                        capacity_floor_internal=0., native_departure=False,
                        registered='GRADUAL_AGE_SPECIFIC_DEPARTURE_PROPOSAL_20261102.md')
            cfg['_gradual_departure_experiment'] = spec
            for key in ('map', 'vis_map'):
                cfg['run_names'][key] = 'age_map_{precision}_grid54_'+step
            write(code/f'config/overlays/grid54_{step}.json', cfg)
            shutil.copy2(R/'monitor/gradual_departure.py', code/'src/model/gradual_departure.py')
            forward = code/'src/model/age_forward.py'
            replace(forward, "OVERFLOW_ENABLED = _OVERFLOW.get('mode','veit') != 'veit'",
                    "from src.model.gradual_departure import departure_probabilities as gradual_probabilities\n_GRADUAL = dict(load_age_model_config()['population_model'].get('gradual_departure') or {})\nOVERFLOW_ENABLED = _GRADUAL.get('enabled',False) or _OVERFLOW.get('mode','veit') != 'veit'")
            replace(forward, 'def overflow_probabilities(a,j,k):\n    return departure_probabilities',
                    "def overflow_probabilities(a,j,k):\n    if _GRADUAL.get('enabled'):\n        return gradual_probabilities(a,j,k,_GRADUAL['curve'],float(_GRADUAL['adult_multiplier']))\n    return departure_probabilities")
            # Reuse the existing frozen-probability audit hook: contribution attribution
            # must hold the fitted probabilities fixed rather than recompute at tag density.
            compile(forward.read_text(), str(forward), 'exec')
            shutil.copy2(source/'grid54_fit.sh', dest/'grid54_fit.sh')
            script = (source/'run_candidate.sh').read_text().replace(PARENT, step).replace('HOUFIN_MAP_SEED=41', f'HOUFIN_MAP_SEED={seed}')
            (dest/'run_candidate.sh').write_text(script)
            write(dest/'RUN_SPEC.json', spec)
            (dest/'CODE_MANIFEST.sha256').write_text('\n'.join(sha(p)+'  ./'+p.relative_to(code).as_posix() for p in sorted(code.rglob('*')) if p.is_file())+'\n')
            specs.append(spec)
    write(R/'monitor/gradual_departure_specs.json', specs)
    print('STAGED', len(specs))


def publish():
    validation = json.loads((R/'monitor/gradual_departure_VALIDATED.json').read_text())
    specs = json.loads((R/'monitor/gradual_departure_specs.json').read_text())
    if not validation.get('passed'):
        raise ValueError('validation required')
    for s in specs:
        run = R/'runs'/s['step']
        if sha(run/'CODE_MANIFEST.sha256') != validation['manifests'][s['step']]:
            raise ValueError('unvalidated manifest')
        for line in (run/'CODE_MANIFEST.sha256').read_text().splitlines():
            h, rel = line.split('  ', 1)
            if sha(run/'code'/rel) != h:
                raise ValueError('validated source changed')
        name = f"{s['queue_priority']}_{s['step']}.sh"
        if any((R/'queue'/state/name).exists() for state in ('pending','running','done','failed')):
            continue
        p = R/'queue/pending'/name
        tmp = p.with_suffix('.tmp')
        tmp.write_text(f'#!/bin/bash\nset -euo pipefail\nexport GRID54_MEM_FRACTION=.85 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4\nexec {R.parents[1]}/venv/bin/python {R}/monitor/stage_gradual_departure.py --execute {s["step"]}\n')
        os.rename(tmp, p)
    print('PUBLISHED', len(specs))


def execute(step):
    specs = json.loads((R/'monitor/gradual_departure_specs.json').read_text())
    if step not in [s['step'] for s in specs]:
        raise ValueError(step)
    run = R/'runs'/step
    with (run/'attempt.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        start = time.time()
        write(run/f'ATTEMPT_{time.time_ns()}.json', dict(epoch=start, slurm_job=os.environ.get('SLURM_JOB_ID'), gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))
        rc = subprocess.run(['timeout','--kill-after=30','10800','bash',str(run/'run_candidate.sh')]).returncode
        write(run/'RUN_RECEIPT.json', dict(exit_code=rc, elapsed_seconds=time.time()-start))
    raise SystemExit(rc)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--execute')
    p.add_argument('--publish', action='store_true')
    a = p.parse_args()
    execute(a.execute) if a.execute else publish() if a.publish else stage()
