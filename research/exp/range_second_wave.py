"""Bounded, restart-safe second-wave dispatch on the existing GRID54 workers.

No model selection or allocation requests. All fits are cold, frozen-source
derivatives. A receipt charges failed and successful attempts alike.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
from datetime import datetime
from zoneinfo import ZoneInfo

try:
    from .range_watch import atomic_json, read_json
except ImportError:
    from range_watch import atomic_json, read_json

BASE = 'step33_nestcbc_s41'
ABS = 'step33_nestcbc_abs_s41'
BUDGET_HOURS = 36
VARIANTS = {
    'abs_s42': (ABS, 42, None, None),
    'foundlo': (BASE, 41, -1.79, None),
    'foundhi': (BASE, 41, 1.23, None),
    'broodlo': (BASE, 41, None, 2.0),
    'broodhi': (BASE, 41, None, 3.125),
    'fl_bl': (BASE, 41, -1.79, 2.0),
    'fl_bh': (BASE, 41, -1.79, 3.125),
    'fh_bl': (BASE, 41, 1.23, 2.0),
    'fh_bh': (BASE, 41, 1.23, 3.125),
    'seed43': (BASE, 43, None, None),
    'seed44': (BASE, 44, None, None),
}
# Each new arm changes one named assumption or movement term on a fixed parent.
PROFILES = {}
for law, parent in (('rel', BASE), ('abs', ABS)):
    for mechanism in ('comp', 'hazard', 'v2', 'comp_natdep', 'hazard_natdep', 'v2_hazard', 'v2_natdep'):
        for seed in (41, 42):
            key = f'{law}_{mechanism}_s{seed}'
            VARIANTS[key] = (parent, seed, None, None)
            PROFILES[key] = mechanism
for seed in (41, 42):
    key = f'abs_natdep_s{seed}'
    VARIANTS[key] = (ABS, seed, None, None)
    PROFILES[key] = 'natdep'
for key, founder, broods in (('abs_foundlo',-1.79,None),('abs_foundhi',1.23,None),
                            ('abs_broodlo',None,2.0),('abs_broodhi',None,3.125)):
    VARIANTS[key] = (ABS, 41, founder, broods)
    PROFILES[key] = 'assumption'
DEADLINE = datetime(2026, 11, 1, 10, tzinfo=ZoneInfo('America/Chicago')).timestamp()


def v2_contract(root, config):
    """Same surveys/grid/dynamics; only feature provenance may differ."""
    import pickle
    import numpy as np
    target = root/'runs/step3_native_v2/code/config/overlays/grid54_step3_native_v2.json'
    v2 = read_json(target)
    def expanded(p):
        return Path(p.replace('${HOUFIN_PROCESSED}', str(root.parents[1]/'processed')))
    a = expanded(config['input_dir'])/'metadata.pkl'
    b = expanded(v2['input_dir'])/'metadata.pkl'
    with a.open('rb') as stream: left = pickle.load(stream)
    with b.open('rb') as stream: right = pickle.load(stream)
    def equal(x,y):
        if isinstance(x,dict):
            return isinstance(y,dict) and x.keys()==y.keys() and all(equal(x[k],y[k]) for k in x)
        if isinstance(x,(list,tuple)):
            return type(x)==type(y) and len(x)==len(y) and all(equal(i,j) for i,j in zip(x,y))
        try: return np.array_equal(x,y,equal_nan=True)
        except TypeError: return np.array_equal(x,y)
    allowed = {'z_kernel_contract','path_feature_meta','ingest_id','z_gathered_path','z_disp_gathered_path'}
    if left.keys()!=right.keys(): raise ValueError('v2 metadata schema mismatch')
    differences = [k for k in left if not equal(left[k],right[k])]
    forbidden = set(differences)-allowed
    if forbidden: raise ValueError(f'v2 survey/grid/dynamics mismatch: {sorted(forbidden)}')
    for k in ('kernel','centered','feature_prior','latent_dim','source_latent_dim','truncation'):
        if not equal(left['z_kernel_contract'].get(k),right['z_kernel_contract'].get(k)):
            raise ValueError(f'v2 feature contract mismatch: {k}')
    receipt = {'base_metadata_sha256':hashlib.sha256(a.read_bytes()).hexdigest(),
               'v2_metadata_sha256':hashlib.sha256(b.read_bytes()).hexdigest(),
               'differing_metadata_keys':differences,'unchanged_observations_grid_dynamics':True}
    return v2, receipt


def segments_ready(root, parent):
    audit = read_json(root/'overnight/native_segments_equivalence.json')
    if not audit or not audit.get('passed'): return False
    cfg = read_json(root/'runs'/parent/'code/config/overlays'/f'grid54_{parent}.json')
    if not cfg: return False
    p = Path(cfg['input_dir'].replace('${HOUFIN_PROCESSED}',str(root.parents[1]/'processed')))
    meta = read_json(p/'z_disp_seg4_meta.json')
    if not meta or meta.get('n_seg')!=4: return False
    shape = meta.get('shape',[])
    if len(shape)!=5 or meta.get('reproduction_vs_stored_Zdisp',{}).get('years')!=shape[0]: return False
    if meta['reproduction_vs_stored_Zdisp'].get('max_median_rel',float('inf'))>1e-5: return False
    data = p/'z_disp_seg4_gathered.dat'
    return data.exists() and data.stat().st_size == math.prod(shape)*4


def charged_hours(root, ledger):
    total = 0.0
    for step in ledger['staged']:
        receipt = read_json(root/'runs'/step/'RUN_RECEIPT.json')
        total += receipt['elapsed_seconds']/3600 if receipt else 2.0
    return total


def features(root, step):
    out = root/'runs'/step/'out'
    p = read_json(out/'pathology.json')
    f = read_json(out/'front_speed.json')
    if not (out/'FIT_DONE').exists() or not p or p.get('status') != 'complete' or not f:
        return None
    m = p['regions']['all_land']['modern']
    result = [m['Sa']['population']['q10_50_90'][1],
              p['nests']['nest_F_real']['mean'],
              m['juvenile_ratio_above_095_population_share'],
              f['fitted']['10pct']['median_model_minus_observed_yr']]
    return result if all(v is not None and math.isfinite(v) for v in result) else None


def sensitive(a, b, basin=False):
    return (abs(a[0]-b[0]) > .05 or
            abs(a[1]-b[1])/max(abs(b[1]), 1e-12) > .20 or
            (not basin and abs(a[2]-b[2]) > .10) or abs(a[3]-b[3]) > 5)


def accepted(root, step):
    report = read_json(root.parents[1]/'processed/model_results'/
                       f'refined_float64__age_map_float32_grid54_{step}_quick90/refine_report.json')
    return bool((root/'runs'/step/'out/FIT_DONE').exists() and report and report.get('valid'))


def eligible(root):
    result = []
    # Rejected or malformed parent fits never seed an automatically published fit.
    if accepted(root, ABS) and features(root, ABS):
        result.append('abs_s42')
    a = features(root, BASE) if accepted(root, BASE) else None
    if a:
        result += ['foundlo', 'foundhi', 'broodlo', 'broodhi']
        # Explicit names avoid the optional absolute replicate changing this gate.
        tests = [features(root, 'step34_'+v) if accepted(root, 'step34_'+v) else None
                 for v in ('foundlo','foundhi','broodlo','broodhi')]
        if all(t is not None for t in tests) and any(sensitive(t, a) for t in tests):
            result += ['fl_bl', 'fl_bh', 'fh_bl', 'fh_bh']
        b = features(root, 'step33_nestcbc_s42')
        if b and accepted(root, 'step33_nestcbc_s42') and sensitive(b, a, basin=True):
            result += ['seed43', 'seed44']
    # Paired cold fits are warranted by the existing historical pathologies,
    # regardless of whether one particular first-wave seed looks favorable.
    for variant, profile in PROFILES.items():
        parent = VARIANTS[variant][0]
        if accepted(root,parent) and features(root,parent):
            if not profile.startswith('comp') or segments_ready(root,parent):
                result.append(variant)
    return result


def stage(root, variant):
    parent, seed, founder, broods = VARIANTS[variant]
    step = 'step34_'+variant
    run = root/'runs'/step
    spec = {'step': step, 'parent': parent, 'seed': seed, 'founder': founder,
            'broods': broods, 'registration': 'SECOND_WAVE_CONTROLLER_20261031.md'}
    profile = PROFILES.get(variant)
    if profile: spec['profile'] = profile
    if run.exists():
        old = read_json(run/'RUN_SPEC.json')
        if not old or any(old.get(k) != v for k,v in spec.items()):
            raise ValueError(f'conflicting or partial immutable run: {step}')
    else:
        source = root/'runs'/parent
        v2 = receipt = None
        if profile and profile.startswith('v2'):
            v2, receipt = v2_contract(root, read_json(source/'code/config/overlays'/f'grid54_{parent}.json'))
        # Verify parent source before copying; do not silently freeze modified code.
        for line in (source/'CODE_MANIFEST.sha256').read_text().splitlines():
            sha, relative = line.split('  ', 1)
            if hashlib.sha256((source/'code'/relative).read_bytes()).hexdigest() != sha:
                raise ValueError(f'parent source mismatch: {relative}')
        temporary = root/'runs'/('.staging_'+step)
        if temporary.exists():
            raise ValueError(f'partial staging retained for inspection: {temporary}')
        temporary.mkdir()
        shutil.copytree(source/'code', temporary/'code', ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        code = temporary/'code'
        config = read_json(code/'config/overlays'/f'grid54_{parent}.json')
        pop = config['population_model']
        if founder is not None: pop['pinned_sites']['log_inv_total'] = founder
        if broods is not None:
            values = pop['nest_likelihood']['broods']['values']
            pop['nest_likelihood']['broods']['values'] = [broods]*len(values)
        if profile and profile.startswith('comp'):
            pop['path_compounding'] = {'enabled':True,'n_seg':4,'reference':'endpoint',
                                       'lambda_scale':.25,'length_km':100}
        elif profile and 'hazard' in profile:
            pop['journey'] = {'distance_hazard':True,'band_mean_km':[80.,286.,867.],'hazard_scale':.05}
        if profile and 'natdep' in profile:
            pop['native_departure'] = {'enabled':True,'beta_a':1.5,'beta_b':1.5}
        if profile and profile.startswith('v2'):
            for key in ('input_dir','raw_z_dir'): config[key] = v2[key]
            atomic_json(temporary/'INPUT_CONTRACT.json',receipt)
        for key in ('map','vis_map'):
            config['run_names'][key] = 'age_map_{precision}_grid54_'+step
        config['_second_wave'] = spec
        overlay = code/'config/overlays'/f'grid54_{step}.json'
        atomic_json(overlay, config)
        spec['config_sha256'] = hashlib.sha256(overlay.read_bytes()).hexdigest()
        shutil.copy2(source/'grid54_fit.sh', temporary/'grid54_fit.sh')
        script = (source/'run_candidate.sh').read_text().replace(parent, step)
        script = script.replace('HOUFIN_MAP_SEED=41', f'HOUFIN_MAP_SEED={seed}')
        (temporary/'run_candidate.sh').write_text(script)
        manifest = [hashlib.sha256(p.read_bytes()).hexdigest()+'  ./'+p.relative_to(code).as_posix()
                    for p in sorted(code.rglob('*')) if p.is_file()]
        (temporary/'CODE_MANIFEST.sha256').write_text('\n'.join(manifest)+'\n')
        atomic_json(temporary/'RUN_SPEC.json', spec)
        os.rename(temporary, run)
    name = f'0600_{step}.sh'
    if not any((root/'queue'/state/name).exists() for state in ('pending','running','done','failed')):
        # Python supervises timeout and always records elapsed time, including failures.
        wrapper = f'''#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 GRID54_MEM_FRACTION=.85
exec {root.parents[1]}/venv/bin/python {root}/monitor/range_second_wave.py --root {root} --execute {step}
'''
        tmp = root/'queue/pending'/('.'+name)
        tmp.write_text(wrapper)
        os.rename(tmp, root/'queue/pending'/name)
    return spec


def tick(root, ledger, now=None):
    now = time.time() if now is None else now
    ledger['staging_deadline'] = datetime.fromtimestamp(DEADLINE, ZoneInfo('America/Chicago')).isoformat()
    ledger['budget_gpu_hours'] = BUDGET_HOURS
    ledger['waiting_inputs'] = [v for v,p in PROFILES.items() if p.startswith('comp')
                                and not segments_ready(root,VARIANTS[v][0])]
    if now >= DEADLINE or (root/'monitor/STOP_SECOND_WAVE').exists():
        ledger['state'] = 'staging stopped'
        return
    ledger['state'] = 'watching registered gates'
    for variant in eligible(root):
        step = 'step34_'+variant
        if step in ledger['staged']: continue
        # Existing immutable specs recover a crash between publication and ledger write.
        recovering = (root/'runs'/step/'RUN_SPEC.json').exists()
        if not recovering and charged_hours(root, ledger)+2 > BUDGET_HOURS:
            ledger.setdefault('deferred', {})[step] = f'{BUDGET_HOURS} GPU-hour reservation cap'
            continue
        try:
            ledger['staged'][step] = stage(root, variant)
        except Exception as exc:
            ledger.setdefault('blocked',{})[step] = str(exc)
            continue
        ledger.setdefault('blocked',{}).pop(step,None)
        ledger.setdefault('deferred',{}).pop(step,None)
        atomic_json(root/'monitor/second_wave_ledger.json', ledger)
    ledger['charged_gpu_hours'] = charged_hours(root, ledger)
    ledger['epoch'] = now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--execute', choices=['step34_'+v for v in VARIANTS])
    args = parser.parse_args()
    root = args.root.resolve()
    if args.execute:
        import subprocess
        run = root/'runs'/args.execute
        started = time.time()
        with (run/'attempt.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            if (run/'ATTEMPT_STARTED.json').exists():
                raise RuntimeError('attempt already started; automatic retry forbidden')
            atomic_json(run/'ATTEMPT_STARTED.json', {'epoch': started})
            try:
                rc = subprocess.run(['timeout','--kill-after=30','7200','bash',str(run/'run_candidate.sh')]).returncode
            except Exception:
                rc = 125
            atomic_json(run/'RUN_RECEIPT.json', {'exit_code': rc, 'started': started,
                        'elapsed_seconds': time.time()-started})
        raise SystemExit(rc)
    monitor = root/'monitor'
    with (monitor/'second_wave.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        (monitor/'second_wave.pid').write_text(str(os.getpid()))
        while True:
            ledger = read_json(monitor/'second_wave_ledger.json') or {'staged': {}}
            try:
                tick(root, ledger)
                ledger.pop('error', None)
            except Exception as exc:
                ledger['error'] = str(exc)
                ledger['epoch'] = time.time()
            atomic_json(monitor/'second_wave_ledger.json', ledger)
            if args.once: break
            time.sleep(60)


if __name__ == '__main__': main()
