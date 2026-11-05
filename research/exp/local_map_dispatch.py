"""TACC-side atomic reservation and strict acceptance of registered local MAPs.

No JAX initialization and no Slurm submission. Existing immutable task scripts
return to the shared queue after the float32 checkpoint is accepted.
"""
import argparse
import fcntl
import hashlib
import json
import os
import re
import time
from importlib.metadata import version
from pathlib import Path

from accept_local_map import identity
from range_watch import HostUnpickler
from dispatch_order import dispatch_order

ROOT = Path('/work/07980/bwa386/ls6/houfin')
REGISTRIES = [('gradual_departure_specs.json', 'gradual_departure_VALIDATED.json'),
              ('crowding_curvature_specs.json', 'crowding_curvature_VALIDATED.json'),
              ('floor_combination_specs.json', 'floor_combination_VALIDATED.json'),
              ('minor_flexibility_specs.json', 'minor_flexibility_VALIDATED.json'),
              ('outstanding_transfer_specs.json','outstanding_transfer_VALIDATED.json')]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, data):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def registry(root):
    r = root / 'research/grid54'
    specs = {}
    for sf, vf in REGISTRIES:
        if not (r / 'monitor' / sf).exists() or not (r / 'monitor' / vf).exists():
            continue
        valid = json.loads((r / 'monitor' / vf).read_text())
        if not valid['passed']:
            raise ValueError('unvalidated registry')
        for s in json.loads((r / 'monitor' / sf).read_text()):
            expected = valid['manifests'][s['step']]
            if digest(r / 'runs' / s['step'] / 'CODE_MANIFEST.sha256') != expected:
                raise ValueError('registration manifest changed')
            specs[s['step']] = s
    return specs


def verify_source(run):
    for line in (run / 'CODE_MANIFEST.sha256').read_text().splitlines():
        expected, rel = line.split('  ', 1)
        path = (run / 'code' / rel).resolve()
        if not path.is_relative_to((run / 'code').resolve()) or digest(path) != expected:
            raise ValueError('frozen source mismatch')


def paths(root):
    r = root / 'research/grid54'
    q = r / 'queue'
    (q / 'local_running').mkdir(exist_ok=True)
    return r, q


def claim(root, token, excludes=(), peek=False, allow_minor=False, allow_transfer=False):
    r, q = paths(root)
    if (r/'monitor'/f'PAUSE_LOCAL_{token}').exists():return None
    specs = registry(root)
    eligible = sorted(((f"{s['queue_priority']}_{s['step']}.sh", s)
                      for s in specs.values() if s['step'] not in excludes
                      and (allow_transfer or not s['step'].startswith('step47_'))
                      and (allow_minor or not s['step'].startswith(('step44_','step45_','step46_','step47_')))),key=lambda pair:dispatch_order(pair[0]))
    with (q / 'local_offload.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # A restarted worker must recover its existing reservation first.
        for p in (q / 'local_running').glob('*.local3090.json'):
            if json.loads(p.read_text()).get('token') == token:
                return json.loads(p.read_text())
        for name, s in eligible:
            run = r / 'runs' / s['step']
            if not (q / 'pending' / name).exists():
                continue
            if (run / 'out/MAP_DONE').exists():
                continue  # Ready-for-refinement tasks belong to the A100s.
            destination = root / 'processed/model_results' / f"age_map_float32_grid54_{s['step']}_quick90"
            if destination.exists():
                continue  # Do not adopt an unexplained partial TACC attempt.
            verify_source(run)
            record = dict(step=s['step'], name=name, seed=s['seed'], token=token,
                          state='local_running', epoch=time.time(), heartbeat=time.time(),
                          manifest_sha256=digest(run / 'CODE_MANIFEST.sha256'))
            if peek:
                return record
            try:
                (q / 'pending' / name).rename(q / 'local_running' / name)
            except FileNotFoundError:
                continue
            write(q / 'local_running' / (name + '.local3090.json'), record)
            return record
    return None


def reservation(root, record):
    r, q = paths(root)
    path = q / 'local_running' / (record['name'] + '.local3090.json')
    current = json.loads(path.read_text())
    if current['token'] != record['token'] or current['step'] != record['step']:
        raise ValueError('reservation ownership differs')
    return r, q, path, current


def release(root, record, reason):
    r, q, path, current = reservation(root, record)
    name = current['name']
    if (q / 'pending' / name).exists():
        raise ValueError('duplicate pending task')
    current.update(state='returned_to_tacc', reason=reason, returned_epoch=time.time())
    write(r / 'runs' / current['step'] / f"LOCAL3090_{current['token']}.json", current)
    # Publish the same immutable script; its MAP_DONE guard skips accepted MAP.
    (q / 'local_running' / name).rename(q / 'pending' / name)
    path.unlink()
    return current


def accept(root, record, upload):
    import sys
    import numpy as np
    r, q, path, current = reservation(root, record)
    step = current['step']
    s = registry(root)[step]
    run = r / 'runs' / step
    verify_source(run)
    if digest(run / 'CODE_MANIFEST.sha256') != current['manifest_sha256']:
        raise ValueError('reservation source changed')
    destination = root / 'processed/model_results' / f'age_map_float32_grid54_{step}_quick90'
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt = destination / 'LOCAL_OFFLOAD_ACCEPTED.json'
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous['token'] != current['token']:
            raise ValueError('different accepted reservation')
        if digest(destination / 'map_checkpoint.pkl') != previous['checkpoint_sha256']:
            raise ValueError('accepted checkpoint changed')
    else:
        if destination.exists():
            raise ValueError('refusing to overwrite an existing MAP')
        if upload != r / 'monitor' / f"local3090_upload_{step}_{current['token']}":
            raise ValueError('unexpected upload directory')
        transfer = json.loads((upload / 'local_map_transfer.json').read_text())
        if transfer['seed'] != s['seed'] or transfer['token'] != current['token']:
            raise ValueError('seed or reservation differs')
        if transfer['code_manifest'] != current['manifest_sha256']:
            raise ValueError('transfer manifest differs')
        if digest(upload / 'map_checkpoint.pkl') != transfer['checkpoint_sha256']:
            raise ValueError('upload checksum differs')
        with (upload / 'map_checkpoint.pkl').open('rb') as stream:
            ck = HostUnpickler(stream).load()
        if ck.get('format_version') != 2 or ck.get('step') != 400 or len(ck.get('losses', [])) != 400:
            raise ValueError('incomplete MAP')
        if not np.isfinite(ck['losses']).all() or not ck.get('params'):
            raise ValueError('nonfinite or missing MAP')
        if any(not k.endswith('_auto_loc') or not np.isfinite(np.asarray(v)).all() for k, v in ck['params'].items()):
            raise ValueError('invalid MAP parameters')
        if digest(upload / 'map_params.pkl') != transfer['params_sha256']:
            raise ValueError('parameter checksum differs')
        with (upload / 'map_params.pkl').open('rb') as stream:
            params = HostUnpickler(stream).load()
        if set(params) != set(ck['params']) or any(not np.array_equal(np.asarray(params[k]), np.asarray(ck['params'][k])) for k in params):
            raise ValueError('parameter artifact differs from checkpoint')
        payload = ck['fingerprint_payload']
        canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'), default=list)
        if hashlib.sha256(canonical.encode()).hexdigest() != ck['fingerprint']:
            raise ValueError('invalid checkpoint fingerprint')
        os.environ['AGE_MODEL_CONFIG'] = str(run / 'code/config/overlays' / f'grid54_{step}.json')
        os.environ['HOUFIN_PROCESSED'] = str(root / 'processed')
        sys.path.insert(0, str(run / 'code'))
        from src.config_utils import load_age_model_config
        cfg = load_age_model_config()
        if payload['precision'] != 'float32' or payload['map_profile'] != 'quick90' or payload['age_model_config'] != cfg:
            raise ValueError('configuration/profile differs')
        profile = cfg['map']['profiles']['quick90']
        expected_schedule = [[int(i), float(v)] for i, v in profile['prior_relaxation']]
        expected = dict(prior_relaxation=expected_schedule, lr_decay_steps=int(profile['lr_decay_steps']),
                        init_lr=float(profile.get('init_lr', cfg['map']['init_lr'])),
                        weight_decay=float(profile.get('weight_decay', cfg['map'].get('weight_decay', 0.))),
                        input_residency=cfg.get('runtime', {}).get('input_residency', 'device'))
        if int(profile['target_steps']) != 400:
            raise ValueError('unsupported MAP target')
        for key, value in expected.items():
            actual = payload.get(key)
            if key == 'prior_relaxation' and actual is not None:
                actual = [[int(i), float(v)] for i, v in actual]
            if actual != value:
                raise ValueError(f'optimizer/input-residency setting differs: {key}')
        for package, expected in payload['versions'].items():
            if version(package) != expected:
                raise ValueError(f'library differs: {package}')
        for item in payload['sources']:
            if identity(run / 'code/src/model' / item['name']) != item:
                raise ValueError('model source identity differs')
        for item in payload['inputs']:
            if identity(Path(cfg['input_dir']) / item['name']) != item:
                raise ValueError('input identity differs')
        if digest(Path(cfg['population_model']['nest_likelihood']['path'])) != transfer['nests']:
            raise ValueError('NestWatch input differs')
        if transfer.get('synthetic'):
            raise ValueError('real-data run carries synthetic observations')
        write(upload / 'LOCAL_OFFLOAD_ACCEPTED.json', dict(token=current['token'], step=step,
              seed=s['seed'], epoch=time.time(), checkpoint_sha256=transfer['checkpoint_sha256'],
              manifest_sha256=current['manifest_sha256'], fingerprint=ck['fingerprint']))
        upload.rename(destination)
    (run / 'out').mkdir(exist_ok=True)
    (run / 'out/MAP_DONE').write_text('Complete float32 RTX 3090 MAP; strict fingerprint, input, seed and source verification.\n')
    return release(root, record, 'validated_local_MAP_ready_for_float64')


def inspect(root, record):
    r, q = paths(root)
    archived = r / 'runs' / record['step'] / f"LOCAL3090_{record['token']}.json"
    if archived.exists():
        return json.loads(archived.read_text())
    _, _, _, current = reservation(root, record)
    return current


def recover(root):
    r, q = paths(root)
    results = []
    with (q / 'local_offload.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for path in (q / 'local_running').glob('*.local3090.json'):
            record = json.loads(path.read_text())
            if time.time() - record['heartbeat'] > 1200:
                results.append(release(root, record, 'expired_local_lease_TACC_can_resume'))
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['claim', 'peek', 'heartbeat', 'accept', 'release', 'inspect', 'recover'])
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--token')
    p.add_argument('--record')
    p.add_argument('--exclude', default='')
    p.add_argument('--allow-minor', default='0')
    p.add_argument('--allow-transfer',default='0')
    p.add_argument('--reason', default='local_failure_TACC_can_run_cold')
    a = p.parse_args()
    if a.action == 'recover':
        print(json.dumps(recover(a.root)))
        return
    if not re.fullmatch('[a-f0-9]{32}', a.token or ''):
        raise ValueError('invalid worker token')
    if a.action in ('claim', 'peek'):
        result = claim(a.root, a.token, a.exclude.split(','), a.action == 'peek', a.allow_minor=='1',a.allow_transfer=='1')
    else:
        record = json.loads(a.record)
        if record['token'] != a.token:
            raise ValueError('token mismatch')
        if a.action == 'inspect':
            result = inspect(a.root, record)
        elif a.action == 'heartbeat':
            _, _, path, current = reservation(a.root, record)
            current['heartbeat'] = time.time()
            write(path, current)
            result = current
        elif a.action == 'release':
            result = release(a.root, record, a.reason)
        else:
            r, _ = paths(a.root)
            result = accept(a.root, record, r / 'monitor' / f"local3090_upload_{record['step']}_{a.token}")
    print(json.dumps(result))


if __name__ == '__main__':
    main()
