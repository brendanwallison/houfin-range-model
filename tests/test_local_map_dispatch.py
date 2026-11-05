"""Queue ownership, partial-checkpoint and recovery invariants for local offload."""
import importlib.util
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import hashlib
import subprocess
import numpy as np

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research/exp'))
import local_map_dispatch as d


@pytest.fixture
def root(tmp_path):
    r = tmp_path / 'research/grid54'
    for folder in ['queue/pending', 'queue/running', 'queue/local_running', 'monitor', 'runs']:
        (r / folder).mkdir(parents=True, exist_ok=True)
    specs, manifests = [], {}
    for seed in [41, 42]:
        step = f'step41_veit_adult20_s{seed}'
        run = r / 'runs' / step
        (run / 'code').mkdir(parents=True)
        (run / 'code/fixture.txt').write_text('immutable source')
        (run / 'CODE_MANIFEST.sha256').write_text(d.digest(run / 'code/fixture.txt') + '  ./fixture.txt\n')
        spec = dict(step=step, seed=seed, queue_priority='0614')
        specs.append(spec)
        manifests[step] = d.digest(run / 'CODE_MANIFEST.sha256')
        (r / 'queue/pending' / f'0614_{step}.sh').write_text('# immutable task\n')
    (r / 'monitor/gradual_departure_specs.json').write_text(json.dumps(specs))
    (r / 'monitor/gradual_departure_VALIDATED.json').write_text(json.dumps(dict(passed=True, manifests=manifests)))
    return tmp_path


def test_concurrent_claims_are_distinct_and_preserve_scripts(root):
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda token: d.claim(root, token), ['a' * 32, 'b' * 32]))
    assert len({r['step'] for r in results}) == 2
    q = root / 'research/grid54/queue'
    assert not list((q / 'pending').glob('*.sh'))
    for rec in results:
        assert (q / 'local_running' / rec['name']).read_text() == '# immutable task\n'
        assert d.claim(root, rec['token'])['step'] == rec['step']


def test_source_mutation_prevents_claim(root):
    path = root / 'research/grid54/runs/step41_veit_adult20_s41/code/fixture.txt'
    path.write_text('changed')
    with pytest.raises(ValueError, match='source mismatch'):
        d.claim(root, 'a' * 32)
    assert len(list((root / 'research/grid54/queue/pending').glob('*.sh'))) == 2


def test_owner_mismatch_cannot_release(root):
    rec = d.claim(root, 'a' * 32)
    with pytest.raises(ValueError, match='ownership'):
        d.release(root, dict(rec, token='b' * 32), 'bad')


def test_release_records_durable_reconciliation(root):
    rec = d.claim(root, 'a' * 32)
    d.release(root, rec, 'failed_without_MAP')
    assert d.inspect(root, rec)['reason'] == 'failed_without_MAP'
    assert (root / 'research/grid54/queue/pending' / rec['name']).is_file()


def test_only_expired_leases_are_recovered(root):
    old = d.claim(root, 'a' * 32)
    fresh = d.claim(root, 'b' * 32)
    _, q, path, current = d.reservation(root, old)
    d.write(path, dict(current, heartbeat=0))
    recovered = d.recover(root)
    assert [r['step'] for r in recovered] == [old['step']]
    assert (q / 'local_running' / fresh['name']).is_file()


def test_incomplete_checkpoint_never_publishes_MAP_DONE(root):
    rec = d.claim(root, 'a' * 32)
    r = root / 'research/grid54'
    upload = r / 'monitor' / f"local3090_upload_{rec['step']}_{rec['token']}"
    upload.mkdir()
    (upload / 'map_checkpoint.pkl').write_bytes(pickle.dumps(dict(format_version=2, step=20, losses=[1.] * 20)))
    transfer = dict(seed=rec['seed'], token=rec['token'], code_manifest=rec['manifest_sha256'],
                    checkpoint_sha256=d.digest(upload / 'map_checkpoint.pkl'))
    (upload / 'local_map_transfer.json').write_text(json.dumps(transfer))
    with pytest.raises(ValueError, match='incomplete MAP'):
        d.accept(root, rec, upload)
    assert not (r / 'runs' / rec['step'] / 'out/MAP_DONE').exists()
    assert (r / 'queue/local_running' / rec['name']).is_file()


def test_ready_MAP_is_not_reserved_for_local_recomputation(root):
    out = root / 'research/grid54/runs/step41_veit_adult20_s41/out'
    out.mkdir()
    (out / 'MAP_DONE').touch()
    assert d.claim(root, 'a' * 32)['seed'] == 42


def complete_upload(root, wrong_version=False):
    r = root / 'research/grid54'
    step = 'step41_veit_adult20_s41'
    run = r / 'runs' / step
    code = run / 'code'
    (code / 'src/model').mkdir(parents=True)
    (code / 'config/overlays').mkdir(parents=True)
    (code / 'src/__init__.py').write_text('')
    (code / 'src/model/fixture.py').write_text('immutable model')
    (code / 'src/config_utils.py').write_text(
        'import json,os\ndef load_age_model_config():\n return json.load(open(os.environ["AGE_MODEL_CONFIG"]))\n')
    inputs = root / 'inputs'
    inputs.mkdir()
    (inputs / 'metadata.pkl').write_bytes(b'trusted input fixture')
    nests = root / 'nests.pkl'
    nests.write_bytes(b'trusted nest fixture')
    cfg = dict(input_dir=str(inputs), population_model=dict(nest_likelihood=dict(path=str(nests))),
               map=dict(init_lr=.01, profiles=dict(quick90=dict(target_steps=400,
                       prior_relaxation=[[0, .1], [100, 1.]], lr_decay_steps=400))))
    (code / f'config/overlays/grid54_{step}.json').write_text(json.dumps(cfg))
    manifest = run / 'CODE_MANIFEST.sha256'
    manifest.write_text('\n'.join(d.digest(p) + '  ./' + p.relative_to(code).as_posix()
                                  for p in sorted(code.rglob('*')) if p.is_file()) + '\n')
    validpath = r / 'monitor/gradual_departure_VALIDATED.json'
    valid = json.loads(validpath.read_text())
    valid['manifests'][step] = d.digest(manifest)
    validpath.write_text(json.dumps(valid))
    rec = d.claim(root, 'a' * 32)
    upload = r / 'monitor' / f"local3090_upload_{step}_{rec['token']}"
    upload.mkdir()
    payload = dict(precision='float32', map_profile='quick90', age_model_config=cfg,
                   prior_relaxation=[[0, .1], [100, 1.]], lr_decay_steps=400,
                   init_lr=.01, weight_decay=0., input_residency='device',
                   versions={'numpy': 'impossible-version'} if wrong_version else {},
                   sources=[d.identity(code / 'src/model/fixture.py')],
                   inputs=[d.identity(inputs / 'metadata.pkl')])
    canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'), default=list)
    params = dict(alpha_auto_loc=np.array([.2, .3]))
    ck = dict(format_version=2, step=400, losses=np.ones(400), params=params,
              fingerprint_payload=payload, fingerprint=hashlib.sha256(canonical.encode()).hexdigest())
    (upload / 'map_checkpoint.pkl').write_bytes(pickle.dumps(ck))
    (upload / 'map_params.pkl').write_bytes(pickle.dumps(params))
    transfer = dict(seed=rec['seed'], token=rec['token'], code_manifest=d.digest(manifest),
                    checkpoint_sha256=d.digest(upload / 'map_checkpoint.pkl'),
                    params_sha256=d.digest(upload / 'map_params.pkl'), nests=d.digest(nests))
    (upload / 'local_map_transfer.json').write_text(json.dumps(transfer))
    return rec, upload


def test_complete_verified_MAP_handoff_and_lost_response_reconciliation(root):
    rec, upload = complete_upload(root)
    proc = subprocess.run([sys.executable, str(Path(d.__file__)), 'accept', '--root', str(root),
                           '--token', rec['token'], '--record', json.dumps(rec)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    r = root / 'research/grid54'
    assert (r / 'runs' / rec['step'] / 'out/MAP_DONE').is_file()
    assert (r / 'queue/pending' / rec['name']).is_file()
    assert not upload.exists()
    assert d.inspect(root, rec)['reason'] == 'validated_local_MAP_ready_for_float64'


def test_library_mismatch_blocks_publication(root):
    rec, upload = complete_upload(root, wrong_version=True)
    proc = subprocess.run([sys.executable, str(Path(d.__file__)), 'accept', '--root', str(root),
                           '--token', rec['token'], '--record', json.dumps(rec)], capture_output=True, text=True)
    assert proc.returncode != 0 and 'library differs' in proc.stderr
    assert not (root / 'research/grid54/runs' / rec['step'] / 'out/MAP_DONE').exists()
