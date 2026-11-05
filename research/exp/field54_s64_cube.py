"""FIELD54 Stage F step 1: the summer 64-d cube (as s64) from a 128-d seasonal cube.

Exact coordinate selection of dims 0-63 (the summer teacher space; linear and per-feature, so it commutes with the path
integration that builds Z_disp), on the cube's own grid (no averaging). Writes Z_latent_{year}.npy (NaN off land) and a
64-d `ruzicka` cube_meta.json in the layout of the existing grid54 s64 inputs (encoder/cube_fdinterp_s64_grid54): the
source contract's summer block, its temporal/DESK provenance, and a seasonal_selection record. The full source contract
is embedded as `source_contract_128`.

    python research/exp/field54_s64_cube.py --src <128-d cube dir> --out <new dir> [--label TEXT]
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.community_encoder.cube_contract import validate_cube_contract  # noqa: E402

LO, HI = 0, 64


def contract_s64(src, label):
    """64-d summer contract from a 128-d seasonal contract (pure)."""
    summer = [b for b in src['feature_blocks'] if b['name'] == 'summer']
    if len(summer) != 1 or (summer[0]['start'], summer[0]['stop']) != (LO, HI):
        raise ValueError('source contract has no summer block at columns 0-63')
    c = {'schema_version': 2, 'kernel': 'ruzicka', 'centered': False, 'latent_dim': HI - LO,
         'feature_blocks': [{**copy.deepcopy(summer[0]), 'start': 0, 'stop': HI - LO}],
         'temporal_output': src.get('temporal_output'),
         'esk_basis_dir': summer[0].get('esk_basis_dir', src.get('esk_basis_dir')),
         'kernel_contract': "Z(x) dot Z(x') ~= uncentered Ruzicka(x,x')" + (
             ' [block mean embedding; see coarsening_note]' if 'coarsened_from' in src else ''),
         'years': src['years'], 'ensemble_method': src.get('ensemble_method'),
         'ensemble_members': src.get('ensemble_members'),
         'desk_checkpoint': src.get('desk_checkpoint'), 'desk_meta': src.get('desk_meta'),
         'source_cube': label,
         'seasonal_selection': {
             'season': 'summer', 'source_columns': list(range(LO, HI)),
             'note': 'dims 0-63 of the 128-d source are the SUMMER teacher space; the winter half (64-127) is dropped. '
                     'Selection precedes path integration; both are linear and per-feature, so order does not matter.',
             'operation': 'exact_coordinate_selection_before_path_integration'}}
    for k in ('evaluation_build', 'coarsened_from', 'coarsening_note', 'assim_arm', 'production_build'):
        if k in src:
            c[k] = copy.deepcopy(src[k])
    c['source_contract_128'] = copy.deepcopy(src)
    return c


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--src', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--label', default=None)
    a = p.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        raise FileExistsError(f'{a.out} exists and is not empty; refusing to overwrite')
    src = json.loads((a.src/'cube_meta.json').read_text())
    if int(src['latent_dim']) != 128:
        raise ValueError('expected a 128-d seasonal cube')
    c = contract_s64(src, a.label or str(a.src))
    files = sorted(a.src.glob('Z_latent_*.npy'), key=lambda q: int(q.stem.split('_')[-1]))
    years = [int(q.stem.split('_')[-1]) for q in files]
    if years != [int(y) for y in src['years']] or years != list(range(1902, 2026)):
        raise ValueError('cube years differ from the contract or from 1902-2025')
    a.out.mkdir(parents=True, exist_ok=True)
    sha = {}
    for q, y in zip(files, years):
        g = np.load(q)
        if g.ndim != 3 or g.shape[-1] != 128:
            raise ValueError(f'{q.name}: shape {g.shape}')
        np.save(a.out/f'Z_latent_{y}.npy', np.ascontiguousarray(g[..., LO:HI]).astype('float32'))
        sha[q.name] = hashlib.sha256(q.read_bytes()).hexdigest()
    c['source_cube_dir'] = str(a.src)
    c['source_file_sha256_digest'] = hashlib.sha256(''.join(f'{k}:{v}\n' for k, v in sorted(sha.items())).encode()).hexdigest()
    validate_cube_contract(c, HI - LO)
    (a.out/'cube_meta.json').write_text(json.dumps(c, indent=2))
    print(json.dumps({'out': str(a.out), 'n_years': len(years), 'shape': list(g.shape[:2]) + [HI - LO]}))


if __name__ == '__main__':
    main()
