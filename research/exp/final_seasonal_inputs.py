"""Derive one explicit seasonal basis from immutable integrated sum-kernel inputs."""
import argparse
import copy
import json
from pathlib import Path
import pickle
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.community_encoder.cube_contract import artifact_identity, validate_model_feature_contract


def derive(source, out, season):
    if (out / 'DERIVED_DONE.json').exists():
        return
    with (source / 'metadata.pkl').open('rb') as f:
        meta = pickle.load(f)
    original = copy.deepcopy(meta['z_kernel_contract'])
    validate_model_feature_contract(original, meta['M'])
    blocks = [b for b in original['feature_blocks'] if b['name'] == season]
    if len(blocks) != 1:
        raise ValueError('select one named, identified seasonal block')
    b = blocks[0]; lo, hi = b['start'], b['stop']; width = hi-lo
    contract = copy.deepcopy(original)
    contract.update(kernel='ruzicka', latent_dim=width, source_latent_dim=width,
                    truncation='none', esk_basis_dir=b['esk_basis_dir'],
                    feature_blocks=[{**copy.deepcopy(b), 'start': 0, 'stop': width}])
    contract.pop('feature_columns', None)
    contract.pop('source_feature_blocks', None)
    contract['seasonal_derivation'] = {'source_contract': original,
                                     'source_columns': list(range(lo, hi)),
                                     'operation': 'exact_coordinate_selection_after_linear_path_integration'}
    validate_model_feature_contract(contract, width)
    out.mkdir(parents=True, exist_ok=True)
    identities = {'source_metadata': artifact_identity(source / 'metadata.pkl')}
    for key, shape in [('z_gathered_path', (meta['time'], meta['N_land'], meta['M'])),
                       ('z_disp_gathered_path', (meta['time'], meta['N_land'], meta['K'], meta['M']))]:
        src = source / meta[key]
        if src.stat().st_size != int(np.prod(shape))*4:
            raise ValueError(f'wrong tensor size: {src}')
        identities[key] = {'source': artifact_identity(src)}
        name = f'{key}_{season}.dat'; dst = out / name
        x = np.memmap(src, dtype='float32', mode='r', shape=shape)
        y = np.memmap(dst, dtype='float32', mode='w+', shape=(*shape[:-1], width))
        for i in range(shape[0]):
            y[i] = x[i, ..., lo:hi]
            if not np.array_equal(y[i], x[i, ..., lo:hi]):
                raise ValueError('derived coordinates differ from exact source selection')
        y.flush(); del y, x
        identities[key]['derived'] = artifact_identity(dst)
        meta[key] = name
    for name in ('obs_route_ids.npz', 'subcell_strata.npz'):
        shutil.copy2(source / name, out / name)
        identities[name] = artifact_identity(out / name)
    meta['M'] = width; meta['z_kernel_contract'] = contract
    # Original path_feature_meta remains provenance of the actual source integration.
    meta['seasonal_derivation'] = contract['seasonal_derivation']
    with (out / 'metadata.pkl').open('wb') as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
    identities['derived_metadata'] = artifact_identity(out / 'metadata.pkl')
    (out / 'DERIVED_DONE.json').write_text(json.dumps(identities, indent=2))
    print('Verified exact seasonal inputs', season, out, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--season', required=True, choices=['summer', 'winter_half'])
    a = p.parse_args(); derive(a.source, a.out, a.season)
