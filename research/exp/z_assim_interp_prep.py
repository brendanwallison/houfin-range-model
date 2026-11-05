"""In-era interpolation test, data side: summer rows 1966-1995 at TRAINING places, projected into the fixed summer64 teacher.

Production observes summer 1966-2025 at most cells, so 1966-2025 regional change is interpolation in space, not
extrapolation in time. This adds every valid pre-1996 summer row outside the spatial holdout and its buffer to the
field fits' summer sources. Only WITHHELD-place scores are valid afterwards: known-place historical evaluation rows
are among the added rows. Nearest prior: E016 skeptic round 3 (position x linear time placebo at zero reach, 0.436).

    python research/exp/z_assim_interp_prep.py --root R --base B
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.exp.z_repair_evaluate import matched_groups
from src.community_encoder.train_DESK.esk_kernel import project_points_to_z


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--base', type=Path, required=True)
    a = p.parse_args()
    root, task = a.root, 'backcast40'
    if torch.cuda.is_available():
        raise RuntimeError('Run on CPU')
    torch.set_num_threads(6)
    data = root/'data'/task
    out = root/'aux'/task/'interp'
    out.mkdir(parents=True, exist_ok=True)
    masks = dict(np.load(data/'summer_masks.npz'))
    keys = masks['keys']
    meta = json.loads((a.base/'meta.json').read_text())
    species = meta['community'] + meta['dev_species']
    if 'houfin' in species or set(species) & set(meta['sealed_species']):
        raise ValueError('Forbidden targets')
    assert np.array_equal(keys, np.load(a.base/'keys.npy'))
    pm = json.loads((data/'summer_train_points/points_meta.json').read_text())
    order = [species.index(s) for s in pm['species']]
    extra = masks['valid'] & (keys[:, 2] < 1996) & ~masks['spatial_holdout'] & ~masks['buffer']
    assert not (extra & masks['train']).any()
    # No added row may sit at a withheld panel cell.
    cells, spatial, _ = matched_groups(keys, masks, 2)
    held = {tuple(c) for c in cells[spatial]}
    assert not any((int(r), int(c)) in held for r, c in keys[extra, :2])
    rows = np.flatnonzero(extra)
    Y = np.hstack([np.load(a.base/'X_comm.npy'), np.load(a.base/'X_dev.npy')]).astype('float64')
    print('Projecting', len(rows), 'pre-1996 training-place summer rows', flush=True)
    Z = project_points_to_z(np.log1p(Y[rows][:, order]).astype('float32'),
                            str(root/'teachers'/task/'summer/spacetime'), 64, batch=1024)
    np.savez_compressed(out/'summer_extra.npz', keys=keys[rows].astype('int32'), Z=np.asarray(Z, 'float32'), source_rows=rows)
    (out/'PREP_DONE.json').write_text(json.dumps({'rows': int(len(rows)), 'years': [int(keys[rows, 2].min()),
        int(keys[rows, 2].max())], 'scope': __doc__}, indent=2))


if __name__ == '__main__':
    main()
