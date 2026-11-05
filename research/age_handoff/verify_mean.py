"""Verify that full-grid mean cubes reproduce the features used in held-out evaluation."""
import argparse
import json
import os
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--family', choices=['v1', 'v2'], required=True)
a = p.parse_args()
root = Path('/work/07980/bwa386/ls6/houfin/research/final_holdout_20261025/features')
family = 'final' if a.family == 'v1' else 'final_v2'
ref = root / family
cells, years = np.load(ref / 'cells.npy'), np.load(ref / 'years.npy')
features = np.load(ref / 'z_raw_all.npy', mmap_mode='r')
cube = Path(os.environ['HOUFIN_PROCESSED']) / 'encoder' / ('cube_desk_final_mean4_diag' if a.family == 'v1' else 'cube_desk_final_v2_mean4_diag')
error = 0.
for y in [1902, 1940, 1966, 2020, 2025]:
    z = np.load(cube / f'Z_latent_{y}.npy', mmap_mode='r')[cells[:, 0], cells[:, 1]]
    expected = features[:, int(np.where(years == y)[0][0])]
    np.testing.assert_allclose(z, expected, atol=1e-7, rtol=1e-6, equal_nan=True)
    error = max(error, float(np.nanmax(np.abs(z - expected))))
(cube / 'evaluation_equivalence.json').write_text(json.dumps({'years_checked': [1902, 1940, 1966, 2020, 2025],
    'cells_checked': len(cells), 'max_abs_difference': error, 'reference': str(ref)}, indent=2))
print('mean cube matches evaluated ensemble', a.family, error, flush=True)
