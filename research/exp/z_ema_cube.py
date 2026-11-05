"""Export a DESK cube as the quantity its trainer supervised: the learned causal output EMA of the raw cube.

DESK fits z_ema = EMA_h(z_raw) to the teacher, but build_spacetime_cube exports z_raw, and z_assim_evaluate scores
the cube as given. The z_assim field arms are fitted on raw Z directly, so comparing them with raw DESK cubes
compares one model's fitted quantity with the other's unfitted one. This writes cubes/backcast40/<cube>_ema with
apply_output_ema (the trainer's numpy twin) over 1902-2025 at the model's own learned half-life, on all 128 dims,
as z_repair_ema_audit.py does. The half-life is the training value; nothing is selected on evaluation data.

    python research/exp/z_ema_cube.py --root R --model KEY --cube NAME
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from src.community_encoder.train_DESK.desk_training import apply_output_ema


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--model', required=True, help='models/backcast40/<model> (desk_meta.npz)')
    p.add_argument('--cube', required=True, help='cubes/<task>/<cube>')
    p.add_argument('--task', default='backcast40', help='models/<task>, cubes/<task> (default backcast40, as before)')
    a = p.parse_args()
    dm = np.load(a.root/'models'/a.task/a.model/'desk_meta.npz', allow_pickle=True)
    if not bool(dm['output_ema']):
        raise ValueError('Model has no training output EMA')
    hl, start = float(dm['ema_half_life']), int(dm['ema_warmup_start'])
    src, out = a.root/'cubes'/a.task/a.cube, a.root/'cubes'/a.task/(a.cube + '_ema')
    if (out/'EMA_DONE.json').exists():
        raise RuntimeError('Refuse to overwrite ' + str(out))
    years = np.arange(start, 2026)
    raw = np.stack([np.load(src/f'Z_latent_{y}.npy') for y in years])          # (T, H, W, 128)
    land = np.isfinite(raw).all((0, 3))
    ema = np.full_like(raw, np.nan)
    ema[:, land] = apply_output_ema(raw[:, land], hl)
    out.mkdir(parents=True, exist_ok=True)
    for j, y in enumerate(years):
        np.save(out/f'Z_latent_{y}.npy', ema[j])
    for f in src.iterdir():
        if not f.name.startswith('Z_latent_'):
            shutil.copyfile(f, out/f.name)
    (out/'EMA_DONE.json').write_text(json.dumps({'source_cube': str(src), 'model': a.model, 'half_life': hl,
        'warmup_start': start, 'dims': 'all 128', 'land_cells': int(land.sum()),
        'note': 'causal output EMA at the trained half-life; the quantity DESK supervised'}, indent=2))
    print(out, hl)


if __name__ == '__main__':
    main()
