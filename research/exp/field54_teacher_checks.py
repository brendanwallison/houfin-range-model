"""FIELD54 Stage C checks: teachers / auxiliaries for the production build and the masked twin (read-only).

    python research/exp/field54_teacher_checks.py --eval ROOT_EVAL --prod ROOT_PROD --obs OBS --out checks.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def teacher(root, task, season):
    t = root / 'teachers' / task / season
    meta = json.loads((t / 'spacetime/meta.json').read_text())
    audit = json.loads((t / 'TRAINING_ONLY_TEACHER.json').read_text())
    lm = np.load(t / 'spacetime/esk_landmarks.npy', mmap_mode='r')
    pm = np.load(t / 'spacetime/esk_projmat.npy', mmap_mode='r')
    d = meta['diagnostics']['uncentered']
    return {'latent_dim': meta['latent_dim'], 'n_species': meta['n_species'], 'n_landmarks': meta['n_landmarks'],
            'landmarks_shape': list(lm.shape), 'projmat_shape': list(pm.shape),
            'source_rows': audit['source_rows'], 'n_synthetic_window_rows': meta['n_synthetic_window_rows'],
            'landmark_synthetic_share': meta['landmark_synthetic_share'], 'recent_frac': meta['recent_frac'],
            'median_z_norm2_annual': meta['median_z_norm2_annual'],
            'median_z_norm2_window_mean': meta['median_z_norm2_window_mean'],
            'uncentered_combined_rmse': d['combined_rmse_norm'], 'uncentered_trunc_rmse': d['truncation_only_rmse_norm'],
            'existing_encoder_checkpoint_used': audit['existing_encoder_checkpoint_used']}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    for k in ['eval', 'prod', 'obs', 'out']:
        ap.add_argument('--' + k, type=Path, required=True)
    a = ap.parse_args()
    obs = json.loads((a.obs / 'DONE.json').read_text())
    sealed = set(obs['checks']['species']['sealed_species_in_production'])
    rep = {}
    # ---- twin ---------------------------------------------------------------------------------------------
    d = a.eval / 'data' / 'backcast40'
    m = dict(np.load(d / 'summer_masks.npz'))
    keys = m['keys']
    ho, bf = np.load(d / 'holdout_cells.npy'), np.load(d / 'buffer_cells.npy')
    tk = np.load(d / 'summer_train_points/point_index.npy')
    src = np.load(d / 'summer_train_source_rows.npy')
    assert np.array_equal(tk, keys[src]) and m['train'][src].all() and m['train'].sum() == len(src)
    assert (tk[:, 2] >= 1996).all() and not ho[tk[:, 0], tk[:, 1]].any() and not bf[tk[:, 0], tk[:, 1]].any()
    sp = pd.read_csv(d / 'summer_species.csv')['species_code'].tolist()
    assert 'houfin' not in sp and not set(sp) & sealed
    wd = d / 'winter_full'
    wk = np.load(wd / 'count_points/point_index.npy')
    rm = dict(np.load(wd / 'record_masks.npz'))
    assert not ho[wk[:, 0], wk[:, 1]].any() and not bf[wk[:, 0], wk[:, 1]].any()
    assert not set(rm['circle'][rm['train']]) & set(rm['circle'][rm['evaluation']])
    wsp = [str(s) for s in np.load(wd / 'train_lists.npz')['species']]
    assert 'houfin' not in wsp and not set(wsp) & sealed
    man = json.loads((d / 'manifest.json').read_text())
    ex = np.load(a.eval / 'aux/backcast40/interp/summer_extra.npz')
    assert (ex['keys'][:, 2] < 1996).all() and not ho[ex['keys'][:, 0], ex['keys'][:, 1]].any() \
        and not bf[ex['keys'][:, 0], ex['keys'][:, 1]].any()
    ip = json.loads((a.eval / 'data/backcast40_interp/summer_train_points/DONE.json').read_text())
    rep['eval'] = {
        'summer_rows_total': int(len(keys)), 'summer_training_rows': int(m['train'].sum()),
        'summer_roles': man['summer_role_counts'], 'panel_known_cells': man['summer_panel_known_cells'],
        'panel_withheld_cells': man['summer_panel_spatial_cells'],
        'summer_rows_at_holdout_cells': int(m['spatial_holdout'].sum()), 'summer_rows_at_buffer_cells': int(m['buffer'].sum()),
        'summer_rows_invalid': int((~m['valid']).sum()),
        'winter': man['winter']['full'], 'species_summer': len(sp), 'species_winter': len(wsp),
        'interp_rows_added': int(len(ex['keys'])), 'interp_points': ip,
        'teacher_summer': teacher(a.eval, 'backcast40', 'summer'),
        'teacher_winter': teacher(a.eval, 'backcast40', 'winter_full'),
        'aux': json.loads((a.eval / 'aux/backcast40/full/DONE.json').read_text()),
        'checks': 'twin summer teacher rows = training mask, all >= 1996, none at 54 km holdout/buffer cells; winter '
                  'count points none at holdout/buffer cells; train/eval circles disjoint; interp rows < 1996 and '
                  'outside holdout/buffer; no sealed species, no House Finch'}
    # ---- production ---------------------------------------------------------------------------------------
    d = a.prod / 'data' / 'production'
    m = dict(np.load(d / 'summer_masks.npz'))
    tk = np.load(d / 'summer_train_points/point_index.npy')
    assert m['train'].sum() == m['valid'].sum() == len(tk)
    sp = pd.read_csv(d / 'summer_species.csv')['species_code'].tolist()
    assert 'houfin' not in sp and sealed <= set(sp)
    wd = d / 'winter_full'
    rm = dict(np.load(wd / 'record_masks.npz'))
    assert rm['train'].sum() == rm['valid'].sum() and not rm['evaluation'].any()
    wsp = [str(s) for s in np.load(wd / 'train_lists.npz')['species']]
    assert 'houfin' not in wsp
    man = json.loads((d / 'manifest.json').read_text())
    rep['prod'] = {
        'summer_rows_total': int(len(m['keys'])), 'summer_training_rows': int(m['train'].sum()),
        'summer_rows_invalid': int((~m['valid']).sum()),
        'summer_year_range': [int(tk[:, 2].min()), int(tk[:, 2].max())],
        'winter': man['winter']['full'], 'species_summer': len(sp), 'species_winter': len(wsp),
        'sealed_in_summer': len(sealed & set(sp)), 'sealed_in_winter': len(sealed & set(wsp)),
        'teacher_summer': teacher(a.prod, 'production', 'summer'),
        'teacher_winter': teacher(a.prod, 'production', 'winter_full'),
        'aux': json.loads((a.prod / 'aux/production/full/DONE.json').read_text()),
        'checks': 'every valid summer row and winter list trains; nothing evaluated; sealed included; no House Finch'}
    a.out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
