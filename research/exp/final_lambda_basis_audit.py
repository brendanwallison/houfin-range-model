"""Replay local fundamental lambda and separate readout amplification from raw Z scale.

This is a fixed-point diagnostic, not a refit or an ecological causal attribution.
Uses the exact pre-clean v1 checkpoint/input pair displayed in the shared figures.
"""
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys

import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / '.research_runtime/lambda_audit_20261026'
OUT = ROOT / 'research/experiments/lambda_audit_20261026'
FIELDS = ['adult_survival', 'fecundity', 'capacity', 'juvenile_ratio']


def coefficients(path):
    p = {k.removesuffix('_auto_loc'): np.asarray(v, float)
         for k, v in pickle.loads(path.read_bytes()).items()}
    scale = np.logaddexp(0, p['w_scale_raw'])
    rho = expit(p['manifold_coupling_logit'])
    unit = np.sqrt(rho)*p['manifold_factor'] + np.sqrt(1-rho)*p['manifold_idio']
    own = np.sqrt(rho)*p['manifold_factor_within'] + np.sqrt(1-rho)*p['manifold_idio_within']
    r = p['within_level_corr']
    level = unit * scale
    within = (r*unit + np.sqrt(1-r*r)*own) * (.14*scale)
    alpha = np.array([p['alpha_a'], p['alpha_f'], p['alpha_k'], p['alpha_j']])
    return level, within, alpha, scale, r


def fundamental(eta):
    sa = expit(eta[..., 0])
    sj = sa * expit(eta[..., 3])
    f = np.logaddexp(0, eta[..., 1])
    return (sa + np.sqrt(sa*sa + 4*sj*f*sa))/2


def main():
    m = pickle.loads((DATA/'fd_v1-metadata.pkl').read_bytes())
    z = np.memmap(DATA/'fd_v1-Z.dat', dtype='float32', mode='r', shape=(m['time'], m['N_land'], m['M']))
    years = np.asarray(m['years'])
    ref = np.asarray(z[(years >= 1996) & (years <= 2025)], float).mean(0)
    early = (years >= 1902) & (years <= 1915)
    modern = (years >= 2010) & (years <= 2025)
    dz = np.asarray(z[modern], float).mean(0)-np.asarray(z[early], float).mean(0)
    lookup = np.full((m['Ny'], m['Nx']), -1, int)
    lookup[m['land_rows'], m['land_cols']] = np.arange(m['N_land'])
    ocell = lookup[m['obs_rows'], m['obs_cols']]
    keep = (np.asarray(m['observed_results']) > 0) & (ocell >= 0)
    pairs = np.unique(np.stack([np.asarray(m['obs_time_indices'])[keep], ocell[keep]], axis=1), axis=0)
    center = np.asarray(z[pairs[:, 0], pairs[:, 1]], float).mean(0)
    from src.config_utils import load_data_config
    import rasterio
    from rasterio.warp import transform
    with rasterio.open(load_data_config()['grid']['ref_raster']) as ds:
        x, y = rasterio.transform.xy(ds.transform, m['land_rows'], m['land_cols'])
        lon, lat = map(np.asarray, transform(ds.crs, 'EPSG:4326', x, y))
    masks = {'all_land': np.ones(m['N_land'], bool),
             'central_east_25_49N': (lon >= -95) & (lon < -80) & (lat >= 25) & (lat < 49),
             'east_of_80W_25_49N': (lon >= -80) & (lat >= 25) & (lat < 49)}
    checkpoints = {
        'map': OUT/'checkpoints/age_map_float32_fd_v1_r64_quick90/map_params.pkl',
        'refined': OUT/'checkpoints/refined_float64__age_map_float32_fd_v1_r64_quick90__finite_restart1/map_params.pkl'}
    beta = {name: coefficients(path) for name, path in checkpoints.items()}
    result = {'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'interpretation': 'Exploratory fixed-point habitat replay. No fit or production Z change.',
              'stages': {}, 'counterfactuals': {}, 'raw_geometry': {}}
    eta_by_stage = {}
    for stage, (bl, bw, alpha, scale, corr) in beta.items():
        level = (ref-center)@bl + alpha
        eta = np.stack([level+(np.asarray(zt, float)-ref)@bw for zt in z])
        lam = fundamental(eta)
        E, M = lam[early].mean(0), lam[modern].mean(0)
        eta_by_stage[stage] = eta
        item = {'fields': {}, 'regions': {}}
        for j, field in enumerate(FIELDS):
            change = dz@bw[:, j]
            nl, nw = np.linalg.norm(bl[:, j]), np.linalg.norm(bw[:, j])
            item['fields'][field] = {
                'amplitude': float(scale[j]), 'within_amplitude': float(.14*scale[j]),
                'level_coefficient_norm': float(nl), 'within_coefficient_norm': float(nw),
                'mixing_parameter': float(corr[j]),
                'level_within_coefficient_cosine': float(bl[:, j]@bw[:, j]/(nl*nw)),
                'century_link_change_mean': float(change.mean()),
                'century_link_change_RMS': float(np.sqrt(np.mean(change*change))),
                'change_RMS_per_coefficient_norm': float(np.sqrt(np.mean(change*change))/nw),
                'spatial_level_SD': float(np.std(ref@bl[:, j]))}
        for region, mask in masks.items():
            item['regions'][region] = {
                'n_cells': int(mask.sum()), 'early_lambda': float(E[mask].mean()),
                'modern_lambda': float(M[mask].mean()), 'lambda_change': float((M-E)[mask].mean()),
                'gained_suitable_fraction': float(((M > 1) & (E <= 1))[mask].mean()),
                'lost_suitable_fraction': float(((M <= 1) & (E > 1))[mask].mean())}
        expected = json.loads((OUT/'map_metrics.json' if stage=='map' else
                               ROOT/'research/validation_figures/fd_v1_r64_refined/viz_ref/metrics.json').read_text())
        item['replay_max_mean_lambda_error'] = max(abs(E.mean()-expected['early_mean_lambda']),
                                                   abs(M.mean()-expected['modern_mean_lambda']))
        assert item['replay_max_mean_lambda_error'] < 2e-6, item
        result['stages'][stage] = item
    # Keep refined spatial levels/intercepts; vary only the temporal coefficient vector.
    bl, bw, alpha, _, _ = beta['refined']
    bmap = beta['map'][1]
    norm_matched = bw * (np.linalg.norm(bmap, axis=0)/np.linalg.norm(bw, axis=0))
    level = (ref-center)@bl + alpha
    for name, weights in [('MAP_temporal_coefficients', bmap),
                          ('refined_direction_MAP_coefficient_norm', norm_matched)]:
        lam = np.stack([fundamental(level+(np.asarray(zt, float)-ref)@weights) for zt in z])
        E, M = lam[early].mean(0), lam[modern].mean(0)
        result['counterfactuals'][name] = {
            r: {'early_lambda': float(E[s].mean()), 'modern_lambda': float(M[s].mean()),
                'lambda_change': float((M-E)[s].mean())} for r, s in masks.items()}
    for j in [0, 1, 3]:
        eta = eta_by_stage['refined'].copy()
        # Remove only this field's temporal departure, preserving its fitted spatial level.
        eta[..., j] = level[None, :, j]
        lam = fundamental(eta)
        result['counterfactuals']['static_'+FIELDS[j]] = {
            r: {'lambda_change': float((lam[modern].mean(0)-lam[early].mean(0))[s].mean())}
            for r, s in masks.items()}
    for name, zz in [('final_v1_r64', z)]:
        spatial = ref-ref.mean(0)
        result['raw_geometry'][name] = geometry(spatial, dz)
        result['raw_geometry'][name]['regions'] = {
            r: geometry(ref[s]-ref[s].mean(0), dz[s]) for r, s in masks.items()}
    oldmeta = pickle.loads((DATA/'cbc1902_r64-metadata.pkl').read_bytes())
    oldpath = Path('/home/breallis/houfin/work/houfin/processed/model_inputs/numpyro_input_cbc1902_r64')/oldmeta['z_gathered_path']
    old = np.memmap(oldpath, dtype='float32', mode='r', shape=z.shape)
    assert np.array_equal(m['land_rows'], oldmeta['land_rows']) and np.array_equal(m['land_cols'], oldmeta['land_cols'])
    oldref = np.asarray(old[(years >= 1996) & (years <= 2025)], float).mean(0)
    olddelta = np.asarray(old[modern], float).mean(0)-np.asarray(old[early], float).mean(0)
    result['raw_geometry']['earlier_cbc1902_r64'] = geometry(oldref-oldref.mean(0), olddelta)
    result['raw_geometry']['earlier_cbc1902_r64']['regions'] = {
        r: geometry(oldref[s]-oldref[s].mean(0), olddelta[s]) for r, s in masks.items()}
    (OUT/'basis_readout_audit.json').write_text(json.dumps(result, indent=2))
    for name, row in result['stages'].items():
        print(name, row['regions'], 'replay_error', row['replay_max_mean_lambda_error'])
        print('FIELDS', name, row['fields'])
    print('RAW_GEOMETRY', result['raw_geometry'])
    print('COUNTERFACTUALS',result['counterfactuals'])


def geometry(spatial, delta):
    energy = np.mean(delta*delta, axis=0)
    trace = np.mean(np.sum(spatial*spatial, axis=1))
    gram = delta.T@delta/len(delta)
    vals = np.maximum(np.linalg.eigvalsh(gram), 0)
    return {'spatial_trace': float(trace), 'century_change_energy': float(energy.sum()),
            'century_change_RMS_over_spatial_SD': float(np.sqrt(energy.sum()/trace)),
            'continental_mean_fraction_of_change_energy': float(np.sum(delta.mean(0)**2)/energy.sum()),
            'change_effective_rank': float(vals.sum()**2/np.sum(vals*vals))}


if __name__ == '__main__':
    main()
