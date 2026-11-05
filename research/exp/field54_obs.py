"""FIELD54 Stage B: native 54 km observations for the production build and the masked twin.

Plan: research/experiments/FIELD54_NATIVE_PLAN_2026-10-29.md (Stage B); record FIELD54_OBS_TEACHERS_2026-10-30.md.

BBS: every QC route keeps its 27 km route cell (bbs_community.route_grid_map on the 27 km mask, as every 27 km build),
which is mapped to its 54 km parent with coarsen_model_grid's coastal rule (parent_cells: a parent that is not 54 km
land is snapped to the nearest 54 km land cell). bbs_community_points.build_bbs_rows then does what it always does --
cell-year = mean over the cell's QC route-years, log1p -- so a 54 km cell-year is the mean over ALL its routes. The
observation set is the 27 km one exactly; only the cell label changes.
CBC: circle row/col from the 54 km reference raster (same lat/lon -> grid formula as cbc_prepare); a list valid at
27 km (its 27 km cell is land) keeps validity at 54 km, with the same coastal snap; a list invalid at 27 km stays out.
Species: twin = the 27 km all-species build (cache community + dev, never sealed, never House Finch);
production = twin + the cache's sealed species (user 2026-10-29: all species except House Finch).
Masks (twin): a 54 km cell is holdout if ANY 27 km child is holdout; buffer likewise (holdout wins); plus every
54 km cell that receives a snapped route / circle from a 27 km holdout (buffer) cell -- so nothing that touched a
withheld place can train.

Runs in the 27 km data environment (the BBS loaders read the 27 km release and mask); the 54 km grid is passed
explicitly. Writes only under --out; refuses to overwrite.

    python research/exp/field54_obs.py --mask54 M --ref54 R --cache-meta META --points27 P --route-years27 RY \
        --cbc CBC --cbc-sealed CBCS --holdout27 H --buffer27 B --panel27 SUMMER_MASKS --out OUT
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

F = 2
CIRCLE_KEYS = ['circles', 'strata', 'circle_idx', 'stratum_idx', 'year_end', 'row', 'col', 'lat', 'lon',
               'party_hours', 'distance_km', 'field_counters', 'max_parties', 'feeder_hrs', 'feeder_counters',
               'nocturnal_hrs']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as fh:
        for block in iter(lambda: fh.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def block_any(mask_f, shape_c, f=F):
    """Coarse cell True iff ANY fine child is True (odd fine edges padded False)."""
    H, W = shape_c
    pad = np.zeros((H * f, W * f), bool)
    pad[:mask_f.shape[0], :mask_f.shape[1]] = mask_f
    return pad.reshape(H, f, W, f).any(axis=(1, 3))


def coarse_masks(holdout_f, buffer_f, land_c, src_rc_f, dst_rc_c, f=F):
    """54 km (holdout, buffer) from the 27 km masks (pure).

    ANY-child rule on coarse land, holdout winning over buffer; then every coarse cell that RECEIVES an
    observation from a fine holdout (buffer) cell is holdout (buffer, unless holdout). ``src_rc_f`` /
    ``dst_rc_c`` are (n, 2) fine source and coarse destination cells of the observations (routes, circles).
    """
    ho = block_any(holdout_f, land_c.shape, f) & land_c
    bf = block_any(buffer_f, land_c.shape, f) & land_c & ~ho
    src, dst = np.asarray(src_rc_f, int), np.asarray(dst_rc_c, int)
    from_ho = holdout_f[src[:, 0], src[:, 1]]
    from_bf = buffer_f[src[:, 0], src[:, 1]]
    ho_snap = from_ho & ~ho[dst[:, 0], dst[:, 1]]
    ho[dst[from_ho, 0], dst[from_ho, 1]] = True
    bf &= ~ho
    bf_snap = from_bf & ~(ho | bf)[dst[:, 0], dst[:, 1]]
    sel = from_bf & ~ho[dst[:, 0], dst[:, 1]]
    bf[dst[sel, 0], dst[sel, 1]] = True
    assert not (ho & bf).any()
    assert ho[dst[from_ho, 0], dst[from_ho, 1]].all()
    assert (ho | bf)[dst[from_bf, 0], dst[from_bf, 1]].all()
    return ho, bf, {'holdout_cells_added_by_snap': int(len({tuple(x) for x in dst[ho_snap]})),
                    'buffer_cells_added_by_snap': int(len({tuple(x) for x in dst[bf_snap]}))}


def points_meta(codes, bmeta, community_csv, X, keys, anchor_year=2025):
    """The consumer contract bbs_community_points.main writes for a BBS-only build."""
    n_recent = int((keys[:, 2] == anchor_year).sum())
    if n_recent == 0:
        raise SystemExit(f'no rows at anchor_year {anchor_year}')
    return {'bbs': bmeta, 'target_source': 'bbs_raw', 'species': list(codes), 'n_species': len(codes),
            'n_weeks': 1, 'community_csv': str(community_csv), 'ruzicka_log1p': True, 'n_rows': int(X.shape[0]),
            'recent_year': anchor_year, 'n_recent': n_recent, 'n_hist': int(X.shape[0]) - n_recent,
            'grid': '54km native (FIELD54 Stage B)'}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    for k in ['mask54', 'ref54', 'cache_meta', 'points27', 'route_years27', 'cbc', 'cbc_sealed', 'holdout27',
              'buffer27', 'panel27', 'out']:
        ap.add_argument('--' + k.replace('_', '-'), dest=k, required=True, type=Path)
    ap.add_argument('--sealed-read-logged', required=True,
                    help='where the sealed-species read for production was logged (research/sealed_reads.csv entry)')
    a = ap.parse_args()
    if (a.out / 'DONE.json').exists():
        raise RuntimeError('Refuse to overwrite a completed Stage B output')
    a.out.mkdir(parents=True, exist_ok=True)
    import rasterio
    from rasterio.warp import transform as warp_transform
    from src.data.preprocess import bbs
    from src.data.identify.bbs_crosswalk import build_crosswalk
    from src.config_utils import load_data_config
    from src.community_encoder.train_DESK.bbs_community_points import build_bbs_rows, write_points
    from src.processing.coarsen_model_grid import parent_cells

    report = {'inputs': {k: str(getattr(a, k)) for k in vars(a)}, 'checks': {}}
    chk = report['checks']
    # ---- grids --------------------------------------------------------------------------------------------
    land27, _o, t27, crs27, nx27, ny27 = bbs.load_grid_reference(bbs.MASK_PATH)
    land27 = np.asarray(land27, bool)
    with rasterio.open(a.mask54) as r:
        land54 = r.read(1) == 0
        t54m = r.transform
    with rasterio.open(a.ref54) as r:
        t54, crs54, H54, W54 = r.transform, r.crs, r.height, r.width
    assert land54.shape == (H54, W54) == (67, 112), (land54.shape, H54, W54)
    assert land27.shape == (133, 224), land27.shape
    assert t54m == t54 and crs54 == crs27
    assert (t54.c, t54.f) == (t27.c, t27.f) and np.isclose(t54.a, 2 * t27.a) and np.isclose(t54.e, 2 * t27.e)
    report['grid'] = {'shape27': list(land27.shape), 'shape54': [H54, W54], 'land27': int(land27.sum()),
                      'land54': int(land54.sum()), 'mask54': str(a.mask54), 'mask27': bbs.MASK_PATH,
                      'mask54_sha256': digest(a.mask54)}
    # ---- species ------------------------------------------------------------------------------------------
    meta = json.loads(a.cache_meta.read_text())
    pm27 = json.loads((a.points27 / 'points_meta.json').read_text())
    twin = [str(c) for c in pm27['species']]
    nonsealed = set(meta['community']) | set(meta['dev_species'])
    sealed = [str(c) for c in meta['sealed_species']]
    assert set(twin) == nonsealed and not nonsealed & set(sealed)
    prod = twin + sealed
    for codes in (twin, prod):
        assert 'houfin' not in codes and len(set(codes)) == len(codes)
    dcfg = load_data_config()
    dr = dcfg['datasets_root']
    xw, diag = build_crosswalk(dcfg.get('bbs', {}).get('species_list') or f'{dr}/bbs_2026_release/SpeciesList.csv',
                               f'{dr}/avonet/eBird_taxonomy.csv',
                               dcfg.get('species_list') or f'{dr}/avonet/reference_community_ranked.csv',
                               top_n=None, community_codes=prod)
    assert bbs.HOUSE_FINCH_AOU not in set(xw['aou'].astype(int)), 'House Finch AOU in the production crosswalk'
    assert 'houfin' not in set(xw['species_code'])
    chk['species'] = {'twin': len(twin), 'production': len(prod), 'sealed_added': len(sealed),
                      'sealed_species_in_production': sealed,
                      'production_crosswalk_matched': int(diag['n_matched']),
                      'production_crosswalk_unmatched': int(diag['n_unmatched']),
                      'house_finch_aou_in_crosswalk': False, 'houfin_in_any_list': False,
                      'rule_twin': 'cache community + dev (non-sealed), House Finch out (= 27 km all369 build)',
                      'rule_production': 'twin + cache sealed species; House Finch out'}
    print('[f54-obs] species', json.dumps({k: v for k, v in chk['species'].items()
                                          if k != 'sealed_species_in_production'}), flush=True)
    pd.DataFrame({'species_code': twin}).to_csv(a.out / 'species_eval.csv', index=False)
    pd.DataFrame({'species_code': prod}).to_csv(a.out / 'species_prod.csv', index=False)

    # ---- route maps ---------------------------------------------------------------------------------------
    maps = {}

    def identity_27km(rc):
        maps['27'] = rc.copy()
        return rc

    def parent_54km_coastal_snap(rc):
        rr, cc, off, keep, dist = parent_cells(rc['row'].to_numpy(), rc['col'].to_numpy(), land54, F, 'snap')
        out = rc.copy()
        out['row'], out['col'] = rr.astype(int), cc.astype(int)
        out['row27'], out['col27'] = rc['row'].to_numpy(), rc['col'].to_numpy()
        out['snapped'], out['snap_cells'] = off, dist
        maps['54'] = out
        return out[['CountryNum', 'StateNum', 'Route', 'row', 'col']]

    # (a) 27 km reproduction of the existing all369 points: the hook is inert when it maps to itself.
    X27, K27, W27, m27 = build_bbs_rows(twin, route_cell_map=identity_27km)
    Xr, Kr, Wr = (np.load(a.points27 / n) for n in ['X_points.npy', 'point_index.npy', 'point_weights.npy'])
    chk['reproduce_27km_all369'] = {'keys_equal': bool(np.array_equal(K27, Kr)),
                                    'X_equal': bool(np.array_equal(X27, Xr)),
                                    'weights_equal': bool(np.array_equal(W27, Wr)), 'rows': int(len(K27))}
    assert all(v for k, v in chk['reproduce_27km_all369'].items() if k != 'rows'), chk['reproduce_27km_all369']
    # (b) twin and (c) production at 54 km.
    Xe, Ke, We, me = build_bbs_rows(twin, route_cell_map=parent_54km_coastal_snap)
    Xp, Kp, Wp, mp = build_bbs_rows(prod, route_cell_map=parent_54km_coastal_snap)
    rmap = maps['54']
    snapped = rmap[rmap['snapped']]
    chk['bbs_routes'] = {
        'routes_27km': int(m27['n_routes_mapped']), 'routes_54km': int(me['n_routes_mapped']),
        'route_years_27km': int(m27['n_route_years']), 'route_years_54km_twin': int(me['n_route_years']),
        'route_years_54km_prod': int(mp['n_route_years']),
        'cell_years_27km': int(m27['n_cell_years_covered']), 'cell_years_54km': int(me['n_cell_years_covered']),
        'cells_27km': int(m27['n_cells']), 'cells_54km_twin': int(me['n_cells']), 'cells_54km_prod': int(mp['n_cells']),
        'rows_twin': int(len(Ke)), 'rows_prod': int(len(Kp)),
        'rows_dropped_all_zero_twin': int(me['rows_dropped_all_zero']),
        'rows_dropped_all_zero_prod': int(mp['rows_dropped_all_zero']),
        'routes_snapped': int(len(snapped)), 'route_cells27_snapped': int(len(snapped[['row27', 'col27']].drop_duplicates())),
        'max_snap_km': float(snapped['snap_cells'].max() * 54) if len(snapped) else 0.0,
        'mean_snap_km': float(snapped['snap_cells'].mean() * 54) if len(snapped) else 0.0,
        'routes_per_cell_year_54km': float(me['n_route_years'] / me['n_cell_years_covered'])}
    rb = chk['bbs_routes']
    assert rb['routes_27km'] == rb['routes_54km'] and rb['route_years_27km'] == rb['route_years_54km_twin'] \
        == rb['route_years_54km_prod'], rb
    print('[f54-obs] routes', json.dumps(rb), flush=True)

    # Count conservation (twin): sum over 27 km children of mean x n_routes == the 54 km cell-year's mean x n_routes.
    ry = pd.read_csv(a.route_years27)
    ry = ry.merge(rmap.rename(columns={'CountryNum': 'country', 'StateNum': 'state', 'Route': 'route',
                                       'row': 'row54', 'col': 'col54'})[
        ['country', 'state', 'route', 'row54', 'col54', 'row27', 'col27']], on=['country', 'state', 'route'],
        how='left', validate='many_to_one')
    assert ry['row54'].notna().all() and (ry['row'] == ry['row27']).all() and (ry['col'] == ry['col27']).all()
    assert len(ry) == rb['route_years_27km'], (len(ry), rb['route_years_27km'])
    ry[['row54', 'col54']] = ry[['row54', 'col54']].astype('int64')
    n27 = ry.groupby(['row', 'col', 'year']).size()
    n54 = ry.groupby(['row54', 'col54', 'year']).size()
    K27i, Kei = K27.astype('int64'), Ke.astype('int64')
    i27 = pd.MultiIndex.from_arrays(list(K27i.T))
    i54 = pd.MultiIndex.from_arrays(list(Kei.T))
    assert n27.reindex(i27).notna().all() and n54.reindex(i54).notna().all()
    tot27 = np.expm1(X27.astype('float64')) * n27.reindex(i27).to_numpy()[:, None]
    tot54 = np.expm1(Xe.astype('float64')) * n54.reindex(i54).to_numpy()[:, None]
    # children -> destination (snap-aware): every route in a 27 km cell shares that cell's 54 km destination
    dest = ry.drop_duplicates(['row', 'col'])[['row', 'col', 'row54', 'col54']].set_index(['row', 'col'])
    d = dest.reindex(pd.MultiIndex.from_arrays([K27i[:, 0], K27i[:, 1]]))
    assert d['row54'].notna().all()
    agg = pd.DataFrame(tot27).groupby([d['row54'].to_numpy().astype('int64'), d['col54'].to_numpy().astype('int64'),
                                       K27i[:, 2]]).sum()
    agg = agg.reindex(i54)
    assert agg.notna().all().all(), 'a 54 km cell-year without 27 km children'
    rel = np.abs(agg.to_numpy() - tot54) / np.maximum(np.abs(tot54), 1.0)
    chk['count_conservation_twin'] = {'max_rel_diff': float(rel.max()), 'total_counts_27km': float(tot27.sum()),
                                      'total_counts_54km': float(tot54.sum()),
                                      'route_years_per_key_match': bool(int(n54.reindex(i54).sum()) == rb['route_years_27km'])}
    assert rel.max() < 1e-4, chk['count_conservation_twin']
    print('[f54-obs] counts', json.dumps(chk['count_conservation_twin']), flush=True)

    # ---- write points -------------------------------------------------------------------------------------
    pe, pp = a.out / 'bbs_points_eval54', a.out / 'bbs_points_prod54'
    write_points(pe, Xe, Ke, We, points_meta(twin, me, a.out / 'species_eval.csv', Xe, Ke))
    write_points(pp, Xp, Kp, Wp, points_meta(prod, mp, a.out / 'species_prod.csv', Xp, Kp))
    rmap.to_csv(a.out / 'route_cells_27_to_54.csv', index=False)
    # Base dirs read by z_repair_data / z_assim_interp_prep (meta lists + keys; raw counts for the twin's interp).
    for name, K in [('base_eval', Ke), ('base_prod', Kp)]:
        b = a.out / name
        b.mkdir(exist_ok=True)
        np.save(b / 'keys.npy', K)
        (b / 'meta.json').write_text(json.dumps({
            'community': meta['community'], 'dev_species': meta['dev_species'],
            'sealed_species': meta['sealed_species'], 'seal_rule': meta.get('seal_rule'),
            'source_cache_meta': str(a.cache_meta),
            'note': 'FIELD54 Stage B base: species lists of the 27 km cache; keys of the 54 km points.'}, indent=1))
    raw = np.expm1(Xe.astype('float64')).astype('float32')
    ci = {c: i for i, c in enumerate(twin)}
    np.save(a.out / 'base_eval' / 'X_comm.npy', raw[:, [ci[c] for c in meta['community']]])
    np.save(a.out / 'base_eval' / 'X_dev.npy', raw[:, [ci[c] for c in meta['dev_species']]])

    # ---- CBC ----------------------------------------------------------------------------------------------
    D = dict(np.load(a.cbc, allow_pickle=False))
    S = dict(np.load(a.cbc_sealed, allow_pickle=False))
    for k in CIRCLE_KEYS:
        x, y = D[k], S[k]
        same = np.array_equal(x, y, equal_nan=True) if x.dtype.kind == 'f' else np.array_equal(x, y)
        assert same, 'sealed and non-sealed CBC exports differ in ' + k
    sp_n, sp_s = [str(s) for s in D['species']], [str(s) for s in S['species']]
    assert not set(sp_n) & set(sp_s) and 'houfin' not in sp_n + sp_s
    assert set(sp_n) <= nonsealed and set(sp_s) <= set(sealed)
    xs, ys = warp_transform('EPSG:4326', crs54, D['lon'].tolist(), D['lat'].tolist())
    c54 = np.floor((np.asarray(xs) - t54.c) / t54.a).astype(int)
    r54 = np.floor((np.asarray(ys) - t54.f) / t54.e).astype(int)
    r27, c27 = D['row'].astype(int), D['col'].astype(int)
    agree = (r54 == r27 // F) & (c54 == c27 // F)
    inside27 = (r27 >= 0) & (c27 >= 0) & (r27 < ny27) & (c27 < nx27)
    valid27 = inside27 & land27[np.clip(r27, 0, ny27 - 1), np.clip(c27, 0, nx27 - 1)]
    rr, cc, off, keep, dist = parent_cells(np.where(valid27, r27, 0), np.where(valid27, c27, 0), land54, F, 'snap')
    row_new = np.where(valid27, rr, -1).astype(np.int32)
    col_new = np.where(valid27, cc, -1).astype(np.int32)
    raw_land54 = land54[np.clip(r54, 0, H54 - 1), np.clip(c54, 0, W54 - 1)]
    circ = D['circle_idx']
    chk['cbc'] = {
        'circle_years': int(len(circ)), 'circles': int(len(np.unique(circ))),
        'raster_row_col_equals_parent_of_27km': int(agree.sum()), 'raster_disagree': int((~agree).sum()),
        'valid_lists_27km': int(valid27.sum()), 'valid_lists_54km': int((row_new >= 0).sum()),
        'valid_circles_27km': int(len(np.unique(circ[valid27]))),
        'valid_circles_54km': int(len(np.unique(circ[row_new >= 0]))),
        'lists_snapped': int((valid27 & off).sum()), 'circles_snapped': int(len(np.unique(circ[valid27 & off]))),
        'max_snap_km': float(dist[valid27 & off].max() * 54) if (valid27 & off).any() else 0.0,
        'invalid27_lists_kept_out': int((~valid27).sum()),
        'invalid27_lists_on_54km_land_not_added': int((~valid27 & raw_land54).sum()),
        'species_nonsealed': len(sp_n), 'species_sealed': len(sp_s), 'species_production': len(sp_n) + len(sp_s)}
    assert chk['cbc']['raster_disagree'] == 0, chk['cbc']
    assert chk['cbc']['valid_lists_27km'] == chk['cbc']['valid_lists_54km']
    assert chk['cbc']['valid_circles_27km'] == chk['cbc']['valid_circles_54km']
    print('[f54-obs] cbc', json.dumps(chk['cbc']), flush=True)
    base = {k: v for k, v in D.items() if k not in ('row', 'col')}
    np.savez_compressed(a.out / 'cbc_eval54.npz', **base, row=row_new, col=col_new,
                        row27=D['row'], col27=D['col'])
    prodcbc = {**base, 'Y': np.hstack([D['Y'], S['Y']]).astype('float32'),
               'species': np.array(sp_n + sp_s), 'reference': D['reference']}
    np.savez_compressed(a.out / 'cbc_prod54.npz', **prodcbc, row=row_new, col=col_new, row27=D['row'], col27=D['col'])

    # ---- twin masks ---------------------------------------------------------------------------------------
    h27, b27 = np.load(a.holdout27).astype(bool), np.load(a.buffer27).astype(bool)
    assert h27.shape == b27.shape == land27.shape and not (h27 & b27).any()
    src = np.r_[rmap[['row27', 'col27']].to_numpy(), np.column_stack([r27, c27])[valid27]]
    dst = np.r_[rmap[['row', 'col']].to_numpy(), np.column_stack([row_new, col_new])[valid27]]
    h54, b54, snapinfo = coarse_masks(h27, b27, land54, src, dst)
    np.save(a.out / 'holdout_cells54.npy', h54)
    np.save(a.out / 'buffer_cells54.npy', b54)
    # Rows withheld, 27 vs 54 (summer: BBS cell-years / route-years; winter: lists, by cell membership).
    ho_ry27 = h27[ry['row'], ry['col']]; bf_ry27 = b27[ry['row'], ry['col']]
    ry54r, ry54c = ry['row54'].astype(int).to_numpy(), ry['col54'].astype(int).to_numpy()
    ho_ry54 = h54[ry54r, ry54c]; bf_ry54 = b54[ry54r, ry54c]
    v54 = row_new >= 0
    hl27 = valid27 & h27[np.clip(r27, 0, ny27 - 1), np.clip(c27, 0, nx27 - 1)]
    hl54 = v54 & h54[np.clip(row_new, 0, H54 - 1), np.clip(col_new, 0, W54 - 1)]
    bl27 = valid27 & b27[np.clip(r27, 0, ny27 - 1), np.clip(c27, 0, nx27 - 1)]
    bl54 = v54 & b54[np.clip(row_new, 0, H54 - 1), np.clip(col_new, 0, W54 - 1)]
    panel = np.load(a.panel27)['panel']
    sp_cells = panel[panel[:, 2] == 1][:, :2]
    # every 27 km withheld-panel cell's observations land in 54 km holdout cells
    dest_of = {(int(r), int(c)): (int(r5), int(c5)) for r, c, r5, c5 in
               ry.drop_duplicates(['row', 'col'])[['row', 'col', 'row54', 'col54']].to_numpy()}
    panel_ok = all(h54[dest_of[(int(r), int(c))]] for r, c in sp_cells if (int(r), int(c)) in dest_of)
    assert panel_ok
    chk['twin_masks'] = {
        'holdout_cells_27km': int(h27.sum()), 'holdout_cells_54km': int(h54.sum()),
        'buffer_cells_27km': int(b27.sum()), 'buffer_cells_54km': int(b54.sum()),
        'holdout_area_27km_equiv_cells': int(h27.sum()), 'holdout_area_54km_in_27km_cells': int(4 * h54.sum()),
        **snapinfo,
        'withheld_panel_cells_27km': int(len(sp_cells)), 'withheld_panel_parents_all_holdout_54km': bool(panel_ok),
        'bbs_route_years_in_holdout_27km': int(ho_ry27.sum()), 'bbs_route_years_in_holdout_54km': int(ho_ry54.sum()),
        'bbs_route_years_in_buffer_27km': int(bf_ry27.sum()), 'bbs_route_years_in_buffer_54km': int(bf_ry54.sum()),
        'bbs_cell_years_holdout_27km': int(h27[K27[:, 0], K27[:, 1]].sum()),
        'bbs_cell_years_holdout_54km': int(h54[Ke[:, 0], Ke[:, 1]].sum()),
        'bbs_cell_years_buffer_27km': int(b27[K27[:, 0], K27[:, 1]].sum()),
        'bbs_cell_years_buffer_54km': int(b54[Ke[:, 0], Ke[:, 1]].sum()),
        'cbc_lists_holdout_27km': int(hl27.sum()), 'cbc_lists_holdout_54km': int(hl54.sum()),
        'cbc_lists_buffer_27km': int(bl27.sum()), 'cbc_lists_buffer_54km': int(bl54.sum()),
        'route_years_from_27km_holdout_all_in_54km_holdout': bool(ho_ry54[ho_ry27].all()),
        'route_years_from_27km_buffer_all_in_54km_withheld': bool((ho_ry54 | bf_ry54)[bf_ry27].all()),
        'lists_from_27km_holdout_all_in_54km_holdout': bool(hl54[hl27].all())}
    tm = chk['twin_masks']
    assert tm['route_years_from_27km_holdout_all_in_54km_holdout'] and tm['lists_from_27km_holdout_all_in_54km_holdout']
    assert tm['route_years_from_27km_buffer_all_in_54km_withheld']
    print('[f54-obs] masks', json.dumps(tm), flush=True)

    report['outputs'] = {p.name: digest(p) for p in sorted(a.out.rglob('*')) if p.is_file() and p.suffix in
                         ('.npy', '.npz', '.json', '.csv') and p.name != 'DONE.json'}
    report['sealed_read_logged'] = a.sealed_read_logged
    report['source_sha256'] = {str(p): digest(p) for p in [Path(__file__), a.cache_meta, a.points27 / 'X_points.npy',
                                                          a.route_years27, a.cbc, a.cbc_sealed, a.holdout27,
                                                          a.buffer27, a.panel27, a.mask54, a.ref54]}
    (a.out / 'DONE.json').write_text(json.dumps(report, indent=1))
    print('[f54-obs] done ->', a.out, flush=True)


if __name__ == '__main__':
    main()
