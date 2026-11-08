"""Native-range hull from pre-CUTOFF western BBS presences + pre-1940 museum specimens (user, 2026-11-08).

Why: the native seed (src/data/preprocess/bbs.py generate_core_margin_initialization) is the convex hull of pre-1970
western BBS presences. BBS barely samples Mexico, so the hull stops near the border: 36 / 365 Mexican cells are seeded,
and the pre-1940 pseudo-zero BBS rows and the range-absence file ("outside hull + 400 km halo") assert ABSENCE on the
Mexican plateau (23-26N) -- the native core. Museum specimens 1859-1939 (GBIF, 3,287 Mexican records) mark the native
range directly.

What it changes (a new input variant; the source directory is never modified):
  * initpop_latent: cells inside the new hull that were unseeded get the margin value (initpop_seed.margin_fraction_of_core);
    existing seed values (core = 1, coarsened margins) are kept.
  * pseudo-zero BBS rows (obs_obsroute_idx < 0, 1902-39) within HALO km of the new hull are DROPPED (all obs-length arrays).
  * the range-absence npz: rows within HALO km of the new hull are dropped.
Specimens: pre-1940 only (the eastern release was 1940; earlier eastern records are cage birds) and west of the same
western limit the BBS hull uses (col < 0.66 Nx).

Usage:
  native_hull_build.py report --input DIR [--absence NPZ]
  native_hull_build.py build  --input DIR --out DIR --cutoff 1970 [--absence NPZ --absence_out NPZ]
"""
import argparse, json, os, pickle, sys
from pathlib import Path
import numpy as np

RECORDS = '/mnt/e/Datasets/houfin/external/gbif_hofi_1800_1940/records.csv'
REF = '/mnt/e/Datasets/houfin/grid54/ref_grid_54km.tif'
CEC = '/mnt/e/Datasets/houfin/grid54/regions/cec_level1_54km.tif'
HALO_KM = 400.
MARGIN = 0.5            # initpop_seed.margin_fraction_of_core in the current overlays
MEX_BORDER = np.array([(-118., 32.5), (-117.1, 32.5), (-114.7, 32.7), (-111.0, 31.3), (-108.2, 31.3), (-108.2, 31.8),
                       (-106.5, 31.8), (-104.5, 29.6), (-103.0, 29.0), (-101.4, 29.8), (-99.5, 27.5), (-97.2, 25.9)])


def grid(m):
    import rasterio
    from rasterio.warp import transform as tf
    lr, lc = np.asarray(m['land_rows']), np.asarray(m['land_cols'])
    with rasterio.open(REF) as s:
        xx, yy = rasterio.transform.xy(s.transform, lr, lc, offset='center'); xx, yy = np.asarray(xx), np.asarray(yy)
        lon, lat = map(np.asarray, tf(s.crs, 'EPSG:4326', list(xx), list(yy)))
        crs, T = s.crs, s.transform
    with rasterio.open(CEC) as s:
        eco = np.asarray(s.read(1))[lr, lc]
    mex = (lat < np.interp(lon, MEX_BORDER[:, 0], MEX_BORDER[:, 1])) & (lon < -97.)
    return lr, lc, xx / 1e3, yy / 1e3, lon, lat, eco, mex, crs, T


def specimen_points(crs, T, Nx, land_lookup, mexico_only=True):
    """Pre-1940 specimen points. mexico_only (default): records with country == Mexico -- a single convex hull over ALL
    western specimens stretches across the Great Plains (a few far-east pre-1940 records seed 524 GP / 246 ETF cells),
    contradicting the specimens' own evidence that the central Plains were empty (report 2026-11-08)."""
    import pandas as pd
    from rasterio.warp import transform as tf
    R = pd.read_csv(RECORDS, usecols=['year', 'decimalLatitude', 'decimalLongitude', 'country'])
    R = R[(R.year < 1940) & R.decimalLatitude.notna() & R.decimalLongitude.notna()]
    if mexico_only:
        R = R[R.country == 'Mexico']
    x, y = map(np.asarray, tf('EPSG:4326', crs, R.decimalLongitude.tolist(), R.decimalLatitude.tolist()))
    col = np.floor((x - T.c) / T.a).astype(int); row = np.floor((y - T.f) / T.e).astype(int)
    ok = (row >= 0) & (row < land_lookup.shape[0]) & (col >= 0) & (col < land_lookup.shape[1])
    ok[ok] &= land_lookup[row[ok], col[ok]] >= 0
    ok &= col < int(Nx * 0.66)
    return x[ok] / 1e3, y[ok] / 1e3, len(R), int(ok.sum())


def hull_mask(px, py, cx, cy):
    from scipy.spatial import ConvexHull
    from matplotlib.path import Path as MPath
    P = np.column_stack([px, py]); h = ConvexHull(P); poly = P[h.vertices]
    inside = MPath(poly).contains_points(np.column_stack([cx, cy]))
    # distance from each cell centre to the hull polygon (0 inside)
    a = poly; b = np.roll(poly, -1, axis=0); d = np.full(cx.size, np.inf)
    for (ax_, ay_), (bx_, by_) in zip(a, b):
        vx, vy = bx_ - ax_, by_ - ay_; L2 = vx * vx + vy * vy
        t = np.clip(((cx - ax_) * vx + (cy - ay_) * vy) / max(L2, 1e-12), 0, 1)
        d = np.minimum(d, np.hypot(cx - (ax_ + t * vx), cy - (ay_ + t * vy)))
    d[inside] = 0.
    return inside, d


def union_hulls(point_sets, cx, cy):
    """Cells inside ANY of the hulls; distance = min distance to any hull."""
    inside = np.zeros(cx.size, bool); d = np.full(cx.size, np.inf)
    for px, py in point_sets:
        if len(px) >= 3:
            i_, d_ = hull_mask(np.asarray(px), np.asarray(py), cx, cy); inside |= i_; d = np.minimum(d, d_)
    return inside, d


def build_points(m, cutoff, with_spec, g):
    lr, lc, cx, cy, lon, lat, eco, mex, crs, T = g
    Nx = int(np.asarray(m['land_mask']).shape[1]); yrs = np.asarray(m['years']).astype(int)
    lookup = -np.ones(np.asarray(m['land_mask']).shape, int); lookup[lr, lc] = np.arange(lr.size)
    y = np.asarray(m['observed_results'], float); idx = np.asarray(m['obs_obsroute_idx'])
    oy = yrs[np.asarray(m['obs_time_indices'])]; orow, ocol = np.asarray(m['obs_rows']), np.asarray(m['obs_cols'])
    sel = (idx >= 0) & (y > 0) & (oy <= cutoff) & (ocol < int(Nx * 0.66))
    cells = np.unique(lookup[orow[sel], ocol[sel]]); cells = cells[cells >= 0]
    sets = [(cx[cells], cy[cells])]; info = dict(bbs_cells=int(cells.size))
    if with_spec:
        sx, sy, n_all, n_ok = specimen_points(crs, T, Nx, lookup)
        sets.append((sx, sy)); info.update(spec_records_pre1940_mexico=n_all, spec_points_used=n_ok)
    return sets, info, lookup


def summarize(tag, inside, d, g, ip_land, pz_cell, ab_cell):
    lr, lc, cx, cy, lon, lat, eco, mex, *_ = g
    names = {9: 'GP', 8: 'ETF', 10: 'Deserts', 6: 'NWMtns', 12: 'SSHighl', 13: 'TSierras', 11: 'MedCal'}
    new = inside & (ip_land <= 0)
    by = {nm: int((new & (eco == c)).sum()) for c, nm in names.items() if (new & (eco == c)).any()}
    near = d <= HALO_KM
    out = dict(tag=tag, hull_cells=int(inside.sum()), newly_seeded=int(new.sum()), new_by_region=by,
               mexico_seeded=f'{int((inside & mex).sum())}/{int(mex.sum())}',
               pseudo_zero_rows_dropped=int(near[pz_cell].sum()), pseudo_zero_rows_dropped_mexico=int((near & mex)[pz_cell].sum()))
    if ab_cell is not None:
        ok = ab_cell >= 0
        out.update(absence_rows_dropped=int(near[ab_cell[ok]].sum()), absence_rows_dropped_mexico=int((near & mex)[ab_cell[ok]].sum()))
    print(json.dumps(out), flush=True)
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['report', 'build']); p.add_argument('--input', required=True)
    p.add_argument('--out'); p.add_argument('--cutoff', type=int, default=1970); p.add_argument('--no_spec', action='store_true')
    p.add_argument('--absence'); p.add_argument('--absence_out')
    a = p.parse_args()
    m = pickle.load(open(f'{a.input}/metadata.pkl', 'rb')); g = grid(m); lr, lc, cx, cy, lon, lat, eco, mex, *_ = g
    ip_land = np.asarray(m['initpop_latent'])[lr, lc]
    lookup = -np.ones(np.asarray(m['land_mask']).shape, int); lookup[lr, lc] = np.arange(lr.size)
    idx = np.asarray(m['obs_obsroute_idx']); pz = idx < 0
    pz_cell = lookup[np.asarray(m['obs_rows'])[pz], np.asarray(m['obs_cols'])[pz]]
    ab = np.load(a.absence) if a.absence else None
    ab_cell = lookup[np.asarray(ab['row']), np.asarray(ab['col'])] if ab is not None else None
    if a.mode == 'report':
        for cutoff in (1970, 1975):
            for spec in (False, True):
                sets, info, _ = build_points(m, cutoff, spec, g)
                inside, d = union_hulls(sets, cx, cy)
                print(json.dumps(info))
                summarize(f'bbs<={cutoff}' + (' + specimens<1940' if spec else ''), inside, d, g, ip_land, pz_cell, ab_cell)
        return
    sets, info, _ = build_points(m, a.cutoff, not a.no_spec, g)
    inside, _d_union = union_hulls(sets, cx, cy)
    # Drop absences / pseudo-zeros only near the NEW (specimen) hull: the existing BBS-hull halo rule is left exactly as built,
    # so this variant changes Mexico and nothing else (measured from the union, 3,136 non-Mexico absences within 400 km of
    # the old BBS hull would also go -- a separate question, not this fix).
    d = hull_mask(np.asarray(sets[1][0]), np.asarray(sets[1][1]), cx, cy)[1] if len(sets) > 1 else np.full(cx.size, np.inf)
    s = summarize(f'BUILD bbs<={a.cutoff}' + ('' if a.no_spec else ' + specimens<1940'), inside, d, g, ip_land, pz_cell, ab_cell)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    m2 = dict(m)
    ip = np.asarray(m['initpop_latent']).copy(); ipl = ip[lr, lc]
    ipl = np.where(inside & (ipl <= 0), MARGIN, ipl); ip[lr, lc] = ipl; m2['initpop_latent'] = ip.astype(np.asarray(m['initpop_latent']).dtype)
    near = d <= HALO_KM
    drop = np.zeros(idx.size, bool); drop[np.flatnonzero(pz)[near[pz_cell]]] = True
    n = idx.size
    for k, v in m.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n and k.startswith('obs'):
            m2[k] = v[~drop]
    m2['observed_results'] = np.asarray(m['observed_results'])[~drop]
    m2['native_hull_variant'] = dict(info, **s, cutoff=a.cutoff, specimens=not a.no_spec, halo_km=HALO_KM, margin=MARGIN,
                                     source=str(a.input), rows_dropped=int(drop.sum()))
    pickle.dump(m2, open(out / 'metadata.pkl', 'wb'))
    for f in os.listdir(a.input):
        if f == 'metadata.pkl':
            continue
        src = Path(a.input) / f
        os.symlink(os.path.realpath(src), out / f)
    if ab is not None and a.absence_out:
        keep = ~(ab_cell >= 0) | ~near[np.maximum(ab_cell, 0)]
        np.savez_compressed(a.absence_out, **{k: np.asarray(ab[k])[keep] for k in ab.files})
        print(f'absences kept {int(keep.sum())} / {keep.size}')
    print(f'built {out}: obs rows {n} -> {n - int(drop.sum())}; seeded land cells {int((ip_land > 0).sum())} -> {int((ipl > 0).sum())}')


if __name__ == '__main__':
    main()
