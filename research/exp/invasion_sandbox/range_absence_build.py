"""Build the rangev2-restoration ABSENCE set for laplace_fit.py --range_absence (user, 2026-11-06).
Adds what the current lineage lacks, without double counting the existing 1902-39 pseudo-zero BBS rows:
  (1) 1902-39: footprint cells dropped by pseudo_zero_subsample (one per grid cell per year on the full footprint);
  (2) 1940-65: full-footprint cells OUTSIDE the introduced eastern envelope (cells within 330 km of eastern CBC presences,
      east of 85W and lat >= 33N, accumulated to that year) -- the 'two hulls' (the footprint already excludes the native
      range). Weight 1 per cell-year.
Usage (any python with numpy + rasterio): range_absence_build.py --input INPUT_DIR --out NPZ
"""
import argparse, pickle
import numpy as np


def main():
    p = argparse.ArgumentParser(); p.add_argument('--input', required=True); p.add_argument('--out', required=True)
    a = p.parse_args()
    import rasterio
    from rasterio.warp import transform
    m = pickle.load(open(f'{a.input}/metadata.pkl', 'rb'))
    yrs = np.asarray(m['years']).astype(int)
    idx = np.asarray(m['obs_obsroute_idx']); rr, cc = np.asarray(m['obs_rows']), np.asarray(m['obs_cols'])
    cur = set(zip(rr[idx < 0].tolist(), cc[idx < 0].tolist()))
    z = np.load('/work/07980/bwa386/ls6/houfin/processed/grid54/bbs/bbs_data_obsroute_for_python.npz')
    pzm = np.asarray(z['obs_route_idx']) < 0
    full = sorted(set(zip(np.asarray(z['obs_rows'])[pzm].tolist(), np.asarray(z['obs_cols'])[pzm].tolist())))
    lr, lc = np.asarray(m['land_rows']), np.asarray(m['land_cols'])
    land_idx = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(lr, lc))}
    full = [rc for rc in full if rc in land_idx]
    with rasterio.open('/work/07980/bwa386/ls6/houfin/processed/grid54/ref_grid_54km.tif') as s:
        xx, yy = rasterio.transform.xy(s.transform, lr, lc); lon, la = map(np.asarray, transform(s.crs, 'EPSG:4326', xx, yy))
    xkm, ykm = np.asarray(xx) / 1e3, np.asarray(yy) / 1e3
    T, R, C = [], [], []
    for t, y in enumerate(yrs):
        if y < 1940:
            for r, c in full:
                if (r, c) not in cur:
                    T.append(t); R.append(r); C.append(c)
    cb = np.load('/work/07980/bwa386/ls6/houfin/research/grid54/inputs/cbc_early_20261102/cbc_early.npz')
    pos = cb['cbc_count'] > 0; seeds = set()
    fi = np.array([land_idx[rc] for rc in full])
    for t, y in enumerate(yrs):
        if y < 1940 or y > 1965:
            continue
        for j in np.flatnonzero(pos & (cb['cbc_t'] == t)):
            ci = cb['cbc_circle'][j]
            for s_, w_ in zip(cb['cbc_source_idx'][ci], cb['cbc_w_local'][ci]):
                if s_ >= 0 and w_ > 0 and lon[s_] > -85 and la[s_] >= 33:
                    seeds.add(int(s_))
        if seeds:
            sd = np.array(sorted(seeds))
            d = np.sqrt((xkm[fi, None] - xkm[sd][None]) ** 2 + (ykm[fi, None] - ykm[sd][None]) ** 2).min(1)
            out = fi[d > 330.]
        else:
            out = fi
        T += [t] * len(out); R += lr[out].tolist(); C += lc[out].tolist()
    T, R, C = map(np.asarray, (T, R, C))
    np.savez_compressed(a.out, t=T, row=R, col=C, w=np.ones(T.size))
    print(f'pre-1940 dropped-cell rows {(yrs[T] < 1940).sum()}, 1940-65 outside-hull rows {(yrs[T] >= 1940).sum()}, footprint {len(full)}, current {len(cur)}')


if __name__ == '__main__':
    main()
