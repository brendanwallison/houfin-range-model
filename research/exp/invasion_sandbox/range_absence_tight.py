"""Tighter native-range halo for absences (user, 2026-11-06: "a deeper Great Plains trough through a few more zeros, e.g.
reducing the buffer distance").

The lineage asserts pre-1940 zeros only outside a 700 km halo around the convex hull of pre-1970 western BBS records
(src/data/preprocess/bbs.py BUFFER_DISTANCE_METERS), and the rangev2 1940-65 absences use the same footprint. This adds
absences for cells whose distance to the model's NATIVE seed range (initpop_latent > 0 west of 100W; a proxy for the
hull) lies between --halo_km and the cells already covered (existing pseudo-zero / rangev2 cells are excluded, so no
double counting), for 1902-1965, excluding the introduced eastern envelope after 1940 (lon > -85). Weight 1 per cell-year.
Output in the range_absence_v2 format, MERGED with an existing file (--base) so laplace_fit takes one --range_absence.
Usage: range_absence_tight.py --input INPUT_DIR --base range_absence_v2.npz --halo_km 400 --out NPZ
"""
import argparse, pickle
import numpy as np


def main():
    p = argparse.ArgumentParser(); p.add_argument('--input', required=True); p.add_argument('--base', required=True)
    p.add_argument('--halo_km', type=float, required=True); p.add_argument('--out', required=True)
    p.add_argument('--lon_min', type=float, default=-106.0, help='only add cells east of this longitude (Plains band; the seed map understates the native West)')
    a = p.parse_args()
    import rasterio
    from rasterio.warp import transform
    m = pickle.load(open(f'{a.input}/metadata.pkl', 'rb'))
    yrs = np.asarray(m['years']).astype(int)
    lr, lc = np.asarray(m['land_rows']), np.asarray(m['land_cols'])
    with rasterio.open('/work/07980/bwa386/ls6/houfin/processed/grid54/ref_grid_54km.tif') as s:
        xx, yy = rasterio.transform.xy(s.transform, lr, lc); lon, la = map(np.asarray, transform(s.crs, 'EPSG:4326', xx, yy))
    xkm, ykm = np.asarray(xx) / 1e3, np.asarray(yy) / 1e3
    ip = np.asarray(m['initpop_latent'])
    ip = ip[lr, lc] if ip.ndim == 2 else ip
    native = (ip > 0) & (lon < -100)
    nx_, ny_ = xkm[native], ykm[native]
    d = np.array([np.min(np.hypot(nx_ - x, ny_ - y)) for x, y in zip(xkm, ykm)])
    # cells already constrained: the lineage's pseudo-zero rows (route idx < 0) and the base absence file
    idx = np.asarray(m['obs_obsroute_idx']); rr, cc = np.asarray(m['obs_rows']), np.asarray(m['obs_cols'])
    lut = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(lr, lc))}
    pz_cells = {lut[(int(r), int(c))] for r, c in zip(rr[idx < 0], cc[idx < 0]) if (int(r), int(c)) in lut}
    b = np.load(a.base); base_cells = set()
    bc = b['cell'] if 'cell' in b.files else np.array([lut.get((int(r), int(c)), -1) for r, c in zip(b['row'], b['col'])])
    base_cells |= set(int(x) for x in bc if x >= 0)
    covered = pz_cells | base_cells
    new = np.array([i for i in range(len(lr)) if d[i] > a.halo_km and i not in covered and not native[i] and lon[i] >= a.lon_min])
    print(f'halo {a.halo_km} km: {new.size} new cells (distance to native seed range > halo, not already constrained); '
          f'lon range of new cells {lon[new].min():.1f} .. {lon[new].max():.1f}' if new.size else 'no new cells')
    T, C, R, W = [list(b[k]) for k in ('t', 'col', 'row', 'w')] if 'row' in b.files else ([], [], [], [])
    added = 0
    for ti, y in enumerate(yrs):
        if y < 1902 or y > 1965:
            continue
        for i in new:
            if y >= 1940 and lon[i] > -85:      # introduced eastern envelope region after the release: not asserted
                continue
            T.append(ti); R.append(int(lr[i])); C.append(int(lc[i])); W.append(1.0); added += 1
    out = {k: b[k] for k in b.files}
    out.update(t=np.asarray(T, np.int32), row=np.asarray(R, np.int32), col=np.asarray(C, np.int32), w=np.asarray(W, float))
    np.savez_compressed(a.out, **out)
    print(f'added {added} cell-years; total {len(T)} (base {len(b["t"])})')


if __name__ == '__main__':
    main()
