"""Native-range PRESENCE constraints 1902-1940 (follow-up to the 2026-11-07 spin-up result: nothing in the likelihood says
the early-century native range was occupied, so the fit settles a 1902 West ~17% smaller than the seed and grows it ~80% by
2010).

WHAT THE SOURCE IS (read before interpreting). The seed map (initpop_latent) is NOT a 1902-40 range reconstruction: it is the
convex hull of PRE-1970 western BBS presences, i.e. 1966-69 routes (src/data/preprocess/bbs.py
generate_core_margin_initialization). Asserting it in 1902-40 is a BACK-CAST ASSUMPTION ("cells occupied in 1966-69 were
occupied in 1902-40"), not 1902-40 data. Footprints:
  bbs6669  -- cells with a positive BBS route count in 1966-69 west of --lon_max (evidence-backed occupancy; DEFAULT)
  core     -- the seed's core hull cells (initpop_latent == 1)
  seed     -- every seed cell (margin + core; the convex hull includes unoccupied desert / montane interior)
Cost in laplace_fit (--range_presence): w * -log(1 - (1 + N/k)^-k), N = cell annual total on the route-count scale, k =
--presence_k: the censored NB2 event paired with the absences' k log(1 + N/k) (same detection model, opposite event).
Weight: --w per cell-year (0.1 = one cell-decade unit per cell, the rangev2 thinning of historical CBC presences).
Usage: range_presence_build.py --input INPUT_DIR --source bbs6669 --w 0.1 --out NPZ
"""
import argparse, pickle
import numpy as np


def main():
    p = argparse.ArgumentParser(); p.add_argument('--input', required=True); p.add_argument('--out', required=True)
    p.add_argument('--source', choices=('bbs6669', 'core', 'seed'), default='bbs6669')
    p.add_argument('--y0', type=int, default=1902); p.add_argument('--y1', type=int, default=1940)
    p.add_argument('--w', type=float, default=0.1); p.add_argument('--lon_max', type=float, default=-100.0)
    a = p.parse_args()
    import rasterio
    from rasterio.warp import transform
    m = pickle.load(open(f'{a.input}/metadata.pkl', 'rb'))
    yrs = np.asarray(m['years']).astype(int)
    lr, lc = np.asarray(m['land_rows']), np.asarray(m['land_cols'])
    with rasterio.open('/work/07980/bwa386/ls6/houfin/processed/grid54/ref_grid_54km.tif') as s:
        xx, yy = rasterio.transform.xy(s.transform, lr, lc); lon, la = map(np.asarray, transform(s.crs, 'EPSG:4326', xx, yy))
    lut = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(lr, lc))}
    ip = np.asarray(m['initpop_latent']); ipl = ip[lr, lc]
    west = lon < a.lon_max
    if a.source == 'bbs6669':
        ty = yrs[np.asarray(m['obs_time_indices'])]
        sel = (np.asarray(m['obs_obsroute_idx']) >= 0) & (ty >= 1966) & (ty <= 1969) & (np.asarray(m['observed_results']) > 0)
        cells = {lut[(int(r), int(c))] for r, c in zip(np.asarray(m['obs_rows'])[sel], np.asarray(m['obs_cols'])[sel])
                 if (int(r), int(c)) in lut}
        on = np.zeros(len(lr), bool); on[list(cells)] = True
    elif a.source == 'core':
        on = ipl >= 1.0 - 1e-9
    else:
        on = ipl > 0
    on &= west
    idx = np.flatnonzero(on)
    print(f'source {a.source}: {idx.size} western cells (lon < {a.lon_max}); seed cells {(ipl > 0).sum()} '
          f'(core {(ipl >= 1 - 1e-9).sum()}); in seed: {(ipl[idx] > 0).sum()}; lat {la[idx].min():.1f}..{la[idx].max():.1f}, '
          f'lon {lon[idx].min():.1f}..{lon[idx].max():.1f}')
    T, R, C = [], [], []
    for ti, y in enumerate(yrs):
        if a.y0 <= y <= a.y1:
            T += [ti] * idx.size; R += list(lr[idx]); C += list(lc[idx])
    np.savez_compressed(a.out, t=np.asarray(T, np.int32), row=np.asarray(R, np.int32), col=np.asarray(C, np.int32),
                        w=np.full(len(T), a.w), cell=np.repeat(idx[None], len(set(T)), 0).ravel().astype(np.int32),
                        source=a.source, y0=a.y0, y1=a.y1)
    print(f'{len(T)} cell-years {a.y0}-{a.y1}, weight {a.w} each (total weight {a.w * len(T):.1f})')


if __name__ == '__main__':
    main()
