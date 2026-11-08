"""Expected likelihood shift from extra ABSENCE constraints, evaluated at fixed fits (no refit) -- user 2026-11-06.
Zero cost per cell-year = k log(1 + N / k), N = the cell's total population on the route-count scale, k = .437 (the BBS
reference NB2 dispersion the old pseudo-zeros and the 11-02 rangev2 terms used).
  A: current pseudo-zero cells (1902-39), one per cell-year -- a check of the existing term's size.
  B: footprint cells dropped by the pseudo_zero_subsample thinning (1902-39) -- 'one per grid cell per year'.
  C: rangev2-style post-1940 absences, 1940-65, footprint cells OUTSIDE the introduced eastern envelope (cells within
     330 km of an eastern CBC presence (east of 85W, lat >= 33N) accumulated to that year) -- the 'two hulls' half.
Run per fit with its overlay: absence_cost.py --params NPZ --tag NAME
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(); p.add_argument('--params', type=Path, required=True); p.add_argument('--tag', required=True)
    a = p.parse_args()
    sys.path.insert(0, str(Path(os.environ['TAKEOVER_CODE_ROOT']) / 'research' / 'exp'))
    import jax.numpy as jnp, rasterio
    from rasterio.warp import transform
    from numpyro import handlers
    from harness import Sandbox, ap
    sb = Sandbox(os.environ['SBX_REFINED'])
    lat = {k: jnp.asarray(v) for k, v in sb.lat.items()}
    with np.load(a.params) as z:
        lat.update({k: jnp.asarray(z[k]) for k in z.files if not k.startswith('laplace_route_')})
    tr = handlers.trace(handlers.seed(handlers.substitute(ap.build_model_2d, data=lat), 1)).get_trace(data=sb.data, prior_scale=1.)
    N = (np.asarray(tr['Na_grid']['value']) + np.asarray(tr['Nj_grid']['value'])) * float(sb.data['pop_scalar'])
    yrs = np.asarray(sb.years).astype(int); k = 0.43724203275856455
    cost = lambda x: k * np.log1p(np.maximum(x, 0.) / k)
    # current pseudo-zero cells and the full (pre-thinning) footprint
    idx = np.asarray(sb.data['obs_obsroute_idx']); rr, cc = np.asarray(sb.data['obs_rows']), np.asarray(sb.data['obs_cols'])
    cur = set(zip(rr[idx < 0].tolist(), cc[idx < 0].tolist()))
    z = np.load('/work/07980/bwa386/ls6/houfin/processed/grid54/bbs/bbs_data_obsroute_for_python.npz')
    pzm = np.asarray(z['obs_route_idx']) < 0
    full = set(zip(np.asarray(z['obs_rows'])[pzm].tolist(), np.asarray(z['obs_cols'])[pzm].tolist()))
    dropped = full - cur
    t40 = yrs < 1940
    A = sum(cost(N[t40, r, c]).sum() for r, c in cur)
    B = sum(cost(N[t40, r, c]).sum() for r, c in dropped)
    # introduced eastern envelope from early CBC presences (local footprint cells), 330 km allowance
    cb = np.load('/work/07980/bwa386/ls6/houfin/research/grid54/inputs/cbc_early_20261102/cbc_early.npz')
    lr, lc = np.asarray(sb.data['land_rows']), np.asarray(sb.data['land_cols'])
    with rasterio.open('/work/07980/bwa386/ls6/houfin/processed/grid54/ref_grid_54km.tif') as s:
        xx, yy = rasterio.transform.xy(s.transform, lr, lc); lon, la = map(np.asarray, transform(s.crs, 'EPSG:4326', xx, yy))
    xkm, ykm = np.asarray(xx) / 1e3, np.asarray(yy) / 1e3
    pos = cb['cbc_count'] > 0
    C = 0.; byyear = {}
    fl = np.array(sorted(full)); land_idx = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(lr, lc))}
    fi = np.array([land_idx[(int(r), int(c))] for r, c in fl if (int(r), int(c)) in land_idx])
    seeds = set()
    for t, y in enumerate(yrs):
        if y < 1940 or y > 1965:
            continue
        sel = pos & (cb['cbc_t'] == t)
        for j in np.flatnonzero(sel):
            ci = cb['cbc_circle'][j]; src = cb['cbc_source_idx'][ci]; w = cb['cbc_w_local'][ci]
            for s_, w_ in zip(src, w):
                if s_ >= 0 and w_ > 0 and lon[s_] > -85 and la[s_] >= 33:
                    seeds.add(int(s_))
        if seeds:
            sd = np.array(sorted(seeds))
            d = np.sqrt((xkm[fi, None] - xkm[sd][None]) ** 2 + (ykm[fi, None] - ykm[sd][None]) ** 2).min(1)
            outside = fi[d > 330.]
        else:
            outside = fi
        cy = float(cost(N[t, lr[outside], lc[outside]]).sum()); byyear[int(y)] = cy; C += cy
    out = dict(tag=a.tag, current_cells=len(cur), footprint_cells=len(full), dropped_cells=len(dropped),
               A_current_pseudozero_cost=float(A), B_dropped_cells_cost=float(B), C_post1940_outside_two_hulls_cost=float(C),
               C_by_year={y: round(v, 2) for y, v in byyear.items() if y % 5 == 0})
    print(json.dumps(out))


if __name__ == '__main__':
    main()
