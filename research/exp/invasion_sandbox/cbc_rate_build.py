"""CBC RATE term input (2026-11-06, FINAL-MODEL ROUND): invasion-era eastern CBC counts with effort, for a within-circle
rate-of-increase likelihood (free source scale + circle effects absorb the winter / breeding level conflict).

Keeps circle-years with summer label in [--y0, --y1], effort recorded, footprint touching lon >= --lonmin, and only
circles with >= 1 positive count in the window. Writes rows (t, circle, count, log effort centred at the median) plus
the presence term's footprint arrays (source cells x weights; local U seasonal), and circle distance from the release.
Usage (takeover runner): cbc_rate_build.py --out NPZ [--y0 1947 --y1 1965 --lonmin -90]
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(os.environ['TAKEOVER_CODE_ROOT']) / 'research' / 'exp'))
CBC = os.environ.get('CBC_EARLY', '/work/07980/bwa386/ls6/houfin/research/grid54/inputs/cbc_early_20261102/cbc_early.npz')


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--y0', type=int, default=1947); p.add_argument('--y1', type=int, default=1965); p.add_argument('--lonmin', type=float, default=-90.)
    a = p.parse_args()
    from harness import Sandbox
    sb = Sandbox(os.environ['SBX_REFINED'])
    z = np.load(CBC); years = np.asarray(sb.years).astype(int)
    t, ci, y = z['cbc_t'], z['cbc_circle'], z['cbc_count']
    eff_ok = z['cbc_effort_missing'] == 0
    src = z['cbc_source_idx']; w = np.where(src >= 0, np.maximum(z['cbc_w_local'], z['cbc_w_seasonal']), 0.)
    wl = np.where(src >= 0, z['cbc_w_local'], 0.)
    s0 = np.maximum(src, 0)
    # circle centre ~ local-weighted mean of its source cells' coordinates
    lon_c = (wl * sb.lon[s0]).sum(1) / np.maximum(wl.sum(1), 1e-12)
    xx_c = (wl * sb.xx[s0]).sum(1) / np.maximum(wl.sum(1), 1e-12); yy_c = (wl * sb.yy[s0]).sum(1) / np.maximum(wl.sum(1), 1e-12)
    from harness import capture
    x, _ = capture(sb.data, sb.lat); il = np.asarray(x['inv_locations']).reshape(-1, 2)[0]
    rel = int(sb.lookup[il[0], il[1]]); d_rel = np.hypot(xx_c - sb.xx[rel], yy_c - sb.yy[rel]) / 1e3
    yr = years[t]
    keep = (yr >= a.y0) & (yr <= a.y1) & eff_ok & (lon_c[ci] >= a.lonmin) & (wl.sum(1)[ci] > 0)
    pos_c = np.unique(ci[keep & (y > 0)]); keep &= np.isin(ci, pos_c)
    circles, cidx = np.unique(ci[keep], return_inverse=True)
    le = z['cbc_log_effort'][keep]; le = le - np.median(le)
    out = dict(t=t[keep].astype(np.int32), circle=cidx.astype(np.int32), count=y[keep], log_effort=le,
               src=s0[circles].astype(np.int32), w=w[circles], d_release_km=d_rel[circles], lon=lon_c[circles],
               circle_orig=circles.astype(np.int32))
    np.savez_compressed(a.out, **out)
    n = keep.sum(); near = d_rel[circles] < 150
    summ = dict(rows=int(n), circles=int(circles.size), positives=int((y[keep] > 0).sum()), circles_within_150km=int(near.sum()),
                rows_within_150km=int(near[cidx].sum()), years={int(v): int((yr[keep] == v).sum()) for v in range(a.y0, a.y1 + 1)})
    print(json.dumps(summ))


if __name__ == '__main__':
    main()
