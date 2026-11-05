"""D2: do D1's matching (lambda0, n50, kernel) combinations survive coarser cells (10 -> 27 -> 54 km)?

The age model runs on 54 km cells (2,916 km^2): the release and every arrival are averaged over the cell before mate
finding and crowding act. Same total founders (.5 birds/km^2 within 10 km = ~157 birds) placed in the release cell.
For each D1 match and each cell size, also scan n50 x {1/30 .. 3} to find whether ANY threshold restores the match there.
Usage: python run_d2_resolution.py D1_rows.jsonl out_dir
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import proto

MATCH = {15: (30., 100.), 25: (140., 350.), 50: (1000., 2200.)}


def founders(g, total=.5 * np.pi * 100.):
    N0 = np.zeros_like(g['r']); c = (len(g['x']) // 2, len(g['x']) // 2); N0[c] = total / g['cell']; return N0


def run(g, kA, kJ, Sa, Sj, Fmax, n50, key):
    P = proto.hofi_params(Sa=Sa, Sj=Sj, Fmax=Fmax, K=3.76, n50=n50)
    h, _ = proto.simulate('hofi', P, g, kA, kJ, years=51, N0=founders(g), thr=.1)
    R = [h[y][key] for y in (15, 25, 50)]
    return R, all(lo <= h[y][key] <= hi for y, (lo, hi) in MATCH.items())


def main():
    rows = [json.loads(l) for l in open(sys.argv[1])]
    out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    picks = {}
    for r in rows:   # one representative match per (split, lambda0, kernel): the middle of its n50 band
        if r['match']:
            picks.setdefault((r['split'], r['lambda0'], r['kernel_scale']), []).append(r)
    res = []
    grids = {dx: proto.grid(2, dx, 3024. if dx == 54 else 3000.) for dx in (10., 27., 54.)}
    kern = {}
    for (split, lam, ks), band in sorted(picks.items()):
        r0 = band[len(band) // 2]
        for dx, g in grids.items():
            if (dx, ks) not in kern:
                kern[(dx, ks)] = (proto.weibull_kernel(g, .859, 76.893, stay=.22),
                                  proto.weibull_kernel(g, .468, 41.056 * ks, stay=.27))
            kA, kJ = kern[(dx, ks)]
            key = 'r_area'
            R, ok = run(g, kA, kJ, r0['Sa'], r0['Sj'], r0['Fmax'], r0['n50'], key)
            scan = []
            for f in (1 / 30, 1 / 10, 1 / 3, 1., 3.):
                Rs, oks = run(g, kA, kJ, r0['Sa'], r0['Sj'], r0['Fmax'], r0['n50'] * f, key)
                scan.append(dict(n50_factor=f, R=[round(v) for v in Rs], match=oks))
            row = dict(split=split, lambda0=lam, kernel_scale=ks, n50=r0['n50'], dx=dx, R=[round(v) for v in R],
                       match=ok, any_match=[s['n50_factor'] for s in scan if s['match']], scan=scan)
            res.append(row); print(json.dumps({k: row[k] for k in ('split', 'lambda0', 'kernel_scale', 'n50', 'dx', 'R', 'match', 'any_match')}), flush=True)
    (out / 'D2_resolution.json').write_text(json.dumps(res, indent=1) + '\n')


if __name__ == '__main__':
    main()
