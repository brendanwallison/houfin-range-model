"""D1: capability map -- can a time-homogeneous model give the lag AND the late speed as low-density growth rises?

Pre-registration: research/experiments/SANDBOX_INVASION_SHAPE_2026-11.md section D1.
Usage: python run_d1_growth_allee.py out_dir [dx_km]
"""
import itertools, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import proto

MATCH = {15: (30., 100.), 25: (140., 350.), 50: (1000., 2200.)}


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    dx = float(sys.argv[2]) if len(sys.argv) > 2 else 15.
    g = proto.grid(2, dx, 3000.)
    kA = proto.weibull_kernel(g, .859, 76.893, stay=.22)
    kJ = {s: proto.weibull_kernel(g, .468, 41.056 * s, stay=.27) for s in (.33, .6, 1.)}
    N0 = proto.release_patch(g, .5, max(10., dx / 2))
    rows = []
    lams = (1.42, 1.8, 2.2, 2.6, 3.0)
    n50s = np.geomspace(5e-4, .2, 12)
    splits = {'vl': (.67, .67), 'fit': (.55, .30)}
    path = out / 'D1_rows.jsonl'
    with path.open('w') as fh:
        for (split, (Sa, Sj)), lam, ks, n50 in itertools.product(splits.items(), lams, kJ, n50s):
            Fmax = (lam - Sa) / Sj
            P = proto.hofi_params(Sa=Sa, Sj=Sj, Fmax=Fmax, K=3.76, n50=float(n50))
            t0 = time.time()
            hist, _ = proto.simulate('hofi', P, g, kA, kJ[ks], years=51, N0=N0, thr=.1)
            s = proto.lag_and_speeds(hist)
            R = [hist[y]['radius_km'] for y in (15, 25, 50)]
            ok = all(lo <= hist[y]['radius_km'] <= hi for y, (lo, hi) in MATCH.items())
            row = dict(split=split, Sa=Sa, Sj=Sj, Fmax=Fmax, lambda0=lam, kernel_scale=ks,
                       juv_mean_km=round(proto.weibull_mean(.468, 41.056 * ks), 1), n50=float(n50),
                       R15=R[0], R25=R[1], R50=R[2], lag=s['lag_year'], late_speed=s['late_speed'],
                       match=ok, total50=hist[50]['total'], seconds=round(time.time() - t0, 2), dx=dx)
            rows.append(row); fh.write(json.dumps(row) + '\n'); fh.flush()
    # compact table: for each split x lambda0 x kernel, the matching n50 band
    summ = []
    for split, lam, ks in itertools.product(splits, lams, kJ):
        sub = [r for r in rows if r['split'] == split and r['lambda0'] == lam and r['kernel_scale'] == ks]
        m = [r['n50'] for r in sub if r['match']]
        best = min(sub, key=lambda r: sum(abs(np.log(max(r[f'R{y}'], 1.) / np.sqrt(lo * hi))) for y, (lo, hi) in MATCH.items()))
        summ.append(dict(split=split, lambda0=lam, kernel_scale=ks, n_match=len(m),
                         n50_band=[min(m), max(m)] if m else None,
                         closest=dict(n50=best['n50'], R=[best['R15'], best['R25'], best['R50']], lag=best['lag'],
                                      late=best['late_speed'])))
        print(json.dumps(summ[-1]), flush=True)
    (out / 'D1_summary.json').write_text(json.dumps(dict(match_window_km=MATCH, summary=summ), indent=1) + '\n')


if __name__ == '__main__':
    main()
