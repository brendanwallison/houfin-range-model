"""Where does potential growth go? Potential vs realised per-cell growth in the fitted replay, by era (unit-free).

For eastern cells, each year: potential lambda (density-, Allee- and movement-free, from the captured rates; harness
potential_lambda), realised per-cell growth N_t+1 / N_t, relative density N / K, and the share of cells near K.
Cells are split by state: FRONT (N/K < .1), FILLING (.1 <= N/K < .7), NEAR K (>= .7); only cells with N above 1% of
the reference arm's modern occupied median are counted (occupied). Also the regional decomposition of the East's
total growth into growth inside cells occupied at both t and t+1 vs newly occupied cells (spread).
Usage (inside sbx_env.sh): python growth_gap.py --out DIR
"""
import argparse, json, os
from pathlib import Path
import numpy as np
from harness import Sandbox, capture, to_land, potential_lambda

ERAS = [(1941, 1955), (1956, 1965), (1966, 1975), (1976, 1990), (1991, 2010)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    sb = Sandbox(os.environ['SBX_REFINED'])
    row = sb.run(dict(name='fitted'), keep_density=True)
    N = row.pop('_density')
    x, _ = capture(sb.data, sb.lat)
    T, Nx = len(sb.years), int(np.asarray(sb.data['land_mask']).shape[1])
    Sa, Sj, F, K = (to_land(x[k], sb.rows, sb.cols, T, Nx) for k in ('Sa_flat', 'Sj_flat', 'Fmax_flat', 'K_flat'))
    lam = potential_lambda(Sa, Sj, F, bool(x.get('yearling_breeding')))
    occ_thr = .01 * sb.base_med
    east = sb.region['east']; yrs = sb.years
    out = dict(yearling=bool(x.get('yearling_breeding')), eras={})
    for a0, b0 in ERAS:
        rec = {}
        idx = [i for i, y in enumerate(yrs[:-1]) if a0 <= y <= b0]
        for state, lo, hi in (('front', 0, .1), ('filling', .1, .7), ('near_K', .7, 1e9)):
            pl, rl, rk = [], [], []
            for i in idx:
                n0, n1 = N[i], N[i + 1]; rel = n0 / np.maximum(K[i], 1e-12)
                m = east & (n0 >= occ_thr) & (rel >= lo) & (rel < hi)
                pl += list(lam[i][m]); rl += list(n1[m] / n0[m]); rk += list(rel[m])
            if pl:
                rec[state] = dict(n_cell_years=len(pl), potential_lambda=float(np.median(pl)),
                                  realised_lambda=float(np.median(rl)), rel_density=float(np.median(rk)))
        # regional decomposition: growth of the East total from cells occupied at both ends vs newly occupied
        inside, new, total0 = 0., 0., 0.
        for i in idx:
            n0, n1 = N[i][east], N[i + 1][east]
            both = (n0 >= occ_thr) & (n1 >= occ_thr); newly = (n0 < occ_thr) & (n1 >= occ_thr)
            inside += (n1[both] - n0[both]).sum(); new += n1[newly].sum(); total0 += n0.sum()
        occ_share = np.mean([((N[i][east] >= occ_thr) & (N[i][east] / np.maximum(K[i][east], 1e-12) >= .7)).sum()
                             / max((N[i][east] >= occ_thr).sum(), 1) for i in idx])
        rec['east_total_growth_share'] = dict(inside_occupied=float(inside / (inside + new)) if inside + new else None,
                                              newly_occupied=float(new / (inside + new)) if inside + new else None)
        rec['share_of_occupied_cells_near_K'] = float(occ_share)
        out['eras'][f'{a0}-{b0}'] = rec
    (a.out / 'growth_gap.json').write_text(json.dumps(out, indent=1) + '\n')
    for era, rec in out['eras'].items():
        print(era, {k: (({kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in v.items()}) if isinstance(v, dict) else round(v, 3))
                    for k, v in rec.items()})


if __name__ == '__main__':
    main()
