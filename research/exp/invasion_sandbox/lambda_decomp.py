"""Which vital rate carries lambda inflation? Attribute the change in potential lambda to Sa, Sj and Fmax.

Per eastern / plains / western land cell, potential lambda (harness.potential_lambda: Leslie, density-, Allee- and
movement-free) at y0 and y1 from the captured fitted rates; one-at-a-time swaps (each rate moved to its y1 value with
the others at y0) and the Shapley average of the two orders over all 3! permutations, so the parts sum exactly to
the total change. Medians over cells. Unit-free. (Under Veit reproduction the Leslie lambda is approximate.)
Usage (inside sbx_env.sh): python lambda_decomp.py --out DIR [--y0 1940 --y1 2010]
"""
import argparse, itertools, json, os
from pathlib import Path
import numpy as np
from harness import Sandbox, capture, to_land, potential_lambda


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--y0', type=int, default=1940); p.add_argument('--y1', type=int, default=2010)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    sb = Sandbox(os.environ['SBX_REFINED'])
    x, _ = capture(sb.data, sb.lat)
    T, Nx = len(sb.years), int(np.asarray(sb.data['land_mask']).shape[1])
    R = {k: to_land(x[f'{k}_flat'], sb.rows, sb.cols, T, Nx) for k in ('Sa', 'Sj', 'Fmax')}
    i0, i1 = int(np.flatnonzero(sb.years == a.y0)[0]), int(np.flatnonzero(sb.years == a.y1)[0])
    yb = bool(x.get('yearling_breeding'))
    lam = lambda r: potential_lambda(r['Sa'], r['Sj'], r['Fmax'], yb)
    r0 = {k: v[i0] for k, v in R.items()}; r1 = {k: v[i1] for k, v in R.items()}
    total = lam(r1) - lam(r0)
    shap = {k: np.zeros_like(total) for k in R}
    perms = list(itertools.permutations(R))
    for order in perms:
        cur = dict(r0)
        for k in order:
            before = lam(cur); cur[k] = r1[k]; shap[k] += (lam(cur) - before) / len(perms)
    out = dict(y0=a.y0, y1=a.y1, yearling=yb, step=os.environ.get('SBX_STEP'))
    for reg in ('east', 'plains', 'west'):
        m = sb.region[reg]
        out[reg] = dict(lambda_y0=float(np.median(lam(r0)[m])), lambda_y1=float(np.median(lam(r1)[m])),
                        total_change=float(np.median(total[m])),
                        shapley_median={k: float(np.median(v[m])) for k, v in shap.items()},
                        shapley_share_of_mean={k: float(v[m].mean() / total[m].mean()) for k, v in shap.items()},
                        rates_y0={k: float(np.median(v[m])) for k, v in r0.items()},
                        rates_y1={k: float(np.median(v[m])) for k, v in r1.items()})
    (a.out / f'lambda_decomp_{a.y0}_{a.y1}.json').write_text(json.dumps(out, indent=1) + '\n')
    for reg in ('east', 'plains', 'west'):
        o = out[reg]
        print(reg, f"lambda {o['lambda_y0']:.2f} -> {o['lambda_y1']:.2f} (median change {o['total_change']:+.2f})",
              "shares of mean change:", {k: round(v, 2) for k, v in o['shapley_share_of_mean'].items()},
              "rates y0 -> y1:", {k: f"{o['rates_y0'][k]:.2f}->{o['rates_y1'][k]:.2f}" for k in o['rates_y0']})


if __name__ == '__main__':
    main()
