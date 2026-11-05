"""R9d: does the fitted model's own vital-rate change track built-up / population growth?

Per land cell, change in the fitted log Fmax, log K and Sa between two years (default 1940 -> 2010), against the change
in the R8 covariates (BUI q90 and HYDE, standardised log1p, 54 km). Pearson / Spearman by region, and the share of
the variance of the fitted change explained by the two covariate changes jointly (OLS R^2). BUI-uncovered cells are
dropped from BUI statistics. No fitting.
Usage (inside sbx_env.sh): python r9_describe.py --out DIR [--y0 1940 --y1 2010]
"""
import argparse, json, os
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from harness import Sandbox, capture, to_land
from r7_shared_calendar import bui_z, population_z


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--y0', type=int, default=1940)
    p.add_argument('--y1', type=int, default=2010)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    sb = Sandbox(os.environ['SBX_REFINED'])
    x, _ = capture(sb.data, sb.lat)
    T, N = len(sb.years), len(sb.rows)
    i0, i1 = int(np.flatnonzero(sb.years == a.y0)[0]), int(np.flatnonzero(sb.years == a.y1)[0])
    rates = {k: to_land(x[f'{k}_flat'], sb.rows, sb.cols, T, int(np.asarray(sb.data['land_mask']).shape[1]))
             for k in ('Fmax', 'K', 'Sa')}
    d_rate = dict(dlogF=np.log(rates['Fmax'][i1] / rates['Fmax'][i0]), dlogK=np.log(rates['K'][i1] / rates['K'][i0]),
                  dSa=rates['Sa'][i1] - rates['Sa'][i0])
    bzbar, bdz, _ = bui_z(sb); pzbar, pdz, _ = population_z(sb)
    covered = np.abs(bdz).sum(0) > 0
    d_cov = dict(dBUI=bdz[i1] - bdz[i0], dPOP=pdz[i1] - pdz[i0])
    out = dict(y0=a.y0, y1=a.y1)
    for reg in ('east', 'plains', 'west'):
        m = sb.region[reg] & covered & np.isfinite(d_rate['dlogF'])
        r = dict(n=int(m.sum()))
        for rk, rv in d_rate.items():
            r[f'{rk}_mean'] = float(np.mean(rv[m]))
            for ck, cv in d_cov.items():
                r[f'{rk}~{ck}_pearson'] = float(np.corrcoef(rv[m], cv[m])[0, 1])
                r[f'{rk}~{ck}_spearman'] = float(spearmanr(rv[m], cv[m]).correlation)
            X = np.c_[np.ones(m.sum()), d_cov['dBUI'][m], d_cov['dPOP'][m]]
            beta, *_ = np.linalg.lstsq(X, rv[m], rcond=None)
            res = rv[m] - X @ beta
            r[f'{rk}_R2_BUI+POP'] = float(1 - res.var() / rv[m].var()) if rv[m].var() > 0 else None
            r[f'{rk}_slope_BUI'] = float(beta[1])
        r['dBUI_mean'] = float(d_cov['dBUI'][m].mean()); r['dPOP_mean'] = float(d_cov['dPOP'][m].mean())
        out[reg] = r
    (a.out / f'r9d_{a.y0}_{a.y1}.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
